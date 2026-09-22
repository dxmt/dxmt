//! Payload索引 - 用于元数据过滤的性能优化
//!
//! payload 索引，支持：
//! - Cardinality估计（filter选择性评估）
//! - 倒排索引加速查找
//! - 智能PRE-FILTERING决策

use crate::types::{
    record::{Condition, Filter, FilterOperation},
    vector::RecordId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// The maps and sets this index keeps for itself, with a faster hasher than std's.
///
/// std's `SipHash` is chosen for resistance to deliberately colliding keys, and it pays for
/// that on every lookup. Measured on a 10M-row rebuild, before this: 8.4 s and 1.37 GB went on
/// assigning ordinals with *no fields at all*, and a four-valued field cost 5.3 s — hashing ten
/// million `u32`s into four sets.
///
/// `ahash::RandomState`, not `FxHash` or a fixed seed. The keys here are payload values and
/// record ids, which arrive from clients, and an unkeyed hash would let a client choose values
/// that all land in one bucket and turn every insert into a scan of it. `RandomState` draws its
/// keys at runtime, which keeps that protection at a fraction of the cost.
///
/// Internal only. What leaves this module — `get_matching_ids` — is still a std `HashSet`, so
/// no caller sees the change.
type FastMap<K, V> = HashMap<K, V, ahash::RandomState>;
type FastSet<T> = HashSet<T, ahash::RandomState>;

/// The key `records/list` sorts on: `created_at` in nanoseconds, newest first.
///
/// One definition for the two places that page records — the disk store's `created` column and
/// this index — so the two cannot drift into different orders. A timestamp outside the
/// nanosecond range (past 2262) sorts last rather than failing.
pub fn created_key(at: &chrono::DateTime<chrono::Utc>) -> i64 {
    at.timestamp_nanos_opt().unwrap_or(i64::MIN)
}

/// A position in the list order — newest `created_at` first, ties by id — for paging by where the
/// last page ended rather than by how many records came before it.
///
/// `offset` paging shifts under writes: the list is newest first, so every record inserted while
/// a client pages through lands in front of it and pushes the rest back, and the next page
/// repeats what the last one ended with. A cursor names a place in the order instead, and every
/// record present for the whole scroll is returned exactly once: an insert lands before the
/// cursor, a delete removes only itself, and an update keeps `created_at`. It holds the key, not
/// the record, so it survives that record being deleted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListCursor {
    pub created: i64,
    pub id: String,
}

impl ListCursor {
    /// The cursor just past a record.
    pub fn after(created: i64, id: &str) -> Self {
        Self {
            created,
            id: id.to_string(),
        }
    }

    /// Whether a record at `(created, id)` comes after this cursor in list order.
    pub fn precedes(&self, created: i64, id: &str) -> bool {
        created < self.created || (created == self.created && id > self.id.as_str())
    }

    /// Opaque to a client: sixteen hex digits of the timestamp, then the id's bytes in hex. No
    /// separator to escape, whatever the id contains.
    pub fn encode(&self) -> String {
        let mut out = format!("{:016x}", self.created as u64);
        for byte in self.id.as_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    pub fn decode(token: &str) -> Result<Self, String> {
        let bad = || format!("not a cursor this server issued: {token:?}");
        if token.len() < 16 || !token.len().is_multiple_of(2) || !token.is_ascii() {
            return Err(bad());
        }
        let created = u64::from_str_radix(&token[..16], 16).map_err(|_| bad())? as i64;
        let bytes = (16..token.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&token[i..i + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(|_| bad())?;
        let id = String::from_utf8(bytes).map_err(|_| bad())?;
        Ok(Self { created, id })
    }
}

/// A record's place in this index. Postings hold one of these instead of the record's id.
///
/// The id is a `String`, and every posting used to hold its own clone of one. A field with a
/// value per row therefore stored a full copy of every id, and a table with three such fields
/// stored three — at 10M rows, tens of millions of heap allocations to say "this row". The
/// ordinal is four bytes, the id is stored once in `by_ordinal`, and rebuilding the index
/// after a restart went from **97 s of construction** to what a `u32` costs.
type Ordinal = u32;

/// Payload索引 - 为每个字段维护倒排索引
#[derive(Debug, Clone)]
pub struct PayloadIndex {
    /// 字段名 -> 字段索引
    field_indices: FastMap<String, FieldIndex>,
    /// 总记录数
    total_records: usize,
    /// Which record each ordinal stands for, and back.
    ids: Ids,
    /// For each key, the records holding a value the index cannot key — an array or an object.
    ///
    /// `NotIn` needs to know every record that *has* the field: `Record::matches_filter` says a
    /// record lacking it does not satisfy `NotIn` (`is_some_and`), while one holding an array
    /// does. For scalar values that set is already here, as the union of the field's postings.
    /// Only these are missing from it, so only these are recorded.
    ///
    /// The first version kept a bitmap of every record per field instead, and it cost the
    /// 10M-row rebuild two seconds — 13.8 s to 15.8 s — to support one operator. `NotEqual`
    /// needs none of this: a missing field satisfies it, so it is answered against every record.
    unindexed: FastMap<String, roaring::RoaringBitmap>,
    /// Set while the index is being rebuilt from the records in one pass. See
    /// [`PayloadIndex::begin_bulk`].
    bulk: bool,
}

impl PayloadIndex {
    /// Stop merging the numeric half after every few thousand inserts, until [`Self::end_bulk`].
    ///
    /// The merge rule is written for a live table, where `pending` has to stay short because
    /// every query scans it. Rebuilding the whole index from the records is the other case
    /// entirely: nothing reads it until the last record is in, and merging on the way costs
    /// `O(rows²/4096)` — 2,441 merges over an array growing towards ten million entries. One
    /// numeric field with a value per row spent **41 s** of a 66 s rebuild there.
    ///
    /// Nothing may query the index between these two calls. The only caller builds it privately
    /// and publishes it afterwards.
    ///
    /// `expected_records` sizes the id bookkeeping up front. Grown one insert at a time the
    /// `id -> ordinal` map doubles two dozen times on the way to ten million entries, and every
    /// doubling rehashes every id already in it — each one a pointer chase to a string on the
    /// heap. It is a hint: more records than expected still work, and fewer waste only the
    /// difference.
    pub fn begin_bulk(&mut self, expected_records: usize) {
        self.bulk = true;
        self.ids.reserve(expected_records);
    }

    /// Sort what `begin_bulk` left unsorted, once per field.
    pub fn end_bulk(&mut self) {
        self.bulk = false;
        for field in self.field_indices.values_mut() {
            field.numeric.merge();
            field.distinct_count = field.value_to_ids.len() + field.numeric.distinct_count();
        }
    }

    /// This record's ordinal, assigning one if it has none.
    fn ordinal_for(&mut self, record_id: &RecordId, created: i64) -> Ordinal {
        self.ids.assign(record_id, created)
    }

    /// Give an ordinal back once nothing references it.
    fn release(&mut self, record_id: &RecordId) {
        self.ids.release(record_id);
    }

    /// Turn a set of ordinals into the ids the caller asked for.
    fn ids_of(&self, ordinals: impl IntoIterator<Item = Ordinal>) -> HashSet<RecordId> {
        ordinals
            .into_iter()
            .filter_map(|ordinal| self.ids.id_of(ordinal).map(str::to_string))
            .collect()
    }
}

/// Record ids and the ordinals standing for them, with each id stored exactly once.
///
/// This was a `Vec<Option<String>>` one way and a `HashMap<String, Ordinal>` the other, which
/// is two copies of every id and a map whose buckets are 32 bytes each. A 10M-row rebuild with
/// *no payload fields at all* spent 7.6 s and 915 MB here — more than any single field. Taken
/// apart on the same machine: cloning ten million ids costs 0.43 s, and inserting them into
/// the `String`-keyed map 2.87 s, against 0.98 s for the same map keyed by `u64`. The strings
/// in the buckets were the cost, not the hashing.
///
/// So the table holds only the ordinal — four bytes — and reads the id it stands for out of
/// `by_ordinal` when it needs to hash or compare one. `hashbrown::HashTable` is the std map's
/// own engine with the key/value split left to the caller, which is exactly that shape.
///
/// The invariant everything relies on: **an ordinal is in `table` if and only if its slot in
/// `by_ordinal` holds an id.** `assign` fills the slot before inserting, and `release` removes
/// from the table before emptying it, so the rehash closure — which may run on any insert and
/// reads every stored ordinal's id — never meets an empty slot.
#[derive(Debug, Clone, Default)]
struct Ids {
    /// Each indexed record's id, addressed by its ordinal. A removed record leaves its slot
    /// empty and the ordinal on `free`, so ids do not accumulate across churn.
    by_ordinal: Vec<Option<Box<str>>>,
    /// `id -> ordinal`, holding only the ordinal.
    table: hashbrown::HashTable<Ordinal>,
    /// Keyed per process, for the same reason as [`FastMap`]: ids come from clients.
    hasher: ahash::RandomState,
    /// Ordinals whose record was removed, ready to be handed out again.
    free: Vec<Ordinal>,
    /// Each ordinal's `created_at`, in nanoseconds — the order `records/list` pages in. See
    /// [`PayloadIndex::page_of_matching`] for why the index keeps it.
    created: Vec<i64>,
}

impl Ids {
    fn hash(hasher: &ahash::RandomState, id: &str) -> u64 {
        hasher.hash_one(id)
    }

    /// The id in a slot the table refers to. The invariant above makes this infallible.
    fn stored(by_ordinal: &[Option<Box<str>>], ordinal: Ordinal) -> &str {
        by_ordinal[ordinal as usize]
            .as_deref()
            .expect("an ordinal in the table always has an id")
    }

    fn reserve(&mut self, additional: usize) {
        let Ids {
            by_ordinal,
            table,
            hasher,
            created,
            ..
        } = self;
        by_ordinal.reserve(additional);
        created.reserve(additional);
        table.reserve(additional, |&o| {
            Self::hash(hasher, Self::stored(by_ordinal, o))
        });
    }

    fn get(&self, id: &str) -> Option<Ordinal> {
        let hash = Self::hash(&self.hasher, id);
        self.table
            .find(hash, |&o| Self::stored(&self.by_ordinal, o) == id)
            .copied()
    }

    /// The ordinal for `id`, handing out a new one if it has none. One probe either way.
    fn assign(&mut self, id: &str, created: i64) -> Ordinal {
        let hash = Self::hash(&self.hasher, id);
        let Ids {
            by_ordinal,
            table,
            hasher,
            free,
            created: created_at,
        } = self;
        match table.entry(
            hash,
            |&o| Self::stored(by_ordinal, o) == id,
            |&o| Self::hash(hasher, Self::stored(by_ordinal, o)),
        ) {
            hashbrown::hash_table::Entry::Occupied(existing) => {
                let ordinal = *existing.get();
                created_at[ordinal as usize] = created;
                ordinal
            }
            hashbrown::hash_table::Entry::Vacant(slot) => {
                // The slot is filled *before* the insert, because inserting may grow the table
                // and growing rehashes every ordinal in it — including, after this line, this
                // one. The vacant entry has already reserved room, so this insert cannot
                // rehash; filling first keeps the invariant true regardless.
                let ordinal = match free.pop() {
                    Some(reused) => {
                        by_ordinal[reused as usize] = Some(id.into());
                        created_at[reused as usize] = created;
                        reused
                    }
                    None => {
                        by_ordinal.push(Some(id.into()));
                        created_at.push(created);
                        (by_ordinal.len() - 1) as Ordinal
                    }
                };
                slot.insert(ordinal);
                ordinal
            }
        }
    }

    /// Forget `id` and put its ordinal up for reuse.
    fn release(&mut self, id: &str) -> Option<Ordinal> {
        let hash = Self::hash(&self.hasher, id);
        let Ids {
            by_ordinal,
            table,
            free,
            ..
        } = self;
        let entry = table
            .find_entry(hash, |&o| Self::stored(by_ordinal, o) == id)
            .ok()?;
        let (ordinal, _) = entry.remove();
        // Out of the table first, then out of the slot: see the invariant on the type.
        by_ordinal[ordinal as usize] = None;
        free.push(ordinal);
        Some(ordinal)
    }

    /// The id behind an ordinal, or `None` if the record has since been removed.
    fn id_of(&self, ordinal: Ordinal) -> Option<&str> {
        self.by_ordinal.get(ordinal as usize)?.as_deref()
    }

    /// Every ordinal currently standing for a record.
    fn live(&self) -> impl Iterator<Item = Ordinal> + '_ {
        self.table.iter().copied()
    }
}

/// 单个字段的索引
#[derive(Debug, Clone)]
struct FieldIndex {
    /// 值 -> 包含该值的记录ID集合。
    ///
    /// Strings, bools and nulls only. Numbers go to `numeric` instead -- see [`NumericIndex`]
    /// for what one set per distinct number costs on a field with a value per row.
    value_to_ids: FastMap<ValueKey, Postings>,
    /// Numeric values, sorted. A field can hold both: the value decides, not the field.
    numeric: NumericIndex,
    /// 字段类型（推断）
    #[allow(dead_code)]
    field_type: FieldType,
    /// 不同值的数量
    distinct_count: usize,
    /// 字段统计信息（用于更准确的基数估计）
    stats: FieldStats,
}

/// The records holding one value of a field.
///
/// This was a `HashSet<Ordinal>` per distinct value, which is wrong at both ends of the range
/// a field can have, measured on a 10M-row rebuild:
///
/// * **A value per row** — a text field, a UUID, a title. Ten million sets of one element
///   each: an allocation apiece, a hasher state apiece (16 bytes under SipHash, 32 under
///   `ahash`, which is how switching hashers made this field *bigger*), and 20.7 s of the
///   rebuild. `One` holds the ordinal inline and allocates nothing.
/// * **A handful of values** — a four-valued `topic`. Four sets of 2.5M hashed integers,
///   inserted at random positions: 4.5 s, for a field that says one of four things. A roaring
///   bitmap takes the same ordinals as appends, because a rebuild hands them out in order, and
///   is what Qdrant, tantivy and Milvus keep for exactly this.
///
/// `One` is the common case and `Many` the general one; a stored entry is never empty, because
/// removal reports when it would be and the caller drops the key instead.
#[derive(Debug, Clone)]
enum Postings {
    One(Ordinal),
    Many(roaring::RoaringBitmap),
}

impl Postings {
    fn insert(&mut self, ordinal: Ordinal) {
        match self {
            Postings::One(existing) if *existing == ordinal => {}
            Postings::One(existing) => {
                let mut many = roaring::RoaringBitmap::new();
                many.insert(*existing);
                many.insert(ordinal);
                *self = Postings::Many(many);
            }
            Postings::Many(many) => {
                many.insert(ordinal);
            }
        }
    }

    /// Take `ordinal` out, and say whether nothing is left — in which case the caller removes
    /// the whole entry, which is what keeps `distinct_count` honest.
    fn remove(&mut self, ordinal: Ordinal) -> bool {
        match self {
            Postings::One(existing) => *existing == ordinal,
            Postings::Many(many) => {
                many.remove(ordinal);
                match many.len() {
                    0 => true,
                    // Back to inline once one is left, so a value that was popular and is not
                    // any more does not keep a bitmap for a single record.
                    1 => {
                        let last = many.min().expect("len is 1");
                        *self = Postings::One(last);
                        false
                    }
                    _ => false,
                }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Postings::One(_) => 1,
            Postings::Many(many) => many.len() as usize,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn for_each(&self, mut visit: impl FnMut(Ordinal)) {
        match self {
            Postings::One(ordinal) => visit(*ordinal),
            Postings::Many(many) => many.iter().for_each(visit),
        }
    }
}

/// A field's numeric values, one entry per record, kept sorted so a range is a binary search.
///
/// `value_to_ids` keys one inverted set per *distinct* value. That is the right shape for a
/// field with a few dozen values and the wrong one for a field with a value per row. The
/// benchmark filters on the primary key, so at 10M rows it built ten million single-element
/// `HashSet`s: 2.5 GB resident to store one `i64` a row (7,149 MB against 4,659 MB with no
/// payload at all), and every range condition walked all ten million entries. Measured cost,
/// 2026-09-09: 10M rows at 1% selectivity answered at 108.8 QPS with a 236 ms serial p99,
/// against 3,682 QPS unfiltered, and the four filtered cases line up with the *distinct
/// count* rather than with the number of rows that match.
///
/// Sorted, the same range is two binary searches over a contiguous slice.
///
/// Inserts append to `pending` instead of sorting, because a read takes `&self` and cannot
/// sort, while `insert_record` takes `&mut self` and can. A query binary-searches `sorted`
/// and scans `pending`, which the merge below keeps small enough that the scan does not
/// matter.
#[derive(Debug, Clone, Default)]
struct NumericIndex {
    /// Sorted by value. Ties keep insertion order, which nothing depends on.
    sorted: Vec<(OrderedFloat, Ordinal)>,
    /// Appended since the last merge, unsorted.
    pending: Vec<(OrderedFloat, Ordinal)>,
    /// Distinct values across both halves, recomputed at each merge.
    distinct: usize,
}

impl NumericIndex {
    /// Merge when `pending` could make a scan of it cost as much as the binary search saved.
    /// A flat floor as well as a fraction, so a small index does not merge on every insert.
    const MERGE_AT: usize = 4_096;

    fn insert(&mut self, value: OrderedFloat, id: Ordinal) {
        self.pending.push((value, id));
        if self.pending.len() >= Self::MERGE_AT
            || self.pending.len().saturating_mul(64) > self.sorted.len()
        {
            self.merge();
        }
    }

    /// Append without considering a merge. See [`PayloadIndex::begin_bulk`].
    fn push(&mut self, value: OrderedFloat, id: Ordinal) {
        self.pending.push((value, id));
    }

    fn remove(&mut self, value: OrderedFloat, id: Ordinal) {
        if let Some(at) = self
            .pending
            .iter()
            .position(|(v, i)| *v == value && *i == id)
        {
            self.pending.swap_remove(at);
            return;
        }
        // In the sorted half the value locates the run and the id picks the entry out of it.
        let lo = self.sorted.partition_point(|(v, _)| *v < value);
        let mut at = lo;
        while at < self.sorted.len() && self.sorted[at].0 == value {
            if self.sorted[at].1 == id {
                self.sorted.remove(at);
                return;
            }
            at += 1;
        }
    }

    fn merge(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        // Append and re-sort, rather than merging two runs into a fresh buffer. The obvious
        // linear merge measured **three times slower** on a 10M-row field: it allocates and
        // copies the whole array every time, where this reuses the buffer and hands pdqsort an
        // array that is already sorted except for a short tail, which it recognises. The cost
        // that matters is not one merge but how many, and `insert` is what decides that.
        self.sorted.append(&mut self.pending);
        self.sorted.sort_unstable_by_key(|(value, _)| *value);
        self.distinct = self.sorted.windows(2).filter(|w| w[0].0 != w[1].0).count()
            + usize::from(!self.sorted.is_empty());
    }

    /// Distinct values, exact after a merge and an over-estimate before one.
    fn distinct_count(&self) -> usize {
        self.distinct + self.pending.len()
    }

    /// Whether any value in the bounded range satisfies `matches`, stopping at the first one.
    ///
    /// [`Self::select`] visits the whole range because its caller wants every id. This caller
    /// wants one bit, and the difference is not small: a `>= threshold` that 99% of rows pass
    /// bounds a slice of 9.9 million entries, and walking it to answer "yes" cost a filtered
    /// search 52.9 ms a query where the short-circuiting version it replaced cost 10.5 ms.
    fn any_matches(
        &self,
        lo: Option<OrderedFloat>,
        hi: Option<OrderedFloat>,
        matches: impl Fn(OrderedFloat) -> bool,
    ) -> bool {
        let start = match lo {
            Some(v) => self.sorted.partition_point(|(k, _)| *k < v),
            None => 0,
        };
        let end = match hi {
            Some(v) => self.sorted.partition_point(|(k, _)| *k <= v),
            None => self.sorted.len(),
        };
        // `start > end` when the bounds contradict each other — `n >= 40` with `n <= 5` on the
        // same field. Slicing with them panicked, and `cannot_match_anything`, which every
        // filtered search calls first, folds a field's ranges into exactly these two bounds, so
        // one contradictory filter took the request down. Nothing lies between them.
        self.sorted[start..end.max(start)]
            .iter()
            .any(|(value, _)| matches(*value))
            || self.pending.iter().any(|(value, _)| matches(*value))
    }

    /// Every id whose value satisfies `matches`, without visiting values that cannot.
    ///
    /// `lo`/`hi` bound the sorted half; `matches` filters within it and decides the whole of
    /// `pending`. A caller that cannot express its condition as bounds passes the full range
    /// and pays a scan, which is still no worse than the map it replaced.
    fn select(
        &self,
        lo: Option<OrderedFloat>,
        hi: Option<OrderedFloat>,
        matches: impl Fn(OrderedFloat) -> bool,
        mut visit: impl FnMut(Ordinal),
    ) {
        let start = match lo {
            Some(v) => self.sorted.partition_point(|(k, _)| *k < v),
            None => 0,
        };
        let end = match hi {
            Some(v) => self.sorted.partition_point(|(k, _)| *k <= v),
            None => self.sorted.len(),
        };
        // Contradictory bounds are an empty range, not a panic; see `any_matches`.
        for (value, id) in &self.sorted[start..end.max(start)] {
            if matches(*value) {
                visit(*id);
            }
        }
        for (value, id) in &self.pending {
            if matches(*value) {
                visit(*id);
            }
        }
    }
}

/// 字段统计信息
#[derive(Debug, Clone)]
struct FieldStats {
    /// 数值字段的最小值
    numeric_min: Option<f64>,
    /// 数值字段的最大值
    numeric_max: Option<f64>,
    /// 字符串前缀集合（用于StartsWith估计，最多保留1000个前缀）
    string_prefixes: HashSet<String>,
    /// 总采样数（用于估计）
    sample_count: usize,
}

/// 值的键（用于HashMap）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueKey {
    String(String),
    Number(i64),         // 整数
    Float(OrderedFloat), // 浮点数（需要特殊处理）
    Bool(bool),
    Null,
}

/// A number as the index keys it: every `f64`, exactly, in numeric order.
///
/// This was `(f * 1_000_000.0) as i64` — six decimal places, truncated, and saturating at
/// ±9.2e12. Both halves of that were wrong for real payloads. A microsecond timestamp is about
/// 1.7e15, so every one of them became `i64::MAX` and `Equal` on a timestamp matched every
/// record that had one — 426 of 426 comparisons against `Record::matches_filter` disagreed, and
/// `GreaterThan` 318 of 398. Below a millionth, values merged the same way.
///
/// The encoding is the standard one: a non-negative float's bits with the sign bit set, a
/// negative float's bits inverted, so unsigned order is numeric order and nothing is lost.
/// `-0.0` is folded into `0.0` first, because they are one number and have to be one key.
/// JSON cannot carry NaN or an infinity, so there is no ordering question about those.
///
/// Integers above 2^53 are keyed as the nearest `f64`, which is what `matches_filter` compares
/// them as too — agreement matters more here than a precision JSON clients mostly cannot
/// represent anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct OrderedFloat(u64);

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        let f = if f == 0.0 { 0.0 } else { f };
        let bits = f.to_bits();
        OrderedFloat(if bits >> 63 == 0 {
            bits | (1 << 63)
        } else {
            !bits
        })
    }
}

impl From<OrderedFloat> for f64 {
    fn from(of: OrderedFloat) -> Self {
        let bits = of.0;
        f64::from_bits(if bits >> 63 == 1 {
            bits & !(1 << 63)
        } else {
            !bits
        })
    }
}

/// 字段类型（推断）
#[derive(Debug, Clone, PartialEq)]
enum FieldType {
    String,
    Number,
    Boolean,
    Mixed, // 混合类型
}

impl FieldStats {
    fn new() -> Self {
        Self {
            numeric_min: None,
            numeric_max: None,
            string_prefixes: HashSet::new(),
            sample_count: 0,
        }
    }

    /// 更新数值统计
    fn update_numeric(&mut self, value: f64) {
        self.numeric_min = Some(self.numeric_min.map_or(value, |min| min.min(value)));
        self.numeric_max = Some(self.numeric_max.map_or(value, |max| max.max(value)));
        self.sample_count += 1;
    }

    /// 更新字符串统计（收集前缀）
    /// The prefixes `StartsWith` estimates from, sampled rather than collected.
    ///
    /// The cap used to be on the number of *distinct* prefixes, which a low-cardinality field
    /// never reaches — a field with four values kept sampling for ever, andeach record paid a
    /// `chars().count()` over the whole value plus two `String` allocations and two hash
    /// inserts to re-learn the same eight prefixes. At 10M rows that was most of the cost of
    /// building the index. The first `PREFIX_SAMPLE` records describe the field's prefixes as
    /// well as all of them do, and the estimate is already an estimate.
    const PREFIX_SAMPLE: usize = 10_000;

    fn update_string(&mut self, value: &str) {
        self.sample_count += 1;
        // 收集2-4字符的前缀用于StartsWith估计
        // IMPORTANT: Use .chars() to handle multi-byte UTF-8 characters correctly
        if self.sample_count <= Self::PREFIX_SAMPLE && self.string_prefixes.len() < 1000 {
            // Get first 2 characters (by character count, not bytes)
            let char_count = value.chars().count();
            if char_count >= 2 {
                let prefix: String = value.chars().take(2).collect();
                self.string_prefixes.insert(prefix);
            }
            // Get first 3 characters (by character count, not bytes)
            if char_count >= 3 {
                let prefix: String = value.chars().take(3).collect();
                self.string_prefixes.insert(prefix);
            }
        }
    }

    /// 估计范围查询的选择性
    fn estimate_range_selectivity(&self, op: &FilterOperation) -> f64 {
        match (self.numeric_min, self.numeric_max) {
            (Some(min), Some(max)) if max > min => {
                let range = max - min;
                match op {
                    FilterOperation::GreaterThan(v) | FilterOperation::GreaterThanOrEqual(v) => {
                        if let Some(threshold) = Self::extract_numeric(v) {
                            if threshold <= min {
                                return 1.0; // 所有值都满足
                            } else if threshold >= max {
                                return 0.01; // 几乎没有值满足
                            } else {
                                return ((max - threshold) / range).max(0.01);
                            }
                        }
                    }
                    FilterOperation::LessThan(v) | FilterOperation::LessThanOrEqual(v) => {
                        if let Some(threshold) = Self::extract_numeric(v) {
                            if threshold >= max {
                                return 1.0; // 所有值都满足
                            } else if threshold <= min {
                                return 0.01; // 几乎没有值满足
                            } else {
                                return ((threshold - min) / range).max(0.01);
                            }
                        }
                    }
                    _ => {}
                }
                0.3 // 默认保守估计
            }
            _ => 0.3, // 没有统计信息时使用保守估计
        }
    }

    /// 估计字符串匹配的选择性
    fn estimate_string_selectivity(&self, op: &FilterOperation) -> f64 {
        match op {
            FilterOperation::StartsWith(prefix) => {
                if self.string_prefixes.is_empty() || self.sample_count == 0 {
                    return 0.2; // 默认估计
                }
                // 计算有多少前缀匹配
                let prefix_lower = prefix.to_lowercase();
                let matching = self
                    .string_prefixes
                    .iter()
                    .filter(|p| {
                        prefix_lower.starts_with(&p.to_lowercase())
                            || p.to_lowercase().starts_with(&prefix_lower)
                    })
                    .count();

                if matching == 0 {
                    return 0.01; // 没有匹配的前缀
                }

                // 基于匹配前缀比例估计
                let ratio = matching as f64 / self.string_prefixes.len() as f64;
                ratio.clamp(0.01, 0.5)
            }
            FilterOperation::Contains(_) => {
                // Contains更难估计，使用保守值
                0.25
            }
            FilterOperation::EndsWith(_) => {
                // EndsWith也使用保守值（需要后缀索引才能更准确）
                0.15
            }
            _ => 0.2,
        }
    }

    /// 从JSON值提取数值
    fn extract_numeric(value: &serde_json::Value) -> Option<f64> {
        match value {
            serde_json::Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }
}

/// Cardinality估计结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardinalityEstimate {
    /// 预期匹配的记录数（最小值）
    pub min: usize,
    /// 预期匹配的记录数（最大值）
    pub max: usize,
    /// 预期匹配的记录数（期望值）
    pub exp: usize,
    /// 选择性（0.0-1.0，越小越selective）
    pub selectivity: f64,
}

impl PayloadIndex {
    /// 创建新的PayloadIndex
    pub fn new() -> Self {
        Self {
            field_indices: FastMap::default(),
            total_records: 0,
            ids: Ids::default(),
            unindexed: FastMap::default(),
            bulk: false,
        }
    }

    /// Index a record.
    ///
    /// `created` is the record's `created_at` in nanoseconds — `records/list` pages newest
    /// first, and [`Self::page_of_matching`] can only choose a page without resolving every
    /// match to its record if the index knows the order. It is a parameter rather than a
    /// default so that a caller cannot leave it out and get a list in the wrong order.
    pub fn insert_record(
        &mut self,
        record_id: &RecordId,
        payload: &serde_json::Value,
        created: i64,
    ) {
        self.total_records += 1;

        // Every record gets an ordinal, including one whose payload is not an object. It has no
        // fields, but `matches_filter` still answers for it — `NotEqual` is true, and `id`
        // conditions read its own id — so the index has to know it exists.
        let ordinal = self.ordinal_for(record_id, created);
        if let Some(obj) = payload.as_object() {
            for (key, value) in obj {
                self.index_field(key, value, ordinal);
            }
        }
    }

    /// Re-index a record whose payload changed. `created` is still its original `created_at`:
    /// an update does not move a record in the list.
    pub fn update_record(
        &mut self,
        record_id: &RecordId,
        old_payload: &serde_json::Value,
        new_payload: &serde_json::Value,
        created: i64,
    ) {
        self.remove_record(record_id, old_payload);
        self.insert_record(record_id, new_payload, created);
    }

    /// 从索引中删除记录
    pub fn remove_record(&mut self, record_id: &RecordId, payload: &serde_json::Value) {
        if self.total_records > 0 {
            self.total_records -= 1;
        }
        let Some(ordinal) = self.ids.get(record_id) else {
            return;
        };

        if let Some(obj) = payload.as_object() {
            for (key, value) in obj {
                self.unindex_field(key, value, ordinal);
            }
        }

        // The ordinal goes back on the free list only now, after every posting naming it has
        // been taken out above. That is an invariant worth stating, because it is what the
        // move from ids to ordinals put at risk: a posting left behind used to name a deleted
        // *id*, which `ids_of` drops, and would now name an ordinal a later record may be
        // given — a wrong record rather than no record. The loop above visits exactly the
        // fields of the payload the caller passed, so the caller has to pass the payload the
        // record was indexed with. `delete_records` and `update_record` both do;
        // `a_reused_ordinal_does_not_carry_the_previous_record_s_values` is the test.
        self.release(record_id);
    }

    /// 估计filter的cardinality
    pub fn estimate_cardinality(&self, filter: &Filter) -> CardinalityEstimate {
        // 处理must条件（AND）
        let must_estimate = if let Some(conditions) = &filter.must {
            self.estimate_conditions_and(conditions)
        } else {
            CardinalityEstimate {
                min: self.total_records,
                max: self.total_records,
                exp: self.total_records,
                selectivity: 1.0,
            }
        };

        // 处理must_not条件（NOT）
        let must_not_estimate = if let Some(conditions) = &filter.must_not {
            let pos_estimate = self.estimate_conditions_or(conditions);
            CardinalityEstimate {
                min: self.total_records.saturating_sub(pos_estimate.max),
                max: self.total_records.saturating_sub(pos_estimate.min),
                exp: self.total_records.saturating_sub(pos_estimate.exp),
                selectivity: 1.0 - pos_estimate.selectivity,
            }
        } else {
            // No must_not conditions - return full selectivity
            CardinalityEstimate {
                min: self.total_records,
                max: self.total_records,
                exp: self.total_records,
                selectivity: 1.0,
            }
        };

        // 处理should条件（OR）
        let should_estimate = if let Some(conditions) = &filter.should {
            if conditions.is_empty() {
                must_estimate.clone()
            } else {
                self.estimate_conditions_or(conditions)
            }
        } else {
            must_estimate.clone()
        };

        // 组合must和must_not（取交集，近似为相乘）
        let combined_min = must_estimate.min.min(must_not_estimate.min);
        let combined_max = must_estimate.max.min(must_not_estimate.max);
        let combined_exp = (must_estimate.exp as f64 * must_not_estimate.selectivity) as usize;

        // 如果有should条件，取并集（取最大值）
        let final_estimate =
            if filter.should.is_some() && !filter.should.as_ref().unwrap().is_empty() {
                CardinalityEstimate {
                    min: combined_min.max(should_estimate.min),
                    max: combined_max.max(should_estimate.max),
                    exp: combined_exp.max(should_estimate.exp),
                    selectivity: (combined_exp.max(should_estimate.exp) as f64
                        / self.total_records.max(1) as f64),
                }
            } else {
                CardinalityEstimate {
                    min: combined_min,
                    max: combined_max,
                    exp: combined_exp,
                    selectivity: combined_exp as f64 / self.total_records.max(1) as f64,
                }
            };

        final_estimate
    }

    /// 获取满足filter的记录ID集合（用于PRE-FILTERING）
    /// `true` when no record can satisfy this filter, decided from the index alone and
    /// without building an id set.
    ///
    /// A `must` condition matching none of its field's indexed values matches no record, and
    /// one failing `must` empties the whole filter whatever else it carries: `should` only
    /// adds a requirement, `must_not` only removes rows. Walking the distinct values costs the
    /// field's cardinality and clones nothing, where [`Self::get_matching_ids`] materialises
    /// every id of every condition before intersecting -- on a 20,000-row table a `<= 600`
    /// accepting 18,000 of them built an 18,000-string set to answer a filter whose result is
    /// empty, which is how the first version of this shortcut came out slower than the scan it
    /// replaced.
    ///
    /// Only conditions whose matches are all indexed can prove anything here; see
    /// [`Self::condition_is_exact`]. `id` is refused because
    /// [`crate::types::record::Record::matches_filter`] reads the record's own id, while this
    /// index only ever knows a payload field of that name.
    pub fn cannot_match_anything(&self, filter: &Filter) -> bool {
        let Some(must) = &filter.must else {
            return false;
        };
        // Grouped by key, because conditions on the same key all judge the same value and so
        // have to be tested together. `bucket >= 400` and `bucket <= 600` each accept plenty
        // of rows on their own; what is empty is their intersection, and it is empty exactly
        // when no indexed value falls between them.
        let mut by_key: HashMap<&str, Vec<&Condition>> = HashMap::new();
        for condition in must {
            // A condition that cannot prove anything is left out rather than refusing the
            // whole filter: proving that a subset of the `must` is unsatisfiable proves the
            // whole `must` is.
            if condition.key == "id" || !Self::condition_is_exact(condition) {
                continue;
            }
            by_key
                .entry(condition.key.as_str())
                .or_default()
                .push(condition);
        }
        by_key.into_iter().any(|(key, conditions)| {
            match self.field_indices.get(key) {
                // No record carries a scalar under this key, so none can match a condition
                // that needs one.
                None => true,
                // A record holding an array can satisfy `>= 400` with one element and `<= 600`
                // with another — each condition asks whether *any* element passes — so on such
                // a field no single value has to satisfy them all, and the joint proof below
                // would call `[100, 700]` unmatchable. Each condition is proved on its own
                // instead: one that nothing satisfies still empties the whole `must`.
                Some(field) if self.unindexed.contains_key(key) => conditions
                    .iter()
                    .any(|c| Self::no_value_satisfies(field, std::slice::from_ref(c))),
                Some(field) => Self::no_value_satisfies(field, &conditions),
            }
        })
    }

    /// Whether no indexed value of the field satisfies every one of `conditions` at once.
    fn no_value_satisfies(field: &FieldIndex, conditions: &[&Condition]) -> bool {
        let satisfied = |value_key: &ValueKey| {
            conditions
                .iter()
                .all(|c| Self::value_key_matches(value_key, &c.operation))
        };
        // Both halves have to be asked. Numbers live in the sorted index and
        // strings in the map, and a field can hold both; asking only the map is
        // how the first version of this decided a satisfiable numeric filter was
        // empty, which returns no rows rather than an error.
        let in_map = field
            .value_to_ids
            .iter()
            .any(|(value_key, ids)| !ids.is_empty() && satisfied(value_key));
        if in_map {
            return false;
        }
        let (lo, hi) = conditions
            .iter()
            .map(|c| Self::numeric_bounds(&c.operation))
            .fold((None, None), |(lo, hi), (l, h)| {
                (Self::tighter(lo, l, true), Self::tighter(hi, h, false))
            });
        // `any_matches`, not `select`: this asks whether anything matches, and
        // every filtered query asks it. Collecting the answer instead of
        // short-circuiting it walks the whole matching range, which for a filter
        // most rows pass is most of the table.
        !field
            .numeric
            .any_matches(lo, hi, |value| satisfied(&Self::from_ordered(value)))
    }

    /// The tighter of two optional bounds: the larger for a lower bound, the smaller for an
    /// upper one. Conditions on one key all have to hold at once, so their bounds intersect.
    fn tighter(
        a: Option<OrderedFloat>,
        b: Option<OrderedFloat>,
        lower: bool,
    ) -> Option<OrderedFloat> {
        match (a, b) {
            (Some(x), Some(y)) => Some(if lower { x.max(y) } else { x.min(y) }),
            (some, None) | (None, some) => some,
        }
    }

    /// Whether every record this condition accepts carries a value the index holds.
    ///
    /// This index keys the scalars [`Self::to_value_key`] can encode, so a record whose value
    /// for the key is an array or an object is absent from it. That only matters for a
    /// condition such a record could satisfy -- and most cannot: a range needs `as_f64`, a
    /// substring test needs `as_str`, an equality against a scalar needs a scalar. `NotEqual`
    /// and `NotIn` are the two that can, because an array is not equal to 3, and they are what
    /// makes this index an approximation of `matches_filter` rather than a restatement of it.
    fn condition_is_exact(condition: &Condition) -> bool {
        match &condition.operation {
            FilterOperation::Equal(value) => Self::to_value_key(value).is_some(),
            FilterOperation::In(values) => values.iter().all(|v| Self::to_value_key(v).is_some()),
            FilterOperation::GreaterThan(target)
            | FilterOperation::GreaterThanOrEqual(target)
            | FilterOperation::LessThan(target)
            | FilterOperation::LessThanOrEqual(target) => target.as_f64().is_some(),
            FilterOperation::Contains(_)
            | FilterOperation::StartsWith(_)
            | FilterOperation::EndsWith(_) => true,
            FilterOperation::NotEqual(_) | FilterOperation::NotIn(_) => false,
        }
    }

    /// Whether one indexed value satisfies the condition.
    ///
    /// The single place the comparison lives, so that asking "does anything match?" and asking
    /// "which records match?" cannot drift apart.
    fn value_key_matches(value_key: &ValueKey, operation: &FilterOperation) -> bool {
        match operation {
            FilterOperation::Equal(value) => {
                Self::to_value_key(value).is_some_and(|k| Self::same_value(&k, value_key))
            }
            FilterOperation::NotEqual(value) => {
                Self::to_value_key(value).is_some_and(|k| !Self::same_value(&k, value_key))
            }
            FilterOperation::In(values) => values
                .iter()
                .filter_map(Self::to_value_key)
                .any(|k| Self::same_value(&k, value_key)),
            // An unencodable target is dropped from the exclusion set rather than excluding
            // everything, which is what the id-building version does too.
            FilterOperation::NotIn(values) => !values
                .iter()
                .filter_map(Self::to_value_key)
                .any(|k| Self::same_value(&k, value_key)),
            FilterOperation::GreaterThan(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v > t),
            FilterOperation::GreaterThanOrEqual(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v >= t),
            FilterOperation::LessThan(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v < t),
            FilterOperation::LessThanOrEqual(target) => Self::as_number(value_key)
                .zip(target.as_f64())
                .is_some_and(|(v, t)| v <= t),
            FilterOperation::Contains(substr) => {
                matches!(value_key, ValueKey::String(s) if s.contains(substr.as_str()))
            }
            FilterOperation::StartsWith(prefix) => {
                matches!(value_key, ValueKey::String(s) if s.starts_with(prefix.as_str()))
            }
            FilterOperation::EndsWith(suffix) => {
                matches!(value_key, ValueKey::String(s) if s.ends_with(suffix.as_str()))
            }
        }
    }

    /// Equality between two indexed values, numerically where both are numbers.
    ///
    /// `ValueKey` splits integers and floats into different variants, so deriving equality
    /// makes `Number(5)` differ from `Float(5.0)`. That never mattered while every lookup was
    /// a hash of the exact variant the value was stored under. It matters now: the sorted
    /// numeric index stores one scaled key for both, so an `Equal` on an integer arrives here
    /// as a `Float` and would match nothing at all.
    fn same_value(a: &ValueKey, b: &ValueKey) -> bool {
        match (Self::as_ordered(a), Self::as_ordered(b)) {
            (Some(x), Some(y)) => x == y,
            _ => a == b,
        }
    }

    /// The sort key a numeric value is stored under, or `None` for anything that is not a
    /// number. This is what decides whether a value lands in `numeric` or in `value_to_ids`,
    /// on insert and on removal alike, so both paths ask exactly this question.
    fn as_ordered(value_key: &ValueKey) -> Option<OrderedFloat> {
        match value_key {
            ValueKey::Number(v) => Some(OrderedFloat::from(*v as f64)),
            ValueKey::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The number an indexed value compares as, for the range operations.
    fn as_number(value_key: &ValueKey) -> Option<f64> {
        match value_key {
            ValueKey::Number(v) => Some(*v as f64),
            ValueKey::Float(f) => Some((*f).into()),
            _ => None,
        }
    }

    pub fn get_matching_ids(&self, filter: &Filter) -> HashSet<RecordId> {
        self.ids_of(self.matching_ordinals(filter))
    }

    /// One page of the records a filter matches — newest `created_at` first, ties by id — and
    /// how many match in all.
    ///
    /// `records/list` used to take [`Self::get_matching_ids`] and look every id up in the
    /// record store to sort by `created_at`. For a filter matching 2.5M of 10M records that was
    /// 2.9 s a page, and taken apart it was almost none of it filtering: 0.1 s to find the
    /// matching ordinals, then 1.9 s turning each into an owned `String`, 1.4 s hashing each one
    /// back to a row to read its timestamp, and 0.6 s freeing the strings — to return five.
    ///
    /// With the timestamp beside the ordinal, the page is chosen over integers and only the
    /// records on it are resolved to ids. The order is the one `list_records` uses on every
    /// other path, `created_at` descending then id ascending, and the id tiebreak is what makes
    /// a page well defined when two records share a timestamp.
    pub fn page_of_matching(
        &self,
        filter: &Filter,
        offset: usize,
        limit: usize,
    ) -> (Vec<&str>, usize) {
        let (page, total) = self.page_matching(filter, None, offset, limit);
        (page.into_iter().map(|(id, _)| id).collect(), total)
    }

    /// The same, starting after `cursor` rather than at an offset, with each id's timestamp so
    /// the caller can say where the page ended. `total` still counts every match.
    pub fn page_of_matching_after(
        &self,
        filter: &Filter,
        cursor: Option<&ListCursor>,
        limit: usize,
    ) -> (Vec<(&str, i64)>, usize) {
        self.page_matching(filter, cursor, 0, limit)
    }

    fn page_matching(
        &self,
        filter: &Filter,
        cursor: Option<&ListCursor>,
        offset: usize,
        limit: usize,
    ) -> (Vec<(&str, i64)>, usize) {
        let matching = self.matching_ordinals(filter);
        let total = matching.len();
        let id = |o: Ordinal| self.ids.id_of(o).unwrap_or("");
        let mut keyed: Vec<(i64, Ordinal)> = matching
            .into_iter()
            .map(|o| (self.ids.created[o as usize], o))
            .filter(|&(created, o)| cursor.is_none_or(|c| c.precedes(created, id(o))))
            .collect();
        if offset >= keyed.len() || limit == 0 {
            return (Vec::new(), total);
        }
        let end = (offset + limit).min(keyed.len());
        let newest_first = |a: &(i64, Ordinal), b: &(i64, Ordinal)| {
            b.0.cmp(&a.0).then_with(|| id(a.1).cmp(id(b.1)))
        };
        if end < keyed.len() {
            keyed.select_nth_unstable_by(end - 1, newest_first);
        }
        keyed[..end].sort_by(newest_first);
        let page = keyed[offset..end]
            .iter()
            .filter_map(|&(created, o)| self.ids.id_of(o).map(|id| (id, created)))
            .collect();
        (page, total)
    }

    /// The same answer as [`Self::get_matching_ids`], before the ids are materialised.
    ///
    /// This has to agree with `Record::matches_filter`, which is the other implementation of
    /// "does this record match" — the one a filtered search falls back to when it walks the
    /// graph instead of consulting the index. It did not, in three ways, and each one was a
    /// silently wrong answer that depended on which endpoint you asked:
    ///
    ///   * **`must_not` on its own matched nothing.** The result set started empty, and with no
    ///     `must` to seed it there was nothing for the exclusions to remove. `records/list` and
    ///     the graph endpoint answered `0`; a search over the same filter answered correctly,
    ///     because it post-filters.
    ///   * **`should` was ignored whenever `must` was present**, where `matches_filter` requires
    ///     at least one `should` to match as well. On a live table, `must` in
    ///     (gardening, spaceflight) with `should` gardening returned 5,000 records through the
    ///     index and 2,500 through the search path.
    ///   * **An empty `must: []` matched nothing** rather than everything, which is what a
    ///     conjunction of no conditions means and what `matches_filter` says.
    ///
    /// So: `must` intersects, `should` intersects a union, neither constrains anything when
    /// absent or empty, and if nothing positive was named at all then every indexed record is a
    /// candidate for `must_not` to subtract from.
    fn matching_ordinals(&self, filter: &Filter) -> FastSet<Ordinal> {
        // `None` means "nothing has narrowed this yet", which is not the same as an empty set.
        let mut narrowed: Option<FastSet<Ordinal>> = None;

        if let Some(conditions) = &filter.must {
            for condition in conditions {
                let ids = self.get_condition_ids(condition);
                narrowed = Some(match narrowed {
                    None => ids,
                    Some(current) => current.intersection(&ids).copied().collect(),
                });
                if narrowed.as_ref().is_some_and(HashSet::is_empty) {
                    return FastSet::default();
                }
            }
        }

        if let Some(conditions) = &filter.should {
            if !conditions.is_empty() {
                let mut any = FastSet::default();
                for condition in conditions {
                    any.extend(self.get_condition_ids(condition));
                }
                narrowed = Some(match narrowed {
                    None => any,
                    Some(current) => current.intersection(&any).copied().collect(),
                });
            }
        }

        let mut result = narrowed.unwrap_or_else(|| self.ids.live().collect());

        if let Some(conditions) = &filter.must_not {
            for condition in conditions {
                let excluded = self.get_condition_ids(condition);
                result.retain(|id| !excluded.contains(id));
                if result.is_empty() {
                    break;
                }
            }
        }

        result
    }

    /// 估计多个条件的AND组合
    fn estimate_conditions_and(&self, conditions: &[Condition]) -> CardinalityEstimate {
        if conditions.is_empty() {
            return CardinalityEstimate {
                min: self.total_records,
                max: self.total_records,
                exp: self.total_records,
                selectivity: 1.0,
            };
        }

        // AND条件：选择性相乘
        let mut combined_selectivity = 1.0;
        let mut min_count = self.total_records;
        let mut max_count = self.total_records;

        for condition in conditions {
            let estimate = self.estimate_single_condition(condition);
            combined_selectivity *= estimate.selectivity;
            min_count = min_count.min(estimate.min);
            max_count = max_count.min(estimate.max);
        }

        CardinalityEstimate {
            min: min_count,
            max: max_count,
            exp: (self.total_records as f64 * combined_selectivity) as usize,
            selectivity: combined_selectivity,
        }
    }

    /// 估计多个条件的OR组合
    fn estimate_conditions_or(&self, conditions: &[Condition]) -> CardinalityEstimate {
        if conditions.is_empty() {
            return CardinalityEstimate {
                min: 0,
                max: 0,
                exp: 0,
                selectivity: 0.0,
            };
        }

        // OR条件：选择性相加（但不超过1.0）
        let mut combined_selectivity = 0.0;
        let mut total_min = 0;
        let mut total_max = 0;

        for condition in conditions {
            let estimate = self.estimate_single_condition(condition);
            combined_selectivity = (combined_selectivity + estimate.selectivity).min(1.0);
            total_min += estimate.min;
            total_max += estimate.max;
        }

        // OR的max不能超过总记录数
        total_max = total_max.min(self.total_records);
        total_min = total_min.min(self.total_records);

        CardinalityEstimate {
            min: total_min,
            max: total_max,
            exp: (self.total_records as f64 * combined_selectivity) as usize,
            selectivity: combined_selectivity,
        }
    }

    /// 估计单个条件的选择性
    fn estimate_single_condition(&self, condition: &Condition) -> CardinalityEstimate {
        let field_index = match self.field_indices.get(&condition.key) {
            Some(idx) => idx,
            None => {
                // No record carries this field, so nothing can match it. The comment here
                // used to call 0.5 a conservative estimate; it is the opposite. The index
                // holds every field any record has written, so a missing key is not missing
                // information -- it is the answer.
                //
                // What the old estimate cost: a filter on an absent field reported half the
                // table as matching, the search took that for a broad filter and ran post
                // filtering over 600 fetched candidates, and one such query on a 5M x 1536
                // table took 16.6 s to return nothing while holding the table read lock for
                // 4.6 s. The same query without a filter takes 3.6 ms
                // (design/DISK_INDEX.md 5.5.1).
                return CardinalityEstimate {
                    min: 0,
                    max: 0,
                    exp: 0,
                    selectivity: 0.0,
                };
            }
        };

        match &condition.operation {
            FilterOperation::Equal(value) => {
                if let Some(value_key) = Self::to_value_key(value) {
                    // Both halves: a number is in the sorted index and everything else in the
                    // map. Counting only the map reads every numeric equality as "no such
                    // value", which is a selectivity of zero and the wrong branch downstream.
                    let count = match Self::as_ordered(&value_key) {
                        Some(number) => {
                            let mut n = 0usize;
                            field_index.numeric.select(
                                Some(number),
                                Some(number),
                                |v| v == number,
                                |_| n += 1,
                            );
                            n
                        }
                        None => field_index
                            .value_to_ids
                            .get(&value_key)
                            .map(|ids| ids.len())
                            .unwrap_or(0),
                    };
                    if count > 0 {
                        return CardinalityEstimate {
                            min: count,
                            max: count,
                            exp: count,
                            selectivity: count as f64 / self.total_records.max(1) as f64,
                        };
                    }
                }
                // 值不存在
                CardinalityEstimate {
                    min: 0,
                    max: 0,
                    exp: 0,
                    selectivity: 0.0,
                }
            }
            FilterOperation::NotEqual(value) => {
                let equal_estimate = self.estimate_single_condition(&Condition {
                    key: condition.key.clone(),
                    operation: FilterOperation::Equal(value.clone()),
                });
                CardinalityEstimate {
                    min: self.total_records.saturating_sub(equal_estimate.max),
                    max: self.total_records.saturating_sub(equal_estimate.min),
                    exp: self.total_records.saturating_sub(equal_estimate.exp),
                    selectivity: 1.0 - equal_estimate.selectivity,
                }
            }
            FilterOperation::In(values) => {
                let mut total_count = 0;
                for value in values {
                    if let Some(value_key) = Self::to_value_key(value) {
                        match Self::as_ordered(&value_key) {
                            Some(number) => field_index.numeric.select(
                                Some(number),
                                Some(number),
                                |v| v == number,
                                |_| total_count += 1,
                            ),
                            None => {
                                if let Some(ids) = field_index.value_to_ids.get(&value_key) {
                                    total_count += ids.len();
                                }
                            }
                        }
                    }
                }
                // In操作可能有重复ID，实际count可能更少
                total_count = total_count.min(self.total_records);
                CardinalityEstimate {
                    min: total_count.saturating_sub(values.len().saturating_sub(1)),
                    max: total_count,
                    exp: total_count,
                    selectivity: total_count as f64 / self.total_records.max(1) as f64,
                }
            }
            FilterOperation::GreaterThan(_)
            | FilterOperation::LessThan(_)
            | FilterOperation::GreaterThanOrEqual(_)
            | FilterOperation::LessThanOrEqual(_) => {
                // 范围查询：使用统计信息进行智能估计
                let selectivity = if let Some(field_index) = self.field_indices.get(&condition.key)
                {
                    field_index
                        .stats
                        .estimate_range_selectivity(&condition.operation)
                } else {
                    0.3 // 字段不存在，使用保守估计
                };
                let estimate = (self.total_records as f64 * selectivity) as usize;
                CardinalityEstimate {
                    min: 0,
                    max: self.total_records,
                    exp: estimate,
                    selectivity,
                }
            }
            FilterOperation::Contains(_)
            | FilterOperation::StartsWith(_)
            | FilterOperation::EndsWith(_) => {
                // 字符串匹配：使用统计信息进行智能估计
                let selectivity = if let Some(field_index) = self.field_indices.get(&condition.key)
                {
                    field_index
                        .stats
                        .estimate_string_selectivity(&condition.operation)
                } else {
                    0.2 // 字段不存在，使用保守估计
                };
                let estimate = (self.total_records as f64 * selectivity) as usize;
                CardinalityEstimate {
                    min: 0,
                    max: self.total_records,
                    exp: estimate,
                    selectivity,
                }
            }
            _ => {
                // 其他操作：保守估计50%
                CardinalityEstimate {
                    min: 0,
                    max: self.total_records,
                    exp: self.total_records / 2,
                    selectivity: 0.5,
                }
            }
        }
    }

    /// The ordinals satisfying one condition.
    ///
    /// Ordinals rather than ids all the way through `get_matching_ids`: the intersections and
    /// differences it performs are then over four-byte integers rather than over hashed heap
    /// strings, and only the survivors are turned back into ids. A filter matching 2.5M records
    /// used to hash and clone 2.5M `String`s per condition on the way in as well as out.
    fn get_condition_ids(&self, condition: &Condition) -> FastSet<Ordinal> {
        let key = condition.key.as_str();
        match &condition.operation {
            // The negations are complements of the positive answer, and the complement has to
            // be taken against the right universe. Asking each indexed value "are you not
            // `t`?" instead — which this did — can only find records that *have* the field as
            // a scalar, so `NotEqual` silently dropped every record lacking it and every record
            // holding an array. `records/list` returns this set without re-checking, so the
            // same filter gave a different answer there than in a search, which walks the graph
            // with `Record::matches_filter`.
            FilterOperation::NotEqual(target) => {
                // `value != Some(target)`: true when the field is missing, so every record is
                // in the universe.
                let equal = self.positive_ids(key, &FilterOperation::Equal(target.clone()));
                return self.ids.live().filter(|o| !equal.contains(o)).collect();
            }
            FilterOperation::NotIn(targets) => {
                // `value.is_some_and(..)`: false when the field is missing, so only the records
                // that carry it — any type of value, arrays included — are in the universe.
                let within = self.positive_ids(key, &FilterOperation::In(targets.clone()));
                let mut result = FastSet::default();
                let mut keep = |o: Ordinal| {
                    if !within.contains(&o) {
                        result.insert(o);
                    }
                };
                if key == "id" {
                    self.ids.live().for_each(&mut keep);
                } else {
                    if let Some(field) = self.field_indices.get(key) {
                        for postings in field.value_to_ids.values() {
                            postings.for_each(&mut keep);
                        }
                        field.numeric.select(None, None, |_| true, &mut keep);
                    }
                    if let Some(held) = self.unindexed.get(key) {
                        held.iter().for_each(&mut keep);
                    }
                }
                return result;
            }
            _ => {}
        }
        self.positive_ids(key, &condition.operation)
    }

    /// `id` means the record's own id, not a payload field of that name.
    ///
    /// That is how [`crate::types::record::Record::matches_filter`] reads it, and the index
    /// answered from the payload instead: measured in a randomised comparison, 984 of 990
    /// `id` filters disagreed. The index already holds every record's id, so it answers from
    /// those. An id is a string, so the numeric comparisons match nothing, as they do there.
    fn id_ordinals(&self, operation: &FilterOperation) -> FastSet<Ordinal> {
        let by_id = |value: &serde_json::Value| value.as_str().and_then(|id| self.ids.get(id));
        let scan = |test: &dyn Fn(&str) -> bool| -> FastSet<Ordinal> {
            self.ids
                .live()
                .filter(|&o| self.ids.id_of(o).is_some_and(test))
                .collect()
        };
        match operation {
            FilterOperation::Equal(value) => by_id(value).into_iter().collect(),
            FilterOperation::In(values) => values.iter().filter_map(by_id).collect(),
            FilterOperation::Contains(s) => scan(&|id| id.contains(s.as_str())),
            FilterOperation::StartsWith(s) => scan(&|id| id.starts_with(s.as_str())),
            FilterOperation::EndsWith(s) => scan(&|id| id.ends_with(s.as_str())),
            FilterOperation::GreaterThan(_)
            | FilterOperation::GreaterThanOrEqual(_)
            | FilterOperation::LessThan(_)
            | FilterOperation::LessThanOrEqual(_) => FastSet::default(),
            // Complements, answered by the caller from the positive forms above.
            FilterOperation::NotEqual(_) | FilterOperation::NotIn(_) => {
                unreachable!("negations are taken as complements in get_condition_ids")
            }
        }
    }

    /// The records satisfying a condition that is not a negation.
    fn positive_ids(&self, key: &str, operation: &FilterOperation) -> FastSet<Ordinal> {
        if key == "id" {
            return self.id_ordinals(operation);
        }
        let condition = Condition {
            key: key.to_string(),
            operation: operation.clone(),
        };
        let condition = &condition;
        let field_index = match self.field_indices.get(&condition.key) {
            Some(idx) => idx,
            None => return FastSet::default(),
        };

        let mut result = FastSet::default();

        // The numeric half. A range names bounds, so the walk visits the rows that can match
        // instead of every distinct value of the field -- which is the whole point of the
        // sorted index. `Contains` and the other string tests name no number at all and skip
        // it entirely.
        if Self::touches_numbers(&condition.operation) {
            let (lo, hi) = Self::numeric_bounds(&condition.operation);
            field_index.numeric.select(
                lo,
                hi,
                |value| Self::value_key_matches(&Self::from_ordered(value), &condition.operation),
                |id| {
                    result.insert(id);
                },
            );
        }

        // The map half: strings, bools, nulls. `Equal` and `In` name the values they want, so
        // they are hash lookups; everything else has to be asked of each value in turn,
        // through the one predicate in [`Self::value_key_matches`] -- this used to be nine
        // near-identical loops, and nine copies of a comparison is nine chances for the "what
        // matches" answer to disagree with the "does anything match" answer.
        //
        // Neither lookup may return early. They did, and an `Equal` on an integer then never
        // reached the numeric half above and matched nothing at all.
        match &condition.operation {
            FilterOperation::Equal(value) => {
                if let Some(ids) =
                    Self::to_value_key(value).and_then(|key| field_index.value_to_ids.get(&key))
                {
                    ids.for_each(|id| {
                        result.insert(id);
                    });
                }
            }
            FilterOperation::In(values) => {
                for value in values {
                    if let Some(ids) =
                        Self::to_value_key(value).and_then(|key| field_index.value_to_ids.get(&key))
                    {
                        ids.for_each(|id| {
                            result.insert(id);
                        });
                    }
                }
            }
            _ => {
                for (value_key, ids) in &field_index.value_to_ids {
                    if Self::value_key_matches(value_key, &condition.operation) {
                        ids.for_each(|id| {
                            result.insert(id);
                        });
                    }
                }
            }
        }
        result
    }

    /// The bounds a condition puts on a numeric value, when it puts any.
    ///
    /// Only the comparisons narrow the sorted range. `Equal` narrows it to a point, and the
    /// negations narrow nothing -- every value except one satisfies `NotEqual`, so the scan
    /// is the whole index either way.
    fn numeric_bounds(operation: &FilterOperation) -> (Option<OrderedFloat>, Option<OrderedFloat>) {
        let at = |v: &serde_json::Value| v.as_f64().map(OrderedFloat::from);
        match operation {
            FilterOperation::Equal(v) => (at(v), at(v)),
            FilterOperation::GreaterThan(v) | FilterOperation::GreaterThanOrEqual(v) => {
                (at(v), None)
            }
            FilterOperation::LessThan(v) | FilterOperation::LessThanOrEqual(v) => (None, at(v)),
            _ => (None, None),
        }
    }

    /// Whether a numeric value could satisfy this condition at all. `Contains` and the other
    /// string tests never can, so the sorted half is skipped for them entirely.
    fn touches_numbers(operation: &FilterOperation) -> bool {
        !matches!(
            operation,
            FilterOperation::Contains(_)
                | FilterOperation::StartsWith(_)
                | FilterOperation::EndsWith(_)
        )
    }

    /// A sort key back as the value key the comparison predicate expects.
    fn from_ordered(value: OrderedFloat) -> ValueKey {
        ValueKey::Float(value)
    }

    /// 为字段建立索引
    /// Index one field of a record. An array is indexed element by element, because a
    /// condition on an array field is satisfied when any element satisfies it (see
    /// `any_element` in `types::record`): `tags: ["a", "b"]` is filed under `"a"` and under
    /// `"b"`, and `tags == "a"` finds it by the same lookup a scalar field uses.
    fn index_field(&mut self, key: &str, value: &serde_json::Value, ordinal: Ordinal) {
        match value {
            serde_json::Value::Array(elements) => {
                // Also recorded as present, which the elements alone do not say for `[]` or an
                // array of nothing but arrays and objects — and `NotIn` needs presence.
                self.mark_unindexed(key, ordinal);
                for element in elements {
                    if !matches!(
                        element,
                        serde_json::Value::Array(_) | serde_json::Value::Object(_)
                    ) {
                        self.index_scalar(key, element, ordinal);
                    }
                }
            }
            // Matched by shape, not by asking `to_value_key`: that clones a string to
            // answer, and this runs for every field of every record in a rebuild.
            serde_json::Value::Object(_) => self.mark_unindexed(key, ordinal),
            _ => self.index_scalar(key, value, ordinal),
        }
    }

    /// The reverse of [`Self::index_field`], for the payload the record was indexed with.
    fn unindex_field(&mut self, key: &str, value: &serde_json::Value, ordinal: Ordinal) {
        match value {
            serde_json::Value::Array(elements) => {
                self.unmark_unindexed(key, ordinal);
                // A repeated element is removed once per occurrence. The postings drop the
                // ordinal on the first and find nothing on the next; the numeric index holds
                // one entry per occurrence and loses one each time.
                for element in elements {
                    if !matches!(
                        element,
                        serde_json::Value::Array(_) | serde_json::Value::Object(_)
                    ) {
                        self.unindex_scalar(key, element, ordinal);
                    }
                }
            }
            serde_json::Value::Object(_) => self.unmark_unindexed(key, ordinal),
            _ => self.unindex_scalar(key, value, ordinal),
        }
    }

    /// Note that a record has `key` with a value the index cannot file — or an array. See
    /// `unindexed`.
    fn mark_unindexed(&mut self, key: &str, ordinal: Ordinal) {
        match self.unindexed.get_mut(key) {
            Some(held) => {
                held.insert(ordinal);
            }
            None => {
                let mut held = roaring::RoaringBitmap::new();
                held.insert(ordinal);
                self.unindexed.insert(key.to_string(), held);
            }
        }
    }

    fn unmark_unindexed(&mut self, key: &str, ordinal: Ordinal) {
        if let Some(held) = self.unindexed.get_mut(key) {
            held.remove(ordinal);
            if held.is_empty() {
                self.unindexed.remove(key);
            }
        }
    }

    fn unindex_scalar(&mut self, key: &str, value: &serde_json::Value, ordinal: Ordinal) {
        let (Some(field_index), Some(value_key)) =
            (self.field_indices.get_mut(key), Self::to_value_key(value))
        else {
            return;
        };
        match Self::as_ordered(&value_key) {
            Some(number) => field_index.numeric.remove(number, ordinal),
            None => {
                if let Some(ids) = field_index.value_to_ids.get_mut(&value_key) {
                    if ids.remove(ordinal) {
                        field_index.value_to_ids.remove(&value_key);
                    }
                }
            }
        }
        field_index.distinct_count =
            field_index.value_to_ids.len() + field_index.numeric.distinct_count();
    }

    /// Index one scalar value under `key`.
    fn index_scalar(&mut self, key: &str, value: &serde_json::Value, ordinal: Ordinal) {
        let Some(value_key) = Self::to_value_key(value) else {
            return;
        };

        // `entry(key.to_string())` allocated a copy of the field's name for every record, to
        // find an entry that is almost always already there. One lookup first, and the
        // allocation only on the record that creates the field.
        if !self.field_indices.contains_key(key) {
            self.field_indices.insert(
                key.to_string(),
                FieldIndex {
                    value_to_ids: FastMap::default(),
                    numeric: NumericIndex::default(),
                    field_type: Self::infer_field_type(value),
                    distinct_count: 0,
                    stats: FieldStats::new(),
                },
            );
        }
        let field_index = self
            .field_indices
            .get_mut(key)
            .expect("inserted just above when absent");

        let bulk = self.bulk;
        match Self::as_ordered(&value_key) {
            Some(number) if bulk => field_index.numeric.push(number, ordinal),
            Some(number) => field_index.numeric.insert(number, ordinal),
            None => match field_index.value_to_ids.get_mut(&value_key) {
                Some(postings) => postings.insert(ordinal),
                None => {
                    field_index
                        .value_to_ids
                        .insert(value_key, Postings::One(ordinal));
                }
            },
        }

        field_index.distinct_count =
            field_index.value_to_ids.len() + field_index.numeric.distinct_count();

        // 更新字段统计信息
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    field_index.stats.update_numeric(f);
                }
            }
            serde_json::Value::String(s) => {
                field_index.stats.update_string(s);
            }
            _ => {}
        }
    }

    /// 将JSON值转换为索引键
    fn to_value_key(value: &serde_json::Value) -> Option<ValueKey> {
        match value {
            serde_json::Value::String(s) => Some(ValueKey::String(s.clone())),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(ValueKey::Number(i))
                } else {
                    n.as_f64().map(|f| ValueKey::Float(OrderedFloat::from(f)))
                }
            }
            serde_json::Value::Bool(b) => Some(ValueKey::Bool(*b)),
            serde_json::Value::Null => Some(ValueKey::Null),
            _ => None, // 数组和对象暂不索引
        }
    }

    /// 推断字段类型
    fn infer_field_type(value: &serde_json::Value) -> FieldType {
        match value {
            serde_json::Value::String(_) => FieldType::String,
            serde_json::Value::Number(_) => FieldType::Number,
            serde_json::Value::Bool(_) => FieldType::Boolean,
            _ => FieldType::Mixed,
        }
    }

    /// 获取索引统计信息
    pub fn get_stats(&self) -> PayloadIndexStats {
        PayloadIndexStats {
            total_records: self.total_records,
            indexed_fields: self.field_indices.len(),
            field_details: self
                .field_indices
                .iter()
                .map(|(name, idx)| (name.clone(), idx.distinct_count))
                .collect(),
        }
    }
}

/// PayloadIndex统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadIndexStats {
    pub total_records: usize,
    pub indexed_fields: usize,
    pub field_details: HashMap<String, usize>,
}

impl Default for PayloadIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod estimator_repro {
    use super::*;
    use crate::types::record::{Condition, Filter, FilterOperation};

    fn payload(pairs: &[(&str, serde_json::Value)]) -> serde_json::Value {
        serde_json::Value::Object(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
        )
    }

    fn equals(field: &str, value: serde_json::Value) -> Filter {
        Filter {
            must: Some(vec![Condition {
                key: field.to_string(),
                operation: FilterOperation::Equal(value),
            }]),
            must_not: None,
            should: None,
        }
    }

    /// Postings hold an ordinal, and an ordinal is handed out again once its record is gone.
    ///
    /// That is the one failure mode the move from ids to ordinals introduced. A posting left
    /// behind by an incomplete removal used to name a deleted id, which resolves to nothing; it
    /// would now name a live record that happens to have inherited the number. This pins the
    /// contract `remove_record` relies on: every posting for a record is taken out before its
    /// ordinal is released.
    #[test]
    fn a_reused_ordinal_does_not_carry_the_previous_record_s_values() {
        let mut index = PayloadIndex::new();
        let gone = payload(&[
            ("topic", serde_json::json!("alpha")),
            ("n", serde_json::json!(1)),
        ]);
        index.insert_record(&"first".to_string(), &gone, 0);
        assert_eq!(
            index
                .get_matching_ids(&equals("topic", serde_json::json!("alpha")))
                .len(),
            1
        );

        index.remove_record(&"first".to_string(), &gone);
        assert!(index
            .get_matching_ids(&equals("topic", serde_json::json!("alpha")))
            .is_empty());

        // The new record takes the freed ordinal.
        let fresh = payload(&[
            ("topic", serde_json::json!("beta")),
            ("n", serde_json::json!(2)),
        ]);
        index.insert_record(&"second".to_string(), &fresh, 0);

        assert_eq!(
            index.get_matching_ids(&equals("topic", serde_json::json!("beta"))),
            HashSet::from(["second".to_string()]),
        );
        assert!(
            index
                .get_matching_ids(&equals("topic", serde_json::json!("alpha")))
                .is_empty(),
            "the reused ordinal answered for the record that used to hold it"
        );
        assert!(index
            .get_matching_ids(&equals("n", serde_json::json!(1)))
            .is_empty());
        assert_eq!(
            index.get_matching_ids(&equals("n", serde_json::json!(2))),
            HashSet::from(["second".to_string()]),
        );
    }

    fn filter(
        must: Option<Vec<Condition>>,
        must_not: Option<Vec<Condition>>,
        should: Option<Vec<Condition>>,
    ) -> Filter {
        Filter {
            must,
            must_not,
            should,
        }
    }

    fn cond(field: &str, value: serde_json::Value) -> Condition {
        Condition {
            key: field.to_string(),
            operation: FilterOperation::Equal(value),
        }
    }

    fn three_topics() -> PayloadIndex {
        let mut index = PayloadIndex::new();
        for (id, topic) in [("a", "alpha"), ("b", "beta"), ("c", "gamma")] {
            index.insert_record(
                &id.to_string(),
                &payload(&[("topic", serde_json::json!(topic))]),
                0,
            );
        }
        index
    }

    fn ids(index: &PayloadIndex, filter: &Filter) -> Vec<String> {
        let mut out: Vec<String> = index.get_matching_ids(filter).into_iter().collect();
        out.sort();
        out
    }

    /// The index has to answer the same question the same way `Record::matches_filter` does.
    ///
    /// It is the other implementation of "does this record match", reached whenever a filtered
    /// search walks the graph instead of consulting the index — so a disagreement is the same
    /// query returning different answers from different endpoints, which is what these three
    /// were. Each assertion is a case measured on a live table before the fix.
    #[test]
    fn the_index_agrees_with_matches_filter_on_the_shapes_it_used_to_get_wrong() {
        let index = three_topics();

        // `must_not` alone: everything except what it excludes. Returned nothing, because the
        // result set began empty and the exclusions had nothing to remove from.
        assert_eq!(
            ids(
                &index,
                &filter(
                    None,
                    Some(vec![cond("topic", serde_json::json!("beta"))]),
                    None
                )
            ),
            vec!["a", "c"],
        );

        // `should` alongside `must`: an additional constraint, not an alternative. `should` was
        // consulted only when `must` was absent, so this used to return both a and b.
        assert_eq!(
            ids(
                &index,
                &filter(
                    Some(vec![Condition {
                        key: "topic".to_string(),
                        operation: FilterOperation::In(vec![
                            serde_json::json!("alpha"),
                            serde_json::json!("beta"),
                        ]),
                    }]),
                    None,
                    Some(vec![cond("topic", serde_json::json!("alpha"))]),
                )
            ),
            vec!["a"],
        );

        // A conjunction of no conditions constrains nothing. Returned nothing.
        assert_eq!(
            ids(&index, &filter(Some(vec![]), None, None)),
            vec!["a", "b", "c"]
        );

        // And the combinations that were already right stay right.
        assert_eq!(
            ids(
                &index,
                &filter(
                    Some(vec![cond("topic", serde_json::json!("alpha"))]),
                    None,
                    None
                )
            ),
            vec!["a"],
        );
        assert_eq!(
            ids(
                &index,
                &filter(
                    Some(vec![Condition {
                        key: "topic".to_string(),
                        operation: FilterOperation::In(vec![
                            serde_json::json!("alpha"),
                            serde_json::json!("beta"),
                        ]),
                    }]),
                    Some(vec![cond("topic", serde_json::json!("alpha"))]),
                    None,
                )
            ),
            vec!["b"],
        );
        // A `must` that matches nothing stays empty whatever else the filter says.
        assert!(ids(
            &index,
            &filter(
                Some(vec![cond("topic", serde_json::json!("nothing"))]),
                Some(vec![cond("topic", serde_json::json!("beta"))]),
                None
            )
        )
        .is_empty());
    }

    /// An update keeps the record's ordinal and replaces what it is indexed under.
    #[test]
    fn an_update_reindexes_the_same_record_rather_than_adding_a_second() {
        let mut index = PayloadIndex::new();
        let before = payload(&[("topic", serde_json::json!("alpha"))]);
        let after = payload(&[("topic", serde_json::json!("beta"))]);
        index.insert_record(&"r".to_string(), &before, 0);
        index.update_record(&"r".to_string(), &before, &after, 0);

        assert!(index
            .get_matching_ids(&equals("topic", serde_json::json!("alpha")))
            .is_empty());
        assert_eq!(
            index.get_matching_ids(&equals("topic", serde_json::json!("beta"))),
            HashSet::from(["r".to_string()]),
        );
    }

    /// A numeric field answers counts, ranges and equality from the sorted half.
    ///
    /// Every one of these went through `value_to_ids` until numbers moved out of it, and none
    /// of the 225 unit tests noticed the move: they were read out of the code, not caught.
    /// Each assertion here is one place that returned an empty answer instead of an error.
    #[test]
    fn a_numeric_field_is_answered_from_the_sorted_half() {
        let mut index = PayloadIndex::new();
        for i in 0..1_000u32 {
            index.insert_record(&i.to_string(), &serde_json::json!({ "n": i }), 0);
        }

        let cond = |op| Condition {
            key: "n".to_string(),
            operation: op,
        };
        let must = |op| Filter {
            must: Some(vec![cond(op)]),
            must_not: None,
            should: None,
        };

        // Equality on an integer. The sorted index stores one scaled key for integers and
        // floats alike, so a comparison by `ValueKey` variant would find nothing here.
        assert_eq!(
            index.get_matching_ids(&must(FilterOperation::Equal(serde_json::json!(7)))),
            ["7".to_string()].into_iter().collect()
        );
        let est = index.estimate_cardinality(&must(FilterOperation::Equal(serde_json::json!(7))));
        assert_eq!(est.exp, 1, "one row has n = 7");

        // A range returns exactly its rows.
        let ge = index.get_matching_ids(&must(FilterOperation::GreaterThanOrEqual(
            serde_json::json!(990),
        )));
        assert_eq!(ge.len(), 10, "n >= 990 is 990..=999");
        assert!(ge.contains("990") && ge.contains("999") && !ge.contains("989"));

        // `In` counts through the sorted half too.
        let in_est = index.estimate_cardinality(&must(FilterOperation::In(vec![
            serde_json::json!(1),
            serde_json::json!(2),
            serde_json::json!(3),
        ])));
        assert_eq!(in_est.exp, 3);

        // And the emptiness proof asks both halves: this one is satisfiable, so it must not
        // be claimed empty just because the string half holds nothing.
        assert!(
            !index.cannot_match_anything(&must(FilterOperation::GreaterThanOrEqual(
                serde_json::json!(500)
            )))
        );
        assert!(
            index.cannot_match_anything(&must(FilterOperation::GreaterThanOrEqual(
                serde_json::json!(5_000)
            )))
        );

        // Removal reaches the sorted half.
        index.remove_record(&"7".to_string(), &serde_json::json!({ "n": 7 }));
        assert!(index
            .get_matching_ids(&must(FilterOperation::Equal(serde_json::json!(7))))
            .is_empty());

        // A field holding both numbers and strings serves each from its own half.
        index.insert_record(&"tag".to_string(), &serde_json::json!({ "n": "seven" }), 0);
        assert_eq!(
            index.get_matching_ids(&must(FilterOperation::Equal(serde_json::json!("seven")))),
            ["tag".to_string()].into_iter().collect()
        );
        assert_eq!(
            index
                .get_matching_ids(&must(FilterOperation::GreaterThanOrEqual(
                    serde_json::json!(990)
                )))
                .len(),
            10,
            "the string value must not join a numeric range"
        );
    }

    /// Deciding that a filter *can* match must not cost more when more rows match.
    ///
    /// `cannot_match_anything` runs on every filtered query. The first version of the sorted
    /// index answered it by collecting matches instead of short-circuiting, so a `>= threshold`
    /// that most rows pass walked most of the table to conclude "yes, something matches". On
    /// 10M rows at 99% selectivity that took a filtered search from 10.5 ms a query to 52.9 ms
    /// -- caught by a benchmark control case, which is far too late for something a test can
    /// state directly.
    #[test]
    fn proving_a_filter_satisfiable_does_not_scale_with_how_many_rows_satisfy_it() {
        let build = |n: u32| {
            let mut index = PayloadIndex::new();
            for i in 0..n {
                index.insert_record(&i.to_string(), &serde_json::json!({ "n": i }), 0);
            }
            index
        };
        let satisfiable = Filter {
            must: Some(vec![Condition {
                key: "n".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(0)),
            }]),
            must_not: None,
            should: None,
        };

        // Everything matches in both, so a short-circuit stops at the first entry either way
        // and the work is the same; a collecting implementation does twenty times more in the
        // larger index.
        let small = build(5_000);
        let large = build(100_000);
        assert!(!small.cannot_match_anything(&satisfiable));
        assert!(!large.cannot_match_anything(&satisfiable));

        let time = |index: &PayloadIndex| {
            let start = std::time::Instant::now();
            for _ in 0..200 {
                std::hint::black_box(index.cannot_match_anything(&satisfiable));
            }
            start.elapsed()
        };
        // Warm both, then take the smaller of two runs each: this is a shape assertion, not a
        // latency one, and the bound is loose enough to survive a busy machine while still
        // failing a walk that is twenty times longer.
        let _ = (time(&small), time(&large));
        let small_us = time(&small).min(time(&small)).as_micros().max(1);
        let large_us = time(&large).min(time(&large)).as_micros().max(1);
        assert!(
            large_us < small_us * 5,
            "proving a 100,000-row filter satisfiable took {large_us} us against {small_us} us \
             for the same filter over 5,000 rows: the proof is walking the matches instead of \
             stopping at the first one"
        );
    }

    /// A range pair with nothing between it is provably empty, and provable without touching
    /// a single record id.
    ///
    /// Neither half is empty on its own -- `>= 400` accepts the 1000s and `<= 600` accepts
    /// the 0s -- so this only works because conditions on one key are tested against one
    /// value together. The first version grouped nothing and proved nothing here.
    #[test]
    fn a_range_with_no_value_between_it_is_provably_empty() {
        let mut index = PayloadIndex::new();
        for i in 0..1_000u32 {
            let bucket = if i % 10 == 0 { 1000 } else { 0 };
            index.insert_record(&i.to_string(), &serde_json::json!({ "bucket": bucket }), 0);
        }

        let between = |lo, hi| Filter {
            must: Some(vec![
                Condition {
                    key: "bucket".to_string(),
                    operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(lo)),
                },
                Condition {
                    key: "bucket".to_string(),
                    operation: FilterOperation::LessThanOrEqual(serde_json::json!(hi)),
                },
            ]),
            must_not: None,
            should: None,
        };

        assert!(
            index.cannot_match_anything(&between(400, 600)),
            "no bucket is between 400 and 600"
        );
        // And the ranges that do contain a value are not claimed empty.
        assert!(!index.cannot_match_anything(&between(0, 600)));
        assert!(!index.cannot_match_anything(&between(600, 2000)));
        assert!(!index.cannot_match_anything(&between(0, 2000)));

        // The claim agrees with what the ids say, which is the property that matters.
        assert!(index.get_matching_ids(&between(400, 600)).is_empty());
        assert_eq!(index.get_matching_ids(&between(600, 2000)).len(), 100);

        // A negation proves nothing: an array-valued record satisfies it without being
        // indexed, so the shortcut has to decline however empty the index looks.
        let not_equal = Filter {
            must: Some(vec![Condition {
                key: "absent".to_string(),
                operation: FilterOperation::NotEqual(serde_json::json!(1)),
            }]),
            must_not: None,
            should: None,
        };
        assert!(!index.cannot_match_anything(&not_equal));

        // But a range on a field no record carries is empty, and that one is provable.
        let absent_range = Filter {
            must: Some(vec![Condition {
                key: "absent".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(1)),
            }]),
            must_not: None,
            should: None,
        };
        assert!(index.cannot_match_anything(&absent_range));
    }

    fn index_with_ids(n: usize) -> PayloadIndex {
        let mut idx = PayloadIndex::new();
        for i in 0..n {
            idx.insert_record(&i.to_string(), &serde_json::json!({ "id_num": i }), 0);
        }
        idx
    }

    fn ge(value: usize) -> Filter {
        Filter {
            must: Some(vec![Condition {
                key: "id_num".to_string(),
                operation: FilterOperation::GreaterThanOrEqual(serde_json::json!(value)),
            }]),
            must_not: None,
            should: None,
        }
    }

    /// The state production was actually in: records loaded, but none carrying the field the
    /// filter selects on, because the harness only writes it when asked to. The server logged
    /// `selectivity=50.00%` there, and a query that could not match anything took 16.6 s.
    #[test]
    fn an_absent_field_estimates_zero_not_half_the_table() {
        let mut idx = PayloadIndex::new();
        for i in 0..100_000usize {
            // Exactly what the unfiltered load writes: an empty payload.
            idx.insert_record(&i.to_string(), &serde_json::json!({}), 0);
        }
        let est = idx.estimate_cardinality(&ge(50_000));
        let matched = idx.get_matching_ids(&ge(50_000)).len();
        println!(
            "\n  field absent: estimate {:.2}%  exp {}  actual matches {}",
            est.selectivity * 100.0,
            est.exp,
            matched
        );
        assert_eq!(matched, 0, "no record carries the field");
        // The estimate is the guard, not the result. A search on an absent field returns an
        // empty page whatever the estimate says -- post-filtering 600 candidates that all
        // fail is still correct -- so correctness tests cannot catch this. What the estimate
        // decides is the *route*: 0 sends the search to the precomputed-empty-set path,
        // while 0.5 sent it to post-filtering, which on a 5M x 1536 table took 16.6 s to
        // return nothing while holding the table read lock for 4.6 s.
        assert_eq!(est.exp, 0, "an absent field cannot match any row");
        assert_eq!(
            est.selectivity, 0.0,
            "an absent field must estimate zero, not half the table"
        );
    }

    /// What the estimator says about `id_num >= t` over a uniform 0..n, which is the shape
    /// every VectorDBBench filtered case uses. Printed rather than asserted: this is the
    /// measurement that decides what to fix.
    #[test]
    fn what_the_estimator_says_about_a_uniform_field() {
        let n = 100_000usize;
        let idx = index_with_ids(n);
        println!("\n  n = {n}");
        for pct in [1usize, 10, 50, 90, 99] {
            let t = n * pct / 100;
            let truth = (n - t) as f64 / n as f64;
            let est = idx.estimate_cardinality(&ge(t));
            let matched = idx.get_matching_ids(&ge(t)).len();
            println!(
                "  id_num >= {t:>6}: truth {:>6.2}%  estimate {:>6.2}%  (exp {:>7})  actual matches {matched:>6}",
                truth * 100.0,
                est.selectivity * 100.0,
                est.exp,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_payload_index_insert_and_cardinality() {
        let mut index = PayloadIndex::new();

        // 插入100条记录，其中10条category=A, 90条category=B
        for i in 0..100 {
            let record_id = format!("rec_{}", i);
            let category = if i < 10 { "A" } else { "B" };
            let payload = json!({
                "category": category,
                "score": i
            });
            index.insert_record(&record_id, &payload, 0);
        }

        // Debug: Check index state
        assert_eq!(index.total_records, 100);
        println!("Total records: {}", index.total_records);
        if let Some(field_idx) = index.field_indices.get("category") {
            println!(
                "Category field distinct count: {}",
                field_idx.distinct_count
            );
            println!(
                "Category field value_to_ids keys: {:?}",
                field_idx.value_to_ids.keys().collect::<Vec<_>>()
            );
            for (key, ids) in &field_idx.value_to_ids {
                println!("  {:?}: {} records", key, ids.len());
            }
        }

        // 测试Equal过滤 - category=A (10%)
        let filter = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(json!("A")),
            }]),
            must_not: None,
            should: None,
        };

        let estimate = index.estimate_cardinality(&filter);
        println!(
            "Estimate: exp={}, selectivity={}",
            estimate.exp, estimate.selectivity
        );
        assert_eq!(estimate.exp, 10);
        assert!((estimate.selectivity - 0.1).abs() < 0.01);

        // 测试In过滤 - category in [A, B] (100%)
        let filter_in = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::In(vec![json!("A"), json!("B")]),
            }]),
            must_not: None,
            should: None,
        };

        let estimate_in = index.estimate_cardinality(&filter_in);
        assert_eq!(estimate_in.exp, 100);
        assert!((estimate_in.selectivity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_get_matching_ids() {
        let mut index = PayloadIndex::new();

        for i in 0..10 {
            let record_id = format!("rec_{}", i);
            let payload = json!({"category": if i < 5 { "A" } else { "B" }});
            index.insert_record(&record_id, &payload, 0);
        }

        let filter = Filter {
            must: Some(vec![Condition {
                key: "category".to_string(),
                operation: FilterOperation::Equal(json!("A")),
            }]),
            must_not: None,
            should: None,
        };

        let matching_ids = index.get_matching_ids(&filter);
        assert_eq!(matching_ids.len(), 5);
        assert!(matching_ids.contains("rec_0"));
        assert!(matching_ids.contains("rec_4"));
        assert!(!matching_ids.contains("rec_5"));
    }
}

/// The index under churn, checked against the record's own answer for every operation.
///
/// Every disagreement this found is a filter that returned different records from
/// `records/list` than from a search that walks the graph. Widened to every operation, it
/// found five kinds, fixed with it: `NotEqual` missing records that lack the field, `NotIn` and
/// arrays, `id` answered from the payload, numbers keyed at six decimals and saturating, and
/// `5` against `5.0`. A filter whose *target* is an array or object is not generated: it is
/// refused before it reaches the index (`Filter::validate`).
///
/// The postings and the id bookkeeping under churn, checked against the record's own answer.
///
/// Both representations changed shape for the 10M-row rebuild — postings became `One` or a
/// roaring bitmap instead of a `HashSet`, and ids are stored once behind a table of ordinals —
/// and both have states a fixed example does not reach: a bitmap shrinking back to `One`, an
/// entry removed when its last record goes, an ordinal handed out again after a delete, a table
/// rehashed while the free list is non-empty. Random churn reaches them; `Record::matches_filter`
/// is the other implementation of "does this match" and is the oracle.
#[cfg(test)]
mod churn {
    use super::*;
    use crate::types::record::{Record, RecordStatus};

    /// xorshift, so the sequence is the same on every run and a failure names a seed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const TOPICS: [&str; 3] = ["alpha", "beta", "gamma"];

    /// Shapes on purpose: a few repeated strings (bitmaps), a string per record (`One`), a
    /// number that is sometimes an integer and sometimes a float, a bool — which lives in the
    /// map beside the strings — an array, which the index cannot hold at all, and a payload
    /// field called `id`, which is not the record's id. Every field is sometimes absent.
    fn random_payload(rng: &mut Rng, serial: u64) -> serde_json::Value {
        // Not every payload is an object. A record whose payload is `null` has no fields at
        // all, which `matches_filter` answers for — `NotEqual` true, `id` its own id — and an
        // index that only numbers records with object payloads never sees.
        if rng.below(20) == 0 {
            return serde_json::Value::Null;
        }
        let mut map = serde_json::Map::new();
        if rng.below(10) > 0 {
            map.insert("topic".into(), TOPICS[rng.below(3) as usize].into());
        }
        if rng.below(10) > 0 {
            map.insert("title".into(), format!("title {serial}").into());
        }
        if rng.below(10) > 0 {
            let n = rng.below(20);
            let value: serde_json::Value = match rng.below(3) {
                0 => (n as f64).into(),       // 5.0
                1 => (n as f64 + 0.5).into(), // 5.5
                _ => (n as i64).into(),       // 5
            };
            map.insert("n".into(), value);
        }
        if rng.below(10) > 0 {
            map.insert("flag".into(), (rng.below(2) == 0).into());
        }
        if rng.below(3) == 0 {
            // Arrays in every shape the element rule has an edge for: empty, one element, a
            // repeat, a nested array (an element that matches nothing scalar), mixed types.
            let tags = match rng.below(7) {
                0 => serde_json::json!([]),
                1 => serde_json::json!(["a"]),
                2 => serde_json::json!(["a", "b"]),
                3 => serde_json::json!(["b", "c"]),
                4 => serde_json::json!(["a", "a"]),
                5 => serde_json::json!([["a"], "c"]),
                _ => serde_json::json!(["a", 1]),
            };
            map.insert("tags".into(), tags);
        }
        if rng.below(3) == 0 {
            let scores = match rng.below(5) {
                0 => serde_json::json!([]),
                1 => serde_json::json!([1, 7]),
                2 => serde_json::json!([3.5]),
                3 => serde_json::json!([7, 7.0]),
                _ => serde_json::json!([100, 700]),
            };
            map.insert("scores".into(), scores);
        }
        if rng.below(10) > 0 {
            // A microsecond timestamp, and a value that differs from its neighbours below a
            // millionth — the two places a fixed-point numeric key cannot tell values apart.
            map.insert(
                "us".into(),
                (1_700_000_000_000_000i64 + rng.below(5) as i64).into(),
            );
            map.insert("tiny".into(), (rng.below(5) as f64 * 1e-7).into());
        }
        if rng.below(4) == 0 {
            map.insert("id".into(), format!("payload-{}", rng.below(3)).into());
        }
        serde_json::Value::Object(map)
    }

    fn topic(rng: &mut Rng) -> serde_json::Value {
        TOPICS[rng.below(3) as usize].into()
    }

    /// Every operation, including the ones that negate — which is where a record *lacking*
    /// the field gets an answer, and so where an index that only knows present fields can
    /// disagree.
    fn random_condition(rng: &mut Rng, serial: u64) -> Condition {
        let number = |rng: &mut Rng| -> serde_json::Value {
            let n = rng.below(20);
            match rng.below(3) {
                0 => (n as f64).into(),
                1 => (n as f64 + 0.5).into(),
                _ => (n as i64).into(),
            }
        };
        let us = |rng: &mut Rng| -> serde_json::Value {
            (1_700_000_000_000_000i64 + rng.below(5) as i64).into()
        };
        let tiny = |rng: &mut Rng| -> serde_json::Value { (rng.below(5) as f64 * 1e-7).into() };
        let score = |rng: &mut Rng| -> serde_json::Value {
            [1.0, 3.5, 7.0, 100.0, 400.0, 600.0, 700.0][rng.below(7) as usize].into()
        };
        let (key, operation) = match rng.below(24) {
            20 => ("scores", FilterOperation::GreaterThanOrEqual(score(rng))),
            21 => ("scores", FilterOperation::LessThanOrEqual(score(rng))),
            22 => ("scores", FilterOperation::Equal(score(rng))),
            23 => ("scores", FilterOperation::NotEqual(score(rng))),
            16 => ("us", FilterOperation::Equal(us(rng))),
            17 => ("us", FilterOperation::GreaterThan(us(rng))),
            18 => ("tiny", FilterOperation::Equal(tiny(rng))),
            19 => ("tiny", FilterOperation::LessThanOrEqual(tiny(rng))),
            0 => ("topic", FilterOperation::Equal(topic(rng))),
            1 => ("topic", FilterOperation::In(vec![topic(rng), topic(rng)])),
            2 => ("topic", FilterOperation::NotEqual(topic(rng))),
            3 => (
                "topic",
                FilterOperation::NotIn(vec![topic(rng), topic(rng)]),
            ),
            4 => (
                "title",
                FilterOperation::Equal(format!("title {}", rng.below(serial.max(1))).into()),
            ),
            5 => ("title", FilterOperation::StartsWith("title 1".into())),
            6 => ("title", FilterOperation::EndsWith("7".into())),
            7 => ("title", FilterOperation::Contains("1".into())),
            8 => ("n", FilterOperation::GreaterThanOrEqual(number(rng))),
            9 => ("n", FilterOperation::LessThan(number(rng))),
            10 => ("n", FilterOperation::Equal(number(rng))),
            11 => ("n", FilterOperation::NotEqual(number(rng))),
            12 => ("flag", FilterOperation::Equal((rng.below(2) == 0).into())),
            13 => {
                let tag = ["a", "b", "c"][rng.below(3) as usize];
                match rng.below(6) {
                    0 => ("tags", FilterOperation::Equal(tag.into())),
                    1 => ("tags", FilterOperation::NotEqual(tag.into())),
                    2 => ("tags", FilterOperation::In(vec![tag.into(), "c".into()])),
                    3 => ("tags", FilterOperation::NotIn(vec![tag.into()])),
                    4 => ("tags", FilterOperation::StartsWith(tag.into())),
                    _ => ("tags", FilterOperation::Equal(1.into())),
                }
            }
            14 => (
                "id",
                FilterOperation::Equal(format!("id{}", rng.below(60)).into()),
            ),
            _ => (
                "id",
                FilterOperation::Equal(format!("payload-{}", rng.below(3)).into()),
            ),
        };
        Condition {
            key: key.into(),
            operation,
        }
    }

    fn record(id: &str, payload: &serde_json::Value) -> Record {
        Record {
            id: id.to_string(),
            vector: vec![0.0],
            sparse_vector: None,
            named_vectors: None,
            payload: payload.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
            status: RecordStatus::Active,
        }
    }

    #[test]
    fn the_index_agrees_with_the_records_through_inserts_updates_and_deletes() {
        for seed in 1..=20u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut index = PayloadIndex::new();
            // id -> (payload, created). Timestamps come from a range of eight, so records
            // share them constantly and a page's order rests on the id tiebreak.
            let mut live: std::collections::BTreeMap<String, (serde_json::Value, i64)> =
                Default::default();
            let mut serial = 0u64;

            for step in 0..600 {
                // A small id space, so the same ids come and go and ordinals are reused.
                let id = format!("id{}", rng.below(60));
                match (live.get(&id).cloned(), rng.below(3)) {
                    (None, _) => {
                        serial += 1;
                        let p = random_payload(&mut rng, serial);
                        let created = rng.below(8) as i64;
                        index.insert_record(&id, &p, created);
                        live.insert(id, (p, created));
                    }
                    (Some((old, _)), 0) => {
                        index.remove_record(&id, &old);
                        live.remove(&id);
                    }
                    (Some((old, created)), _) => {
                        // An update keeps the record's original `created_at`.
                        serial += 1;
                        let new = random_payload(&mut rng, serial);
                        index.update_record(&id, &old, &new, created);
                        live.insert(id, (new, created));
                    }
                }

                if step % 20 == 0 {
                    for _ in 0..8 {
                        let mut must = vec![random_condition(&mut rng, serial)];
                        if rng.below(2) == 0 {
                            must.push(random_condition(&mut rng, serial));
                        }
                        // Two ranges on one array field, which one element each can satisfy —
                        // the case that breaks a proof testing them against a single value.
                        if rng.below(4) == 0 {
                            must = vec![
                                Condition {
                                    key: "scores".into(),
                                    operation: FilterOperation::GreaterThanOrEqual(400.into()),
                                },
                                Condition {
                                    key: "scores".into(),
                                    operation: FilterOperation::LessThanOrEqual(600.into()),
                                },
                            ];
                        }
                        let must_not =
                            (rng.below(3) == 0).then(|| vec![random_condition(&mut rng, serial)]);
                        let filter = Filter {
                            must: Some(must),
                            must_not,
                            should: None,
                        };
                        let mut got: Vec<String> =
                            index.get_matching_ids(&filter).into_iter().collect();
                        got.sort();
                        let mut want: Vec<(i64, String)> = live
                            .iter()
                            .filter(|(id, (p, _))| record(id, p).matches_filter(&filter))
                            .map(|(id, (_, created))| (*created, id.clone()))
                            .collect();
                        let want_ids: Vec<String> = want.iter().map(|(_, id)| id.clone()).collect();
                        // The shortcut a search takes before anything else: it may only say
                        // "nothing matches" when nothing does.
                        if index.cannot_match_anything(&filter) {
                            assert!(
                                want_ids.is_empty(),
                                "seed {seed}, step {step}: cannot_match_anything said no record \
                                 matches {filter:?}, and {want_ids:?} do"
                            );
                        }
                        assert_eq!(
                            got, want_ids,
                            "seed {seed}, step {step}: the index and the records disagree on \
                             {filter:?}"
                        );

                        // And the page `records/list` asks for is that set in list order —
                        // newest first, ties by id — cut at the offset and limit asked for.
                        want.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                        for (offset, limit) in [(0, 5), (3, 4), (0, 1000), (want.len(), 5)] {
                            let (page, total) = index.page_of_matching(&filter, offset, limit);
                            let expected: Vec<&str> = want
                                .iter()
                                .skip(offset)
                                .take(limit)
                                .map(|(_, id)| id.as_str())
                                .collect();
                            assert_eq!(total, want.len(), "seed {seed}, step {step}: total");
                            assert_eq!(
                                page, expected,
                                "seed {seed}, step {step}: page {offset}+{limit} of {filter:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_posting_list_moves_between_its_two_shapes_both_ways() {
        let mut p = Postings::One(7);
        p.insert(7);
        assert!(
            matches!(p, Postings::One(7)),
            "inserting the same ordinal twice is one"
        );
        p.insert(9);
        assert!(matches!(p, Postings::Many(_)));
        assert_eq!(p.len(), 2);

        // Removing something that is not there leaves it alone and is not "now empty".
        assert!(!Postings::One(7).remove(8));

        // Down to one: back to inline, so a formerly popular value does not keep a bitmap.
        assert!(!p.remove(9));
        assert!(matches!(p, Postings::One(7)), "{p:?}");
        // And the last one out says so, which is the caller's signal to drop the entry.
        assert!(p.remove(7));
    }

    /// The table only holds ordinals and reads the ids back out of `by_ordinal` to hash them,
    /// so a rehash with a slot emptied or refilled in the wrong order would hash nothing, or the
    /// wrong id, and lose entries without failing. Growing from nothing forces many rehashes,
    /// and the deletes in between put ordinals on the free list for the next inserts to reuse.
    #[test]
    fn ids_survive_growth_and_reuse() {
        let mut ids = Ids::default();
        for i in 0..5_000 {
            assert_eq!(ids.assign(&format!("r{i}"), 0), i as Ordinal);
        }
        for i in (0..5_000).step_by(3) {
            assert_eq!(ids.release(&format!("r{i}")), Some(i as Ordinal));
        }
        // New ids take the freed ordinals before any fresh ones.
        let reused = ids.assign("new-0", 0);
        assert!(
            reused % 3 == 0 && reused < 5_000,
            "{reused} was not a freed ordinal"
        );
        for i in 1..20_000 {
            ids.assign(&format!("new-{i}"), 0);
        }
        for i in 0..5_000u32 {
            let id = format!("r{i}");
            if i % 3 == 0 {
                assert_eq!(ids.get(&id), None, "{id} was released");
            } else {
                let o = ids
                    .get(&id)
                    .unwrap_or_else(|| panic!("{id} was lost in a rehash"));
                assert_eq!(ids.id_of(o), Some(id.as_str()));
            }
        }
        assert_eq!(
            ids.get("new-19999").and_then(|o| ids.id_of(o)),
            Some("new-19999")
        );
        // Assigning an id that already has an ordinal returns that one.
        let o = ids.get("r1").unwrap();
        assert_eq!(ids.assign("r1", 0), o);
        assert_eq!(ids.live().count(), 5_000 - 1667 + 20_000);
    }
}

#[cfg(test)]
mod contradictory_ranges {
    use super::*;

    fn ranged(lo: i64, hi: i64) -> Filter {
        Filter {
            must: Some(vec![
                Condition {
                    key: "n".into(),
                    operation: FilterOperation::GreaterThanOrEqual(lo.into()),
                },
                Condition {
                    key: "n".into(),
                    operation: FilterOperation::LessThanOrEqual(hi.into()),
                },
            ]),
            must_not: None,
            should: None,
        }
    }

    /// `n >= 40 and n <= 5` is a filter nothing satisfies, and it panicked: the two bounds
    /// became a slice whose start was past its end. A search calls `cannot_match_anything`
    /// before anything else, so this was one request away from a crashed handler.
    #[test]
    fn a_lower_bound_above_the_upper_is_empty_not_a_panic() {
        let mut index = PayloadIndex::new();
        for i in 0..50 {
            index.insert_record(&format!("r{i}"), &serde_json::json!({ "n": i }), 0);
        }
        assert!(index.cannot_match_anything(&ranged(40, 5)));
        assert!(index.get_matching_ids(&ranged(40, 5)).is_empty());
        // And an ordinary range still finds what it should.
        assert!(!index.cannot_match_anything(&ranged(5, 40)));
        assert_eq!(index.get_matching_ids(&ranged(5, 40)).len(), 36);
    }
}

#[cfg(test)]
mod cursor {
    use super::ListCursor;

    #[test]
    fn a_cursor_survives_its_own_encoding_whatever_the_id() {
        for id in ["r1", "", "with.dots:and colons", "ünïcödé/🦀", "0000"] {
            for created in [0, 1, -1, i64::MIN, i64::MAX, 1_700_000_000_123_456_789] {
                let c = ListCursor::after(created, id);
                assert_eq!(
                    ListCursor::decode(&c.encode()),
                    Ok(c.clone()),
                    "{id:?} at {created}"
                );
            }
        }
    }

    #[test]
    fn a_token_this_server_did_not_issue_is_refused() {
        for bad in [
            "",
            "123",
            "zzzzzzzzzzzzzzzz",
            "0000000000000000f",
            "0000000000000000ff",
            "000000000000000g",
        ] {
            assert!(ListCursor::decode(bad).is_err(), "{bad:?} was accepted");
        }
    }

    /// Newest first, ties by id: after the cursor means older, or as old with a larger id.
    #[test]
    fn precedes_is_the_list_order() {
        let c = ListCursor::after(100, "m");
        assert!(c.precedes(99, "a"));
        assert!(c.precedes(100, "n"));
        assert!(
            !c.precedes(100, "m"),
            "the record the cursor names is not after itself"
        );
        assert!(!c.precedes(100, "a"));
        assert!(!c.precedes(101, "z"));
    }
}
