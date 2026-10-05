//! Fast-scan: 4-bit PQ codes laid out in blocks of 32, scored 32 vectors at a time with
//! byte shuffles.
//!
//! The 8-bit codes the graph walks on are scored one at a time: a gather per subspace into
//! a 256-entry table, about 116 ns a code at 192 dimensions. A filter that matches 1% of
//! 10M rows is 100,000 of those; one that matches 10% is more than a walk. With 16
//! centroids per subspace the table is 16 bytes, and `pshufb` looks it up for 16 codes in
//! one instruction, so a block of 32 vectors costs a few instructions per subspace and a
//! code about 2-3 ns -- a scan of *every* code of a 10M table in tens of milliseconds, cut
//! down by the filter's mask block by block. That is the brute force over compressed codes
//! the filtered-search systems built for the big-ann track answer the mid-selectivity band
//! with (Faiss `IndexPQFastScan`, 2020, is the design this follows).
//!
//! The codes are coarser than the graph's: half the bits per subspace, so half the subspace
//! width to keep the bits per dimension -- and the same bytes per vector. They decide only
//! which candidates reach the exact rerank, never a distance a caller sees.
//!
//! Layout (Faiss's): block `b` holds vectors `32b..32b+32`; subspaces go in pairs, and for
//! pair `p` the block's 32 bytes at `(b * m / 2 + p) * 32` carry, in byte `j`, vector
//! `32b + j`'s code for subspace `2p` in the low nibble and for subspace `2p + 1` in the high
//! nibble. An odd subspace count is padded with one whose table is all zeros. A query's
//! table is quantised to one byte per entry: per subspace the minimum is taken out (the sum
//! of minimums is the bias), and one scale over all subspaces maps the largest residual to
//! 255. The kernel shuffles both nibbles of a pair, then widens the 32 byte scores to 16
//! bits with a mask and a shift (even vectors in the low bytes, odd in the high) rather
//! than two conversions, which is what the layout buys. Scores spill into 32 bits every
//! 256 subspaces, so no width overflows.

use crate::search::node_filter::NodeFilter;
use crate::search::quantization::product_quantization::ProductQuantizer;
use crate::types::vector::{DenseVector, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

/// Vectors per block.
pub const BLOCK: usize = 32;
/// Centroids per subspace: one nibble.
pub const CENTROIDS: usize = 16;
/// Subspaces accumulated in 16 bits before a spill to 32.
const SPILL_EVERY: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastScan {
    pq: ProductQuantizer,
    /// Subspaces.
    m: usize,
    /// Vectors held: positions `0..n`.
    n: usize,
    /// Subspaces as laid out: `m` rounded up to even.
    pairs: usize,
    /// `ceil(n / 32) * pairs * 32` bytes; see the module docs.
    blocks: Vec<u8>,
    /// The codebooks as one array, `flat[(s * 16 + c) * d ..][..d]`, for the table build.
    #[serde(skip)]
    flat: Vec<f32>,
}

/// A query's quantised table.
pub struct Lut {
    /// `m * 16` bytes.
    table: Vec<u8>,
    bias: f32,
    scale: f32,
}

impl Lut {
    /// The approximate squared distance a raw score stands for.
    #[inline]
    pub fn distance(&self, score: u32) -> f32 {
        self.bias + score as f32 / self.scale
    }
}

impl FastScan {
    /// The subspace width for `dim`, keeping the bits per dimension of the 8-bit layout
    /// (`HnswPqConfig::default_subvectors`: one dimension a subspace under 64, two under
    /// 256, four from there -- here half of that, at least one, and two from 128 so a
    /// 192-dimension code is 48 bytes like its 8-bit one), and `None` when no such width
    /// divides the dimension.
    pub fn subvector_dim_for(dim: usize) -> Option<usize> {
        let wanted = if dim < 128 { 1 } else { 2 };
        (1..=wanted)
            .rev()
            .find(|d| dim.is_multiple_of(*d))
            .filter(|&d| dim / d <= 4_096)
    }

    /// Train 16-centroid codebooks on `sample` (vectors already in the index's prepared
    /// form) and start with no codes.
    pub fn train(
        dim: usize,
        metric: DistanceMetric,
        sample: &[DenseVector],
        iterations: usize,
    ) -> crate::Result<Self> {
        let sub_dim = Self::subvector_dim_for(dim).ok_or_else(|| {
            crate::Error::InvalidInput(format!("no fast-scan layout for {dim} dimensions"))
        })?;
        let m = dim / sub_dim;
        let mut pq = ProductQuantizer::new(dim, m, CENTROIDS, metric)?;
        pq.train(sample, iterations)?;
        let mut this = Self {
            pq,
            m,
            pairs: m.div_ceil(2),
            n: 0,
            blocks: Vec::new(),
            flat: Vec::new(),
        };
        this.rebuild_flat();
        Ok(this)
    }

    /// Restore what serialisation skips.
    pub fn after_load(&mut self) {
        self.rebuild_flat();
    }

    fn rebuild_flat(&mut self) {
        let mut flat = Vec::with_capacity(self.m * CENTROIDS * self.pq.subvector_dim);
        for codebook in &self.pq.codebooks {
            for centroid in codebook {
                flat.extend_from_slice(centroid);
            }
        }
        self.flat = flat;
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn subspaces(&self) -> usize {
        self.m
    }

    /// Bytes held for the codes.
    pub fn code_bytes(&self) -> usize {
        self.blocks.len()
    }

    fn block_bytes(&self) -> usize {
        self.pairs * 32
    }

    /// Append `vector` at position `n`.
    pub fn push(&mut self, vector: &[f32]) -> crate::Result<()> {
        let idx = self.n;
        self.n += 1;
        if idx.is_multiple_of(BLOCK) {
            let bytes = (idx / BLOCK + 1) * self.block_bytes();
            self.blocks.resize(bytes, 0);
        }
        self.set(idx, vector)
    }

    /// The code for `vector`, one nibble per subspace, for a caller encoding in parallel
    /// and pushing in order with [`Self::push_code`].
    pub fn encode(&self, vector: &[f32]) -> crate::Result<Vec<u8>> {
        Ok(self.pq.encode(&vector.to_vec())?.codes)
    }

    /// Append a code from [`Self::encode`] at position `n`.
    pub fn push_code(&mut self, code: &[u8]) {
        let idx = self.n;
        self.n += 1;
        if idx.is_multiple_of(BLOCK) {
            let bytes = (idx / BLOCK + 1) * self.block_bytes();
            self.blocks.resize(bytes, 0);
        }
        self.set_code(idx, code);
    }

    /// Write `vector`'s code at an existing position.
    pub fn set(&mut self, idx: usize, vector: &[f32]) -> crate::Result<()> {
        let code = self.encode(vector)?;
        self.set_code(idx, &code);
        Ok(())
    }

    fn set_code(&mut self, idx: usize, code: &[u8]) {
        let (block, slot) = (idx / BLOCK, idx % BLOCK);
        let base = block * self.block_bytes();
        for (s, &c) in code.iter().enumerate() {
            let at = base + (s / 2) * 32 + slot;
            let c = c & 0x0F;
            self.blocks[at] = if s % 2 == 1 {
                (self.blocks[at] & 0x0F) | (c << 4)
            } else {
                (self.blocks[at] & 0xF0) | c
            };
        }
    }

    /// The code at `idx`, one nibble per subspace (tests and the decode below).
    pub fn code(&self, idx: usize) -> Vec<u8> {
        let (block, slot) = (idx / BLOCK, idx % BLOCK);
        let base = block * self.block_bytes();
        (0..self.m)
            .map(|s| {
                let v = self.blocks[base + (s / 2) * 32 + slot];
                if s % 2 == 1 {
                    v >> 4
                } else {
                    v & 0x0F
                }
            })
            .collect()
    }

    /// The query's table, quantised: distances from each query subvector to the 16
    /// centroids of its subspace.
    pub fn lut(&self, query: &[f32]) -> Lut {
        let d = self.pq.subvector_dim;
        let mut f = vec![0.0f32; self.m * CENTROIDS];
        let mut mins = vec![0.0f32; self.m];
        let mut widest = 0.0f32;
        for s in 0..self.m {
            let q = &query[s * d..(s + 1) * d];
            let mut min = f32::INFINITY;
            let mut max = 0.0f32;
            for c in 0..CENTROIDS {
                let centroid = &self.flat[(s * CENTROIDS + c) * d..(s * CENTROIDS + c + 1) * d];
                let dist: f32 = q.iter().zip(centroid).map(|(x, y)| (x - y) * (x - y)).sum();
                f[s * CENTROIDS + c] = dist;
                min = min.min(dist);
                max = max.max(dist);
            }
            mins[s] = min;
            widest = widest.max(max - min);
        }
        let scale = if widest > 0.0 { 255.0 / widest } else { 1.0 };
        let mut table: Vec<u8> = f
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                ((v - mins[i / CENTROIDS]) * scale)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
            .collect();
        // The padding subspace of an odd count scores zero.
        table.resize(self.pairs * 2 * CENTROIDS, 0);
        Lut {
            table,
            bias: mins.iter().sum(),
            scale,
        }
    }

    /// Scores of the 32 vectors of block `b`.
    #[inline]
    fn score_block(&self, lut: &Lut, b: usize, out: &mut [u32; BLOCK]) {
        let block = &self.blocks[b * self.block_bytes()..(b + 1) * self.block_bytes()];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: the feature was just detected.
                unsafe { score_block_avx2(block, &lut.table, self.pairs, out) };
                return;
            }
        }
        score_block_scalar(block, &lut.table, self.pairs, out);
    }

    /// The `k` nearest positions by the table, over every block, or only the positions
    /// `mask` allows. Nearest first, with the approximate distance. Blocks the mask leaves
    /// empty are not loaded; a block with any allowed position is scored whole.
    pub fn scan(&self, lut: &Lut, mask: Option<&NodeFilter>, k: usize) -> Vec<(usize, f32)> {
        let blocks = self.n.div_ceil(BLOCK);
        let heap = if blocks >= 4_096 {
            // Rayon over block ranges, one heap each, merged below.
            use rayon::prelude::*;
            let chunk = blocks.div_ceil(rayon::current_num_threads() * 4).max(64);
            (0..blocks)
                .into_par_iter()
                .step_by(chunk)
                .map(|start| self.scan_range(lut, mask, k, start, (start + chunk).min(blocks)))
                .reduce(BinaryHeap::new, |mut a, b| {
                    for item in b {
                        push_bounded(&mut a, item, k);
                    }
                    a
                })
        } else {
            self.scan_range(lut, mask, k, 0, blocks)
        };
        let mut out: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|Scored(score, idx)| (idx as usize, lut.distance(score)))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn scan_range(
        &self,
        lut: &Lut,
        mask: Option<&NodeFilter>,
        k: usize,
        from: usize,
        to: usize,
    ) -> BinaryHeap<Scored> {
        let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
        let mut scores = [0u32; BLOCK];
        for b in from..to {
            let allowed = match mask {
                Some(m) => m.mask32(b * BLOCK),
                None => u32::MAX,
            };
            if allowed == 0 {
                continue;
            }
            #[cfg(target_arch = "x86_64")]
            if b + 1 < to {
                // The next block's first lines, while this one is scored.
                let next = &self.blocks[(b + 1) * self.block_bytes()..];
                for line in (0..self.block_bytes().min(256)).step_by(64) {
                    // SAFETY: a prefetch takes any address and never faults.
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            next.as_ptr().add(line) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        )
                    };
                }
            }
            self.score_block(lut, b, &mut scores);
            let limit = if heap.len() < k {
                u32::MAX
            } else {
                heap.peek().map_or(u32::MAX, |s| s.0)
            };
            let end = (self.n - b * BLOCK).min(BLOCK);
            let mut bits = allowed
                & if end == BLOCK {
                    u32::MAX
                } else {
                    (1u32 << end) - 1
                };
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if scores[j] < limit || heap.len() < k {
                    push_bounded(&mut heap, Scored(scores[j], (b * BLOCK + j) as u32), k);
                }
            }
        }
        heap
    }
}

/// `(score, position)`, ordered by score so the heap's top is the worst kept.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Scored(u32, u32);

#[inline]
fn push_bounded(heap: &mut BinaryHeap<Scored>, item: Scored, k: usize) {
    if heap.len() < k {
        heap.push(item);
    } else if heap.peek().is_some_and(|worst| item < *worst) {
        heap.pop();
        heap.push(item);
    }
}

fn score_block_scalar(block: &[u8], table: &[u8], pairs: usize, out: &mut [u32; BLOCK]) {
    out.fill(0);
    for p in 0..pairs {
        let lo = &table[(2 * p) * 16..(2 * p + 1) * 16];
        let hi = &table[(2 * p + 1) * 16..(2 * p + 2) * 16];
        let bytes = &block[p * 32..(p + 1) * 32];
        for (j, &byte) in bytes.iter().enumerate() {
            out[j] += lo[(byte & 0x0F) as usize] as u32 + hi[(byte >> 4) as usize] as u32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn score_block_avx2(block: &[u8], table: &[u8], pairs: usize, out: &mut [u32; BLOCK]) {
    use std::arch::x86_64::*;
    let low_nibble = _mm256_set1_epi8(0x0F);
    let low_byte = _mm256_set1_epi16(0x00FF);
    let mut total = [0u32; BLOCK];
    // Up to `SPILL_EVERY / 2` pairs in 16 bits: two bytes of at most 255 per pair.
    let pairs_per_spill = SPILL_EVERY / 2;
    let mut p = 0;
    while p < pairs {
        let run = (pairs - p).min(pairs_per_spill);
        // Even vectors' sums in one register, odd vectors' in the other: the 32 byte scores
        // of a pair split by the mask and the shift below.
        let mut even = _mm256_setzero_si256();
        let mut odd = _mm256_setzero_si256();
        for pair in p..p + run {
            let codes = _mm256_loadu_si256(block.as_ptr().add(pair * 32) as *const __m256i);
            let lut_lo = _mm_loadu_si128(table.as_ptr().add((2 * pair) * 16) as *const __m128i);
            let lut_hi = _mm_loadu_si128(table.as_ptr().add((2 * pair + 1) * 16) as *const __m128i);
            let lut_lo = _mm256_broadcastsi128_si256(lut_lo);
            let lut_hi = _mm256_broadcastsi128_si256(lut_hi);
            let lo = _mm256_and_si256(codes, low_nibble);
            let hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_nibble);
            let d_lo = _mm256_shuffle_epi8(lut_lo, lo);
            let d_hi = _mm256_shuffle_epi8(lut_hi, hi);
            // Byte lane j is vector j's score for one subspace. Split by parity into 16-bit
            // lanes -- even vectors in the low bytes, odd in the high -- so no byte overflows.
            even = _mm256_add_epi16(even, _mm256_and_si256(d_lo, low_byte));
            even = _mm256_add_epi16(even, _mm256_and_si256(d_hi, low_byte));
            odd = _mm256_add_epi16(odd, _mm256_srli_epi16(d_lo, 8));
            odd = _mm256_add_epi16(odd, _mm256_srli_epi16(d_hi, 8));
        }
        let mut e = [0u16; 16];
        let mut o = [0u16; 16];
        _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, even);
        _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, odd);
        for j in 0..16 {
            total[2 * j] += e[j] as u32;
            total[2 * j + 1] += o[j] as u32;
        }
        p += run;
    }
    *out = total;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed | 1;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|_| (0..dim).map(|_| next() - 0.5).collect())
            .collect()
    }

    fn brute(fs: &FastScan, lut: &Lut, idx: usize) -> u32 {
        fs.code(idx)
            .iter()
            .enumerate()
            .map(|(s, &c)| lut.table[s * 16 + c as usize] as u32)
            .sum()
    }

    #[test]
    fn the_kernel_agrees_with_the_byte_loop_and_the_scan_with_a_brute_force_pass() {
        for (dim, n) in [(24usize, 100usize), (64, 70), (300, 33)] {
            let rows = corpus(n, dim, 7);
            let mut fs = FastScan::train(dim, DistanceMetric::Euclidean, &rows, 8).unwrap();
            for v in &rows {
                fs.push(v).unwrap();
            }
            assert_eq!(fs.len(), n);
            let query = &corpus(1, dim, 99)[0];
            let lut = fs.lut(query);
            // Kernel against the byte loop, block by block.
            for b in 0..n.div_ceil(BLOCK) {
                let mut fast = [0u32; BLOCK];
                let mut slow = [0u32; BLOCK];
                let block = &fs.blocks[b * fs.block_bytes()..(b + 1) * fs.block_bytes()];
                score_block_scalar(block, &lut.table, fs.pairs, &mut slow);
                fs.score_block(&lut, b, &mut fast);
                assert_eq!(fast, slow, "dim {dim} block {b}");
                for (j, &score) in slow.iter().enumerate().take(BLOCK.min(n - b * BLOCK)) {
                    assert_eq!(score, brute(&fs, &lut, b * BLOCK + j));
                }
            }
            // The scan's top-k is the brute force's top-k by score.
            let got = fs.scan(&lut, None, 10);
            let mut all: Vec<(u32, usize)> = (0..n).map(|i| (brute(&fs, &lut, i), i)).collect();
            all.sort();
            let want: Vec<u32> = all.iter().take(10).map(|(s, _)| *s).collect();
            let got_scores: Vec<u32> = got.iter().map(|(i, _)| brute(&fs, &lut, *i)).collect();
            assert_eq!(got_scores, want, "dim {dim}");
            // And a mask keeps the scan to the allowed positions.
            let mut mask = NodeFilter::with_capacity(n);
            for i in (0..n).step_by(3) {
                mask.insert(i);
            }
            let masked = fs.scan(&lut, Some(&mask), 5);
            assert!(masked.iter().all(|(i, _)| i % 3 == 0));
            assert_eq!(masked.len(), 5.min(mask.len()));
            // A rewrite at a position changes only that position.
            let before: Vec<Vec<u8>> = (0..n).map(|i| fs.code(i)).collect();
            fs.set(17 % n, &rows[0]).unwrap();
            for (i, code) in before.iter().enumerate() {
                if i != 17 % n {
                    assert_eq!(&fs.code(i), code);
                }
            }
        }
    }

    /// What a scan costs: 1M random 192-d rows, 50 queries, one thread and the rayon
    /// pool; prints ns per code. `cargo test --release --lib fast_scan_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn fast_scan_cost() {
        let (n, dim) = (1_000_000usize, 192usize);
        let rows = corpus(n, dim, 11);
        let sample: Vec<Vec<f32>> = rows.iter().step_by(16).take(16_384).cloned().collect();
        let mut fs = FastScan::train(dim, DistanceMetric::Euclidean, &sample, 10).unwrap();
        let t = std::time::Instant::now();
        for v in &rows {
            fs.push(v).unwrap();
        }
        println!(
            "encoded {n} rows in {:.1}s ({} subspaces, {} MB of codes)",
            t.elapsed().as_secs_f64(),
            fs.subspaces(),
            fs.code_bytes() / 1_000_000
        );
        let queries = corpus(50, dim, 500);
        let serial = std::time::Instant::now();
        for q in &queries {
            let lut = fs.lut(q);
            let blocks = n.div_ceil(BLOCK);
            let _ = fs.scan_range(&lut, None, 100, 0, blocks);
        }
        let per_query = serial.elapsed().as_secs_f64() / queries.len() as f64;
        println!(
            "one thread: {:.2} ms a query, {:.2} ns a code",
            per_query * 1e3,
            per_query * 1e9 / n as f64
        );
        let par = std::time::Instant::now();
        for q in &queries {
            let _ = fs.scan(&fs.lut(q), None, 100);
        }
        let per_query = par.elapsed().as_secs_f64() / queries.len() as f64;
        println!("rayon pool: {:.2} ms a query", per_query * 1e3);
        let t = std::time::Instant::now();
        for q in &queries {
            let _ = fs.lut(q);
        }
        println!(
            "table build: {:.3} ms",
            t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64
        );
        // The 8-bit path the graph scores with, per code, on the same rows and box: a
        // 256-entry table per subspace and a gather per subspace per code.
        let mut pq8 = ProductQuantizer::new(dim, dim / 2, 256, DistanceMetric::Euclidean).unwrap();
        pq8.train(&sample, 10).unwrap();
        let codes8: Vec<u8> = rows
            .iter()
            .take(200_000)
            .flat_map(|v| pq8.encode(v).unwrap().codes)
            .collect();
        let m8 = pq8.num_subvectors;
        let t = std::time::Instant::now();
        let mut sink = 0.0f32;
        for q in &queries {
            let d = pq8.subvector_dim;
            let table: Vec<f32> = (0..m8)
                .flat_map(|s| {
                    let qs = &q[s * d..(s + 1) * d];
                    pq8.codebooks[s]
                        .iter()
                        .map(|c| {
                            qs.iter()
                                .zip(c)
                                .map(|(x, y)| (x - y) * (x - y))
                                .sum::<f32>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            for code in codes8.chunks_exact(m8) {
                let mut acc = [0.0f32; 8];
                for (i, &c) in code.iter().enumerate() {
                    acc[i % 8] += table[i * 256 + c as usize];
                }
                sink += acc.iter().sum::<f32>();
            }
        }
        println!(
            "8-bit ADC, one thread: {:.1} ns a code (sink {sink:.0})",
            t.elapsed().as_secs_f64() * 1e9 / (queries.len() * 200_000) as f64
        );
    }

    /// Candidate quality against the 8-bit codes the graph uses: how often the exact
    /// top-10 is inside each layout's top-100, on 50,000 random 192-d rows. Prints as well
    /// as asserting a floor.
    #[test]
    fn the_fast_pool_holds_the_true_neighbours_about_as_well_as_the_wide_codes() {
        let (n, dim, queries) = (50_000usize, 192usize, 30usize);
        let rows = corpus(n, dim, 3);
        let sample: Vec<Vec<f32>> = rows.iter().step_by(4).take(8_192).cloned().collect();
        let mut fs = FastScan::train(dim, DistanceMetric::Euclidean, &sample, 10).unwrap();
        for v in &rows {
            fs.push(v).unwrap();
        }
        let mut pq8 = ProductQuantizer::new(dim, dim / 2, 256, DistanceMetric::Euclidean).unwrap();
        pq8.train(&sample, 10).unwrap();
        let codes8: Vec<Vec<u8>> = rows.iter().map(|v| pq8.encode(v).unwrap().codes).collect();
        let (mut hit_fast, mut hit_wide) = (0usize, 0usize);
        for q in 0..queries {
            let query = &corpus(1, dim, 1_000 + q as u64)[0];
            let mut exact: Vec<(f32, usize)> = rows
                .iter()
                .enumerate()
                .map(|(i, v)| (v.iter().zip(query).map(|(x, y)| (x - y) * (x - y)).sum(), i))
                .collect();
            exact.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let truth: Vec<usize> = exact.iter().take(10).map(|(_, i)| *i).collect();
            let fast: Vec<usize> = fs
                .scan(&fs.lut(query), None, 100)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            // 8-bit ADC over every code, top 100.
            let d = pq8.subvector_dim;
            let table: Vec<f32> = (0..pq8.num_subvectors)
                .flat_map(|s| {
                    let qs = &query[s * d..(s + 1) * d];
                    pq8.codebooks[s]
                        .iter()
                        .map(|c| {
                            qs.iter()
                                .zip(c)
                                .map(|(x, y)| (x - y) * (x - y))
                                .sum::<f32>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut wide: Vec<(f32, usize)> = codes8
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    (
                        c.iter()
                            .enumerate()
                            .map(|(s, &b)| table[s * 256 + b as usize])
                            .sum(),
                        i,
                    )
                })
                .collect();
            wide.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let wide: Vec<usize> = wide.iter().take(100).map(|(_, i)| *i).collect();
            hit_fast += truth.iter().filter(|t| fast.contains(t)).count();
            hit_wide += truth.iter().filter(|t| wide.contains(t)).count();
        }
        let (fast, wide) = (
            hit_fast as f32 / (queries * 10) as f32,
            hit_wide as f32 / (queries * 10) as f32,
        );
        println!("top-10 inside top-100: fast-scan {fast:.3}, 8-bit ADC {wide:.3}");
        assert!(
            fast >= wide - 0.1,
            "fast-scan pool {fast} against 8-bit {wide}"
        );
    }
}
