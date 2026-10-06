//! A corpus whose correct answer is known in closed form, so the assertion carries no tolerance.
//!
//! Every other ranking test in this tree judges by one of two means, and both of them have a
//! floor. Either it computes a brute-force reference in the test — which makes the assertion a
//! float comparison, and float comparisons need an epsilon, and an epsilon chosen tighter than
//! the arithmetic noise reports a defect that is not there (this happened: a `1e-6` tolerance
//! against a corpus whose adjacent scores differed by less produced "87.9% unreachable", which
//! was the tolerance, not the index). Or it asserts on the *set* of ids and says nothing about
//! their order, which is the half of a search result that ranking is.
//!
//! Epsilla's `db_server.cpp` has the idea this borrows. It places points on a unit semicircle,
//! shuffles the insertion order, queries the first basis vector, and asserts
//! `result[i].id == i` — integer equality, no tolerance, no reference scan. The corpus is
//! constructed so that the true answer *is* the identity, and then a wrong order is a wrong
//! integer rather than a number close to another number.
//!
//! **Generalised to d dimensions here, deliberately.** The original is 2-dimensional, so every
//! coordinate past the second is zero: a PQ subspace over those is degenerate, a normalisation
//! bug on an all-zero tail is invisible, and a dimension-mismatch check has nothing to bite on.
//! Pick orthonormal `a`, `b` in R^d — `a` the normalised ones vector, `b` the alternating ±1
//! vector orthogonalised against it — and put the points on the great circle they span:
//!
//!     v_i = cos(πi/N)·a + sin(πi/N)·b        i = 0 .. N-1
//!
//! `|v_i| = 1` and `v_i · a = cos(πi/N)`, strictly decreasing in `i` over `[0, π)`. So the true
//! top-k for the query `a` is the ids `0..k`, in that order, exactly — and every one of the d
//! coordinates carries variance (measured: min 0.0371, max 0.0372 over the corpus below), so no
//! subspace is degenerate.
//!
//! **N is chosen from the f32 resolution, not picked round.** Adjacent similarities differ by
//! `cos(πi/N) - cos(π(i+1)/N)`, which shrinks as N grows; near the top of the ranking, where
//! the values are close to 1.0, one f32 ulp is about 6e-8. Measured smallest adjacent gap:
//!
//!       N        adjacent gap     strictly decreasing after f32 storage
//!     500          1.973e-05      yes
//!    2000          1.252e-06      yes
//!   10000          5.960e-08      no margin left — one ulp
//!
//! At N = 2000 the gap is ~20 ulp, which leaves room for the summation error of a d-term dot
//! product. At N = 10000 the corpus would be *at* the representable limit and the test would
//! start failing for arithmetic reasons on some other machine. `the_corpus_is_separable_in_f32`
//! asserts the margin up front, so raising N fails by naming the resolution rather than by
//! looking like an index defect. It earned that place immediately: the construction was checked
//! in Python first, which summed the dot product in f64 and reported N = 10000 as still
//! separable. The engine sums in f32, and at N = 10000 the *first* pair already collapses —
//! `0.99999994` against `1.0`. The guard is what makes that a named failure.
//!
//! **Verified by injection**, three ways, because a ranking test that cannot fail is decoration:
//! storing one record with another's vector fails the two exact assertions; N = 10000 fails the
//! guard; and reversing the comparator in `assemble_hits` fails all four ranking assertions.
//! One injection did *not* fail anything, which is worth knowing — dropping the final
//! `sort_by` from `keep_best` changed no answer, because `assemble_hits` sorts again afterwards.
//! The page's order is decided there and only there.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

const DIM: usize = 32;
const N: usize = 2_000;
const K: usize = 100;

/// Ids are zero-padded so that lexicographic order — which is the server's tiebreak between
/// two hits at the same distance — agrees with numeric order. Nothing below depends on that
/// for its correctness; it keeps a failure message readable.
fn id(i: usize) -> String {
    format!("v{i:05}")
}

/// Two orthonormal vectors in R^d: the normalised ones vector, and the alternating ±1 vector
/// orthogonalised against it. Computed in f64 — the corpus is the reference, so it is built at
/// a precision above the one it will be stored at.
fn basis() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![1.0f64; DIM];
    let mut b: Vec<f64> = (0..DIM)
        .map(|d| if d % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let na = norm(&a);
    for x in &mut a {
        *x /= na;
    }
    // Gram-Schmidt. For an even d the alternating vector is already orthogonal to the ones
    // vector and this subtracts zero; it is here so an odd DIM stays correct rather than
    // silently skewing the whole construction.
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    for (x, ax) in b.iter_mut().zip(&a) {
        *x -= dot * ax;
    }
    let nb = norm(&b);
    for x in &mut b {
        *x /= nb;
    }
    (a, b)
}

/// The point at rank `i`: angle `πi/N` away from `a`, in the plane `a`,`b` span.
fn vector(i: usize, a: &[f64], b: &[f64]) -> Vec<f32> {
    let theta = std::f64::consts::PI * i as f64 / N as f64;
    let (s, c) = theta.sin_cos();
    a.iter()
        .zip(b)
        .map(|(x, y)| (c * x + s * y) as f32)
        .collect()
}

/// Deterministic shuffle. The insertion order must not be the id order, or a storage layer that
/// returned rows in file order would pass every assertion here without an index being consulted
/// at all. A fixed seed rather than a random one: a ranking test that fails one run in twenty
/// teaches nobody anything.
fn shuffled(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = 0x243F_6A88_85A3_08D3u64;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    order
}

fn storage_config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn table(name: &str) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
        rerank: None,
    }
}

/// A table holding the whole corpus, inserted out of order.
async fn loaded() -> (IntelligentIndexedStorage, Vec<f64>, Vec<f64>) {
    let (a, b) = basis();
    let storage = IntelligentIndexedStorage::new_with_wal(&storage_config(), None)
        .await
        .expect("failed to open storage");
    storage.create_table(table("t")).await.unwrap();
    let records: Vec<CreateRecordRequest> = shuffled(N)
        .into_iter()
        .map(|i| CreateRecordRequest {
            id: id(i),
            vector: vector(i, &a, &b),
            sparse_vector: None,
            named_vectors: None,
            payload: Some(serde_json::json!({ "rank": i })),
            text: None,
        })
        .collect();
    storage.insert_records("t", records).await.unwrap();
    (storage, a, b)
}

fn query(vector: Vec<f32>, top_k: usize, mode: SearchMode) -> SearchQuery {
    SearchQuery {
        vector,
        top_k,
        limit: top_k,
        search_mode: mode,
        ..Default::default()
    }
}

fn f32_of(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

// ── the resolution guard ────────────────────────────────────────────────────────────────────

/// The premise every other test in this file rests on: once the corpus has been rounded to the
/// f32 the engine stores, the similarity with `a` is *still* strictly decreasing in the id.
///
/// If it is not, the analytic answer is no longer the answer and every assertion below is
/// meaningless — not failing, meaningless, which is worse. This runs first and names the cause.
#[test]
fn the_corpus_is_separable_in_f32() {
    let (a, b) = basis();
    let aq = f32_of(&a);
    // Scored the way the engine will score it: f32 inputs, summed in order.
    let sim = |i: usize| -> f32 {
        vector(i, &a, &b)
            .iter()
            .zip(&aq)
            .map(|(x, y)| x * y)
            .sum::<f32>()
    };

    let mut min_gap = f32::INFINITY;
    for i in 0..N - 1 {
        let gap = sim(i) - sim(i + 1);
        assert!(
            gap > 0.0,
            "the corpus is not separable at N={N} in f32: v{i} and v{} score {} and {}, so the \
             analytic order is not recoverable from what is stored. Lower N, or raise DIM \
             (a longer dot product does not help; the angle between adjacent points does).",
            i + 1,
            sim(i),
            sim(i + 1)
        );
        min_gap = min_gap.min(gap);
    }
    // One ulp near 1.0 is about 6e-8. Anything under ~10 ulp is a corpus with no margin, and a
    // test with no margin passes here and fails on a machine with a different fma decision.
    assert!(
        min_gap > 5e-7,
        "smallest adjacent gap is {min_gap:e}, which is within a few f32 ulp of zero. The \
         header table has the measured gaps per N; this corpus needs a smaller N."
    );
}

// ── the ranking assertions ──────────────────────────────────────────────────────────────────

/// The whole point: exact search returns ids `0..K`, in order, with nothing to compare floats
/// against.
#[tokio::test]
async fn exact_returns_the_analytic_order_exactly() {
    let (storage, a, _b) = loaded().await;

    let hits = storage
        .search("t", query(f32_of(&a), K, SearchMode::Exact))
        .await
        .unwrap();

    let got: Vec<String> = hits.hits.iter().map(|h| h.id.clone()).collect();
    let want: Vec<String> = (0..K).map(id).collect();
    assert_eq!(
        got,
        want,
        "exact search did not return the analytic order. The corpus is a semicircle: the true \
         top-{K} for this query is v00000..v{:05} in that order, by construction and not by \
         measurement.",
        K - 1
    );
}

/// Query from the middle of the arc, where the answer is a symmetric window rather than a
/// prefix — and where ranks `j-1` and `j+1` are exactly equidistant.
///
/// The prefix case can be satisfied by an index that has merely preserved insertion-independent
/// order. This one cannot: the correct answer for `v_j` is `{j-m .. j+m}`, which is a *set* the
/// construction fixes exactly, so it is still tolerance-free, while the order inside each
/// mirrored pair is a genuine tie and is left to the server's tiebreak.
#[tokio::test]
async fn a_middle_query_returns_the_analytic_window() {
    let (storage, a, b) = loaded().await;
    const J: usize = N / 2;
    const M: usize = 20;

    let hits = storage
        .search("t", query(vector(J, &a, &b), 2 * M + 1, SearchMode::Exact))
        .await
        .unwrap();

    assert_eq!(
        hits.hits[0].id,
        id(J),
        "the query point itself must rank first"
    );
    let mut got: Vec<usize> = hits
        .hits
        .iter()
        .map(|h| h.id[1..].parse::<usize>().unwrap())
        .collect();
    got.sort_unstable();
    let want: Vec<usize> = (J - M..=J + M).collect();
    assert_eq!(
        got,
        want,
        "the analytic window around v{J:05} is {}..={}; a hit outside it is a wrong answer, not \
         a near miss",
        J - M,
        J + M
    );

    // Rank distance from J must not decrease down the page. Equality is expected, and only
    // between the two members of a mirrored pair.
    let ranks: Vec<usize> = hits
        .hits
        .iter()
        .map(|h| h.id[1..].parse::<usize>().unwrap())
        .collect();
    for pair in ranks.windows(2) {
        let (d0, d1) = (pair[0].abs_diff(J), pair[1].abs_diff(J));
        assert!(
            d0 <= d1,
            "the page is not ordered by distance along the arc: v{:05} (|Δ|={d0}) came before \
             v{:05} (|Δ|={d1})",
            pair[0],
            pair[1]
        );
    }
}

/// What the four approximate modes owe, stated without an epsilon.
///
/// They are allowed to miss — that is what approximate means, and pinning a recall number here
/// would pin the corpus rather than the engine. What they are not allowed to do is return a hit
/// from *outside* the true top-2K, or return a page that is not ordered by the score printed on
/// it. Both are integer facts about a corpus whose answer is known, so neither needs a
/// tolerance; and the recall each mode achieves is printed rather than asserted, because a
/// number worth knowing is not the same as a number worth failing on.
#[tokio::test]
async fn the_approximate_modes_stay_inside_the_analytic_neighbourhood() {
    let (storage, a, _b) = loaded().await;

    for mode in [
        SearchMode::Fast,
        SearchMode::Normal,
        SearchMode::Accurate,
        SearchMode::Auto,
    ] {
        let hits = storage
            .search("t", query(f32_of(&a), K, mode))
            .await
            .unwrap();
        assert_eq!(hits.hits.len(), K, "{mode:?} returned a short page");

        let ranks: Vec<usize> = hits
            .hits
            .iter()
            .map(|h| h.id[1..].parse::<usize>().unwrap())
            .collect();

        let strays: Vec<usize> = ranks.iter().copied().filter(|r| *r >= 2 * K).collect();
        assert!(
            strays.is_empty(),
            "{mode:?} returned {strays:?}, which are outside the true top-{} of a corpus whose \
             ranking is exact by construction. Missing a neighbour is approximation; reaching \
             past twice the page for one is not.",
            2 * K
        );

        for pair in hits.hits.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "{mode:?} returned a page that is not ordered by its own score: {} then {}",
                pair[0].score,
                pair[1].score
            );
        }

        let agreed = ranks.iter().filter(|r| **r < K).count();
        println!(
            "{mode:?}: {agreed}/{K} of the analytic top-{K}, worst rank reached {}",
            ranks.iter().max().unwrap()
        );
    }
}

/// The analytic order survives a rebuild.
///
/// A rebuild constructs a second graph from the rows and swaps it in, so it is the one operation
/// that can change an answer without any write having happened. Asserting the *same exact
/// sequence* before and after is only possible because the sequence is known; against a
/// brute-force reference this test would be comparing two runs of the same code.
#[tokio::test]
async fn a_rebuild_does_not_move_the_analytic_order() {
    let (storage, a, _b) = loaded().await;
    let want: Vec<String> = (0..K).map(id).collect();

    let before: Vec<String> = storage
        .search("t", query(f32_of(&a), K, SearchMode::Exact))
        .await
        .unwrap()
        .hits
        .iter()
        .map(|h| h.id.clone())
        .collect();
    assert_eq!(before, want);

    storage.rebuild_index("t").await.unwrap();

    let after: Vec<String> = storage
        .search("t", query(f32_of(&a), K, SearchMode::Exact))
        .await
        .unwrap()
        .hits
        .iter()
        .map(|h| h.id.clone())
        .collect();
    assert_eq!(
        after, want,
        "the analytic order did not survive a rebuild — the graph was replaced and the answer \
         moved with it"
    );
}
