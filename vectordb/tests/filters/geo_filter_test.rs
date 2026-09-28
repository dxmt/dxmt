//! Filtering by place, and the one property that makes it safe: the index and the matcher
//! agree exactly.
//!
//! This filter is the first one whose value is an *object*. Every other condition reads a
//! scalar, which the payload index keys directly, so "which records match" is a lookup. A
//! coordinate pair means nothing one number at a time — a bounding box of latitudes and a
//! bounding box of longitudes is not a circle — so the geo half of the index stores the two
//! together and runs the real distance test in place.
//!
//! That matters more here than it looks, because **the search path takes the index's answer**:
//! when the index answered a filter, post-filtering is switched off, so a superset from the
//! index is not a slower query, it is wrong records returned as right ones. The agreement test
//! below is the one that would catch that, and it is deliberately randomised rather than
//! hand-picked — a hand-picked corpus tests the boundaries someone thought of.

use std::collections::HashSet;

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{
            Condition, CreateRecordRequest, Filter, FilterOperation, GeoBox, GeoPoint, GeoRadius,
            SearchMode, SearchQuery,
        },
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use serde_json::json;

const DIM: usize = 4;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 128,
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
    }
}

async fn open() -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new(&config()).await.unwrap()
}

fn record(id: &str, payload: serde_json::Value) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        // One direction for every record, so the ranking never decides what the filter should.
        vector: vec![1.0, 0.0, 0.0, 0.0],
        text: None,
        sparse_vector: None,
        named_vectors: None,
        payload: Some(payload),
    }
}

fn at(id: &str, lat: f64, lon: f64) -> CreateRecordRequest {
    record(id, json!({"shop": {"lat": lat, "lon": lon}}))
}

fn radius(lat: f64, lon: f64, radius_m: f64) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "shop".to_string(),
            operation: FilterOperation::GeoRadius(GeoRadius {
                center: GeoPoint { lat, lon },
                radius_m,
            }),
        }]),
        must_not: None,
        should: None,
    }
}

fn bbox(north: f64, west: f64, south: f64, east: f64) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "shop".to_string(),
            operation: FilterOperation::GeoBox(GeoBox {
                top_left: GeoPoint {
                    lat: north,
                    lon: west,
                },
                bottom_right: GeoPoint {
                    lat: south,
                    lon: east,
                },
            }),
        }]),
        must_not: None,
        should: None,
    }
}

/// The ids a filtered search returns, through whichever path the planner picks.
async fn found(
    storage: &IntelligentIndexedStorage,
    filter: Filter,
    mode: SearchMode,
) -> Vec<String> {
    let mut ids: Vec<String> = storage
        .search(
            "places",
            SearchQuery {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                top_k: 1000,
                limit: 1000,
                filter: Some(filter),
                search_mode: mode,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    ids
}

/// The ids `records/list` returns for a filter, which is the index's answer with no search in
/// front of it.
async fn listed(storage: &IntelligentIndexedStorage, filter: Filter) -> HashSet<String> {
    storage
        .list_records("places", Some(&filter), 10_000, 0, false)
        .await
        .unwrap()
        .0
        .into_iter()
        .map(|record| record.id)
        .collect()
}

/// Berlin, and points at known distances from it.
async fn berlin() -> IntelligentIndexedStorage {
    let storage = open().await;
    storage.create_table(table("places")).await.unwrap();
    storage
        .insert_records(
            "places",
            vec![
                at("brandenburg", 52.5163, 13.3777),
                // ~1.1 km east of the gate.
                at("reichstag", 52.5186, 13.3761),
                // ~8 km south-east.
                at("tempelhof", 52.4730, 13.4030),
                // Another city entirely.
                at("hamburg", 53.5511, 9.9937),
                at("munich", 48.1351, 11.5820),
                // A record whose field is not a point at all. It must not match, and it must
                // not break the walk either: a payload has no declared shape.
                record("no_shop", json!({"shop": "a string"})),
                record("no_field", json!({"other": 1})),
            ],
        )
        .await
        .unwrap();
    storage
}

#[tokio::test]
async fn a_radius_returns_what_is_inside_it_and_nothing_else() {
    let storage = berlin().await;

    let near = found(
        &storage,
        radius(52.5163, 13.3777, 2_000.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(
        near,
        vec!["brandenburg", "reichstag"],
        "2 km around the gate"
    );

    let wider = found(
        &storage,
        radius(52.5163, 13.3777, 10_000.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(wider, vec!["brandenburg", "reichstag", "tempelhof"]);

    // Zero metres is a point, not "everything": the centre only, and only if something is
    // exactly there. This is the degenerate case a bounding-box implementation gets wrong.
    let none = found(&storage, radius(52.5163, 13.3777, 0.0), SearchMode::Normal).await;
    assert_eq!(none, vec!["brandenburg"]);

    let empty = found(&storage, radius(0.0, 0.0, 1_000.0), SearchMode::Normal).await;
    assert!(
        empty.is_empty(),
        "the Gulf of Guinea has no shops: {empty:?}"
    );
}

#[tokio::test]
async fn a_box_is_not_a_circle_and_the_corners_are_the_difference() {
    let storage = berlin().await;

    // A box around Berlin that excludes Hamburg and Munich.
    let inside = found(
        &storage,
        bbox(52.60, 13.20, 52.40, 13.50),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(inside, vec!["brandenburg", "reichstag", "tempelhof"]);

    // A record in the box's north-east corner, about 13.8 km from its centre, where the
    // circle inscribed in the box reaches 10.2 km. It is in the box and out of the circle, and
    // a radius answered as a bounding box would return it.
    storage
        .insert_records("places", vec![at("corner", 52.59, 13.49)])
        .await
        .unwrap();
    let boxed = found(
        &storage,
        bbox(52.60, 13.20, 52.40, 13.50),
        SearchMode::Normal,
    )
    .await;
    assert!(boxed.contains(&"corner".to_string()), "{boxed:?}");

    let circle = found(&storage, radius(52.50, 13.35, 10_000.0), SearchMode::Normal).await;
    assert!(
        !circle.contains(&"corner".to_string()),
        "the corner of the box is inside the circle, so the radius is boxy: {circle:?}"
    );
    assert!(circle.contains(&"brandenburg".to_string()), "{circle:?}");
}

#[tokio::test]
async fn every_search_path_gives_the_same_answer() {
    // A filtered search has several routes through the planner — exact scoring of a small
    // candidate set, an in-graph filtered walk, a post-filtered fetch — and which one runs
    // depends on the estimated selectivity. They must not disagree, and the geo index is
    // reached by some of them and not others: `SearchMode::Exact` bypasses the index entirely
    // and scores every row through `Record::matches_filter`, which is the matcher this is
    // checked against.
    let storage = berlin().await;
    for (name, filter) in [
        ("tight", radius(52.5163, 13.3777, 2_000.0)),
        ("wide", radius(52.5163, 13.3777, 400_000.0)),
        ("box", bbox(53.0, 13.0, 52.0, 14.0)),
        ("nothing", radius(-33.86, 151.20, 100.0)),
    ] {
        let approximate = found(&storage, filter.clone(), SearchMode::Normal).await;
        let exact = found(&storage, filter, SearchMode::Exact).await;
        assert_eq!(
            approximate, exact,
            "the two paths disagree on the {name} filter"
        );
    }
}

#[tokio::test]
async fn the_index_and_the_matcher_agree_on_a_random_corpus() {
    // The property that makes this safe, asked of points and circles nobody chose. A
    // hand-written case tests the boundary someone thought of; a boundary this is likely to
    // get wrong is one nobody did — a point a metre outside, a circle spanning the equator, a
    // radius wider than the corpus.
    let storage = open().await;
    storage.create_table(table("places")).await.unwrap();

    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut points = Vec::new();
    let mut records = Vec::new();
    for i in 0..400 {
        let lat = next() * 180.0 - 90.0;
        let lon = next() * 360.0 - 180.0;
        points.push((format!("p{i}"), lat, lon));
        records.push(at(&format!("p{i}"), lat, lon));
    }
    storage.insert_records("places", records).await.unwrap();

    for case in 0..60 {
        let lat = next() * 180.0 - 90.0;
        let lon = next() * 360.0 - 180.0;
        // From a few hundred metres to a third of the way round the earth, so both the narrow
        // band and the "every longitude" case are exercised.
        let radius_m = 10f64.powf(2.5 + next() * 4.5);
        let centre = GeoPoint { lat, lon };

        let expected: HashSet<String> = points
            .iter()
            .filter(|(_, plat, plon)| {
                centre.distance_m(&GeoPoint {
                    lat: *plat,
                    lon: *plon,
                }) <= radius_m
            })
            .map(|(id, _, _)| id.clone())
            .collect();

        // Through `records/list` rather than `search`: the question here is what the *index*
        // answers, and a vector search also has to find the records, which on 400 identical
        // vectors it does only approximately. Mixing the two would make a recall shortfall
        // look like an index bug -- it did, the first time this ran.
        let got: HashSet<String> = listed(&storage, radius(lat, lon, radius_m)).await;

        assert_eq!(
            got,
            expected,
            "case {case}: circle at ({lat:.3}, {lon:.3}) of {radius_m:.0} m. Missing {:?}, \
             extra {:?}",
            expected.difference(&got).collect::<Vec<_>>(),
            got.difference(&expected).collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn a_box_may_cross_the_antimeridian() {
    // Reading such a box as empty is the obvious bug, and it hides in the Pacific where
    // nobody looks. `west > east` means the box wraps.
    let storage = open().await;
    storage.create_table(table("places")).await.unwrap();
    storage
        .insert_records(
            "places",
            vec![
                at("fiji", -17.75, 177.5),
                at("samoa", -13.76, -172.1),
                at("hawaii", 21.3, -157.8),
            ],
        )
        .await
        .unwrap();

    let wrapped = found(
        &storage,
        bbox(0.0, 170.0, -30.0, -170.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(wrapped, vec!["fiji", "samoa"], "the box did not wrap");

    // And a radius near the date line, where the wrap is the distance function's problem and
    // not the box's. Fiji to Samoa is about 1,150 km across the line.
    let across = found(
        &storage,
        radius(-17.75, 177.5, 1_300_000.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(across, vec!["fiji", "samoa"]);
}

#[tokio::test]
async fn a_geo_filter_survives_an_update_and_a_delete() {
    // The index is maintained, not rebuilt, so a moved shop has to leave its old cell. A stale
    // entry here returns a record the matcher would reject — which post-filtering is switched
    // off for.
    let storage = berlin().await;

    storage
        .apply_update(
            "places",
            &"tempelhof".to_string(),
            dxmt_vectordb::storage::RecordUpdate {
                payload: Some(json!({"shop": {"lat": 52.5170, "lon": 13.3780}})),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let near = found(
        &storage,
        radius(52.5163, 13.3777, 2_000.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(
        near,
        vec!["brandenburg", "reichstag", "tempelhof"],
        "the moved record did not arrive"
    );
    let old = found(
        &storage,
        radius(52.4730, 13.4030, 500.0),
        SearchMode::Normal,
    )
    .await;
    assert!(
        old.is_empty(),
        "the record is still filed at its old address: {old:?}"
    );

    storage
        .delete_records("places", vec!["reichstag".to_string()])
        .await
        .unwrap();
    let after = found(
        &storage,
        radius(52.5163, 13.3777, 2_000.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(after, vec!["brandenburg", "tempelhof"]);
}

#[tokio::test]
async fn geometry_that_is_not_a_place_is_refused_rather_than_answered_empty() {
    // A latitude of 452 is a swapped pair or a degrees/radians slip. An empty page is a
    // plausible answer to a real question, so it is the worst possible response to this one.
    let storage = berlin().await;
    for (why, filter) in [
        ("latitude past the pole", radius(452.0, 13.0, 100.0)),
        ("longitude past the line", radius(52.0, 613.0, 100.0)),
        ("a negative radius", radius(52.0, 13.0, -5.0)),
        ("a box with no height", bbox(52.0, 13.0, 53.0, 14.0)),
    ] {
        let err = storage
            .search(
                "places",
                SearchQuery {
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    top_k: 10,
                    limit: 10,
                    filter: Some(filter),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.status_code(), 400, "{why}: {err}");
    }
}

#[tokio::test]
async fn a_coordinate_written_as_an_array_is_not_silently_read_backwards() {
    // GeoJSON is [lon, lat]; every map UI is [lat, lon]. Anywhere both numbers are a plausible
    // latitude the two are indistinguishable, and guessing puts a record on the wrong
    // continent with nothing about the result looking wrong. So an array is not a point, and
    // the record simply does not match — the same as any other field holding the wrong shape.
    let storage = open().await;
    storage.create_table(table("places")).await.unwrap();
    storage
        .insert_records(
            "places",
            vec![
                record("array", json!({"shop": [52.5163, 13.3777]})),
                at("object", 52.5163, 13.3777),
            ],
        )
        .await
        .unwrap();

    let here = found(
        &storage,
        radius(52.5163, 13.3777, 1_000.0),
        SearchMode::Normal,
    )
    .await;
    assert_eq!(here, vec!["object"]);
}

#[tokio::test]
async fn the_nested_numeric_filters_still_work_beside_the_geo_index() {
    // `shop.lat` was the only way to ask a geographic question before this existed, so taking
    // it away to save the duplicated storage would break the workaround everyone had found.
    let storage = berlin().await;
    let ids = found(
        &storage,
        Filter {
            must: Some(vec![Condition {
                key: "shop.lat".to_string(),
                operation: FilterOperation::GreaterThan(json!(53.0)),
            }]),
            must_not: None,
            should: None,
        },
        SearchMode::Normal,
    )
    .await;
    assert_eq!(ids, vec!["hamburg"]);
}

#[tokio::test]
async fn a_geo_filter_combines_with_the_others() {
    let storage = open().await;
    storage.create_table(table("places")).await.unwrap();
    storage
        .insert_records(
            "places",
            vec![
                record(
                    "cafe",
                    json!({"shop": {"lat": 52.5163, "lon": 13.3777}, "kind": "cafe"}),
                ),
                record(
                    "bank",
                    json!({"shop": {"lat": 52.5164, "lon": 13.3778}, "kind": "bank"}),
                ),
                record(
                    "far_cafe",
                    json!({"shop": {"lat": 48.1351, "lon": 11.5820}, "kind": "cafe"}),
                ),
            ],
        )
        .await
        .unwrap();

    let mut filter = radius(52.5163, 13.3777, 1_000.0);
    filter.must.as_mut().unwrap().push(Condition {
        key: "kind".to_string(),
        operation: FilterOperation::Equal(json!("cafe")),
    });
    assert_eq!(
        found(&storage, filter, SearchMode::Normal).await,
        vec!["cafe"]
    );

    // And as a negation: everything that is *not* near the gate.
    let away = Filter {
        must: None,
        must_not: radius(52.5163, 13.3777, 1_000.0).must,
        should: None,
    };
    assert_eq!(
        found(&storage, away, SearchMode::Normal).await,
        vec!["far_cafe"]
    );
}
