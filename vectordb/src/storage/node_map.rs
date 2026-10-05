//! The two position spaces a filtered search crosses, mapped once instead of per query.
//!
//! A payload filter answers in the payload index's *ordinals*; the graph walks and scores
//! by *node position*; the two meet only through the record id. Resolving every matching
//! ordinal to a node by hashing its id cost one hash per match per query -- 20 ms of a
//! filter matching 200,000 rows before a single distance was taken. This map resolves each
//! node once, when the map first sees it, and is read by position after that.
//!
//! Both spaces are append-mostly: a graph upsert tombstones the old node and appends a new
//! one, and the payload index keeps an id's ordinal across an update. So `fill` only ever
//! has a tail of new nodes to resolve, each by one hash into the payload index, and an
//! ordinal that is reused for a new id is repointed when that id's node is reached. A
//! tombstoned node keeps a stale entry, which nothing reads: the walk and the scan skip
//! deleted nodes before they ask. The map belongs to one index: a rebuild replaces the
//! graph and its positions, so the wrapper's generation is checked on every fill and a
//! mismatch starts over.

use crate::search::index_traits::VectorIndex;
use crate::search::node_filter::NodeGate;

/// A position with no counterpart (yet).
pub const UNKNOWN: u32 = u32::MAX;

#[derive(Debug, Default)]
pub struct NodeMap {
    generation: u64,
    node_to_ordinal: Vec<u32>,
    ordinal_to_node: Vec<u32>,
}

impl NodeMap {
    /// Bring the map up to the index's current positions. `ordinal_of` is the payload
    /// index's lookup. `false` means the index has no positions (a brute-force scan) and the
    /// map cannot serve it.
    pub fn fill(
        &mut self,
        generation: u64,
        index: &dyn VectorIndex,
        ordinal_of: impl Fn(&str) -> Option<u32>,
    ) -> bool {
        if self.generation != generation {
            self.node_to_ordinal.clear();
            self.ordinal_to_node.clear();
            self.generation = generation;
        }
        let slots = index.slot_count();
        for idx in self.node_to_ordinal.len()..slots {
            let Some(id) = index.node_id(idx) else {
                return false;
            };
            let ordinal = ordinal_of(id).unwrap_or(UNKNOWN);
            self.node_to_ordinal.push(ordinal);
            if ordinal != UNKNOWN {
                let o = ordinal as usize;
                if o >= self.ordinal_to_node.len() {
                    self.ordinal_to_node.resize(o + 1, UNKNOWN);
                }
                self.ordinal_to_node[o] = idx as u32;
            }
        }
        true
    }

    /// The node holding the record at `ordinal`, if the map has seen one.
    #[inline]
    pub fn node_of(&self, ordinal: u32) -> Option<usize> {
        match self.ordinal_to_node.get(ordinal as usize) {
            Some(&node) if node != UNKNOWN => Some(node as usize),
            _ => None,
        }
    }

    /// The ordinal of the record at node `idx`, if the map has seen it.
    #[inline]
    pub fn ordinal_of_node(&self, idx: usize) -> Option<u32> {
        match self.node_to_ordinal.get(idx) {
            Some(&o) if o != UNKNOWN => Some(o),
            _ => None,
        }
    }

    /// Nodes known to the map.
    pub fn len(&self) -> usize {
        self.node_to_ordinal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node_to_ordinal.is_empty()
    }
}

/// A gate for the walk: a node passes when its record's ordinal is in the matched set.
pub struct OrdinalGate<'a> {
    pub map: &'a NodeMap,
    pub matched: &'a roaring::RoaringBitmap,
}

impl NodeGate for OrdinalGate<'_> {
    #[inline]
    fn allows(&self, idx: usize) -> bool {
        self.map
            .ordinal_of_node(idx)
            .is_some_and(|o| self.matched.contains(o))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::hnsw::{HnswConfig, HnswIndex};
    use crate::storage::payload_index::PayloadIndex;
    use crate::types::vector::DistanceMetric;

    fn vector(i: usize) -> Vec<f32> {
        (0..8).map(|d| ((i * 7 + d * 3) % 11) as f32).collect()
    }

    /// Nodes resolve once; an ordinal handed to a new id is repointed to the new id's node
    /// when the map reaches it; a new generation starts over.
    #[test]
    fn the_map_follows_the_graph_and_the_payload_index() {
        let mut graph = HnswIndex::new(DistanceMetric::Euclidean, Some(HnswConfig::default()));
        let mut payload = PayloadIndex::new();
        for i in 0..50usize {
            let id = format!("r{i}");
            payload.insert_record(&id, &serde_json::json!({ "k": i % 5 }), 0);
            graph.insert(id, vector(i)).unwrap();
        }
        let mut map = NodeMap::default();
        assert!(map.fill(1, &graph, |id| payload.ordinal_of(id)));
        assert_eq!(map.len(), 50);
        for i in 0..50usize {
            let ordinal = payload.ordinal_of(&format!("r{i}")).unwrap();
            let node = map.node_of(ordinal).expect("every live record has a node");
            assert_eq!(graph.node_id(node), format!("r{i}"));
            assert_eq!(map.ordinal_of_node(node), Some(ordinal));
        }

        // Delete r7 from both; a new record takes its ordinal and gets a new node.
        let freed = payload.ordinal_of("r7").unwrap();
        let old_node = map.node_of(freed).unwrap();
        payload.remove_record(&"r7".to_string(), &serde_json::json!({ "k": 2 }));
        futures::executor::block_on(crate::search::index_traits::VectorIndex::delete(
            &mut graph, "r7",
        ))
        .unwrap();
        payload.insert_record(&"new".to_string(), &serde_json::json!({ "k": 9 }), 0);
        assert_eq!(
            payload.ordinal_of("new"),
            Some(freed),
            "the free list hands the ordinal on"
        );
        graph.insert("new".to_string(), vector(99)).unwrap();
        assert!(map.fill(1, &graph, |id| payload.ordinal_of(id)));
        let new_node = map.node_of(freed).unwrap();
        assert_ne!(
            new_node, old_node,
            "the ordinal points at the new id's node"
        );
        assert_eq!(graph.node_id(new_node), "new");

        // An upsert keeps the ordinal and appends a node; the map follows.
        let kept = payload.ordinal_of("r3").unwrap();
        payload.update_record(
            &"r3".to_string(),
            &serde_json::json!({ "k": 3 }),
            &serde_json::json!({ "k": 4 }),
            0,
        );
        // The graph side of an upsert: the old node is removed, a new one appended.
        futures::executor::block_on(crate::search::index_traits::VectorIndex::delete(
            &mut graph, "r3",
        ))
        .unwrap();
        graph.insert("r3".to_string(), vector(123)).unwrap();
        assert!(map.fill(1, &graph, |id| payload.ordinal_of(id)));
        assert_eq!(payload.ordinal_of("r3"), Some(kept));
        assert_eq!(graph.node_id(map.node_of(kept).unwrap()), "r3");

        // Another generation: positions are a different graph's; start over.
        assert!(map.fill(2, &graph, |id| payload.ordinal_of(id)));
        assert_eq!(map.len(), graph.slot_count());
    }
}
