//! A filter over graph node positions, for a filtered walk or a scan of the matching nodes.
//!
//! The planner used to hand the index its matching records as a `HashSet<String>`: every
//! matching id was turned into an owned `String` under the table lock, hashed into the set,
//! and hashed again inside the index to find its node -- 100,000 allocations and 200,000
//! hashes per query on a 1%-selectivity filter over 10M rows, before any distance was
//! computed. A [`NodeFilter`] is built once from borrowed ids (one hash each, to the node
//! position) and read by position: a bit per node, so a walk tests a visited node with one
//! load and a scan walks the set bits in position order, which is also memory order for the
//! codes it scores.

use std::collections::HashSet;

/// Whether a node may be returned. The walk asks this of every node it visits.
pub trait NodeGate: Sync {
    fn allows(&self, idx: usize) -> bool;
}

impl NodeGate for HashSet<usize> {
    #[inline]
    fn allows(&self, idx: usize) -> bool {
        self.contains(&idx)
    }
}

/// A set of node positions as a bitmap.
#[derive(Debug, Clone, Default)]
pub struct NodeFilter {
    bits: Vec<u64>,
    len: usize,
}

impl NodeFilter {
    /// Room for positions below `nodes`; nothing set.
    pub fn with_capacity(nodes: usize) -> Self {
        Self {
            bits: vec![0u64; nodes.div_ceil(64)],
            len: 0,
        }
    }

    /// Mark `idx`. A position past the capacity grows the bitmap.
    #[inline]
    pub fn insert(&mut self, idx: usize) {
        let word = idx / 64;
        if word >= self.bits.len() {
            self.bits.resize(word + 1, 0);
        }
        let mask = 1u64 << (idx % 64);
        if self.bits[word] & mask == 0 {
            self.bits[word] |= mask;
            self.len += 1;
        }
    }

    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        self.bits
            .get(idx / 64)
            .is_some_and(|w| w & (1u64 << (idx % 64)) != 0)
    }

    /// The 32 positions from `start` (a multiple of 32) as a mask, bit `i` for `start + i`.
    /// Zero past the end. What a block scan tests before it loads a block.
    #[inline]
    pub fn mask32(&self, start: usize) -> u32 {
        debug_assert!(start.is_multiple_of(32));
        match self.bits.get(start / 64) {
            Some(word) if start.is_multiple_of(64) => *word as u32,
            Some(word) => (*word >> 32) as u32,
            None => 0,
        }
    }

    /// How many positions are set.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The set positions, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(w, &word)| {
            let mut bits = word;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(w * 64 + bit)
            })
        })
    }
}

impl NodeGate for NodeFilter {
    #[inline]
    fn allows(&self, idx: usize) -> bool {
        self.contains(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_contains_and_iterates_in_order() {
        let mut f = NodeFilter::with_capacity(100);
        for idx in [0usize, 63, 64, 65, 99, 7, 7, 1_000] {
            f.insert(idx);
        }
        assert_eq!(f.len(), 7, "a repeated position counts once");
        assert_eq!(
            f.iter().collect::<Vec<_>>(),
            vec![0, 7, 63, 64, 65, 99, 1_000]
        );
        assert!(f.contains(1_000) && !f.contains(999) && !f.contains(100_000));
        assert!(f.allows(63) && !f.allows(62));
        assert_eq!(f.mask32(0), (1 << 0) | (1 << 7));
        assert_eq!(f.mask32(32), 1 << 31);
        assert_eq!(f.mask32(64), 1 | 2);
        assert_eq!(f.mask32(96), 1 << 3);
        assert_eq!(f.mask32(1_024), 0);
    }
}
