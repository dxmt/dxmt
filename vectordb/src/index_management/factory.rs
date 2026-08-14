//! IndexFactory - Creates and configures index instances
//!
//! Responsible for:
//! - Creating new index instances of any type
//! - Configuring indexes with adaptive parameters
//! - Validating configurations
//! - Error handling for invalid requests

use crate::adaptive_index::AdaptiveConfig;
use crate::search::IndexWrapper;
use crate::types::table::IndexType;
use crate::error::{Error, Result};

/// Factory for creating index instances
pub struct IndexFactory {
    config: AdaptiveConfig,
}

impl IndexFactory {
    /// Create a new IndexFactory with configuration
    pub fn new(config: AdaptiveConfig) -> Self {
        Self { config }
    }

    /// Create an index instance of the specified type
    ///
    /// # Arguments
    /// * `index_type` - Type of index to create (HNSW, IVF, DiskANN, BruteForce)
    ///
    /// # Returns
    /// * `Result<IndexWrapper>` - Wrapped index instance or error
    ///
    /// # Example
    /// ```ignore
    /// let factory = IndexFactory::new(config);
    /// let index = factory.create_index(IndexType::Hnsw)?;
    /// ```
    pub fn create_index(&self, index_type: IndexType) -> Result<IndexWrapper> {
        match index_type {
            IndexType::Hnsw => self.create_hnsw_index(),
            IndexType::Ivf => self.create_ivf_index(),
            IndexType::BruteForce => self.create_bruteforce_index(),
            IndexType::OptimizedHnsw => self.create_hnsw_index(), // Use regular HNSW for now
            IndexType::Lsh => {
                // LSH not yet implemented, fallback to BruteForce
                self.create_bruteforce_index()
            },
            IndexType::IvfPq => {
                // IVF-PQ not yet implemented, fallback to IVF
                self.create_ivf_index()
            },
        }
    }

    /// Create HNSW index with adaptive parameters
    fn create_hnsw_index(&self) -> Result<IndexWrapper> {
        // Extract HNSW parameters from config
        let m = self.config.hnsw_m;
        let ef_construction = self.config.hnsw_ef_construction;

        // Validate HNSW parameters
        if !(4..=64).contains(&m) {
            return Err(Error::InvalidConfig(
                format!("HNSW M must be between 4 and 64, got {}", m)
            ));
        }

        if ef_construction < 100 {
            return Err(Error::InvalidConfig(
                format!("HNSW ef_construction must be at least 100, got {}", ef_construction)
            ));
        }

        // Import necessary types
        use crate::search::HnswIndex;
        use crate::search::hnsw::HnswConfig;

        // Create HNSW index with configuration
        let hnsw_config = HnswConfig::from_params(m, ef_construction, self.config.hnsw_ef_search);
        let hnsw_index = HnswIndex::new(self.config.distance_metric, Some(hnsw_config));

        // Wrap in IndexWrapper for thread-safe access
        Ok(IndexWrapper::new(Box::new(hnsw_index)))
    }

    /// Create IVF index with adaptive parameters
    fn create_ivf_index(&self) -> Result<IndexWrapper> {
        use crate::search::ivf::{IvfIndex, IvfConfig};

        // Extract IVF parameters from config
        let nlist = self.config.ivf_n_lists;
        let nprobe = self.config.ivf_n_probes;

        // Validate IVF parameters
        if nlist < 100 {
            return Err(Error::InvalidConfig(
                format!("IVF nlist must be at least 100, got {}", nlist)
            ));
        }

        if nprobe < 1 || nprobe > nlist {
            return Err(Error::InvalidConfig(
                format!("IVF nprobe must be between 1 and {}, got {}", nlist, nprobe)
            ));
        }

        // Create IVF configuration
        let ivf_config = IvfConfig {
            nlist,
            nprobe,
            ..IvfConfig::default()
        };

        // Create IVF index
        let ivf_index = IvfIndex::new(self.config.distance_metric, Some(ivf_config));

        // Wrap and return
        Ok(IndexWrapper::new(Box::new(ivf_index)))
    }

    /// Create DiskANN index with adaptive parameters
    #[allow(dead_code)]
    fn create_diskann_index(&self) -> Result<IndexWrapper> {
        use crate::search::BruteForceIndex;

        // DiskANN typically requires filesystem access which we'll implement
        // as a variant of BruteForce for now, with SSD storage backend
        // This is a simplified implementation - full DiskANN would need path configuration

        let bruteforce_index = BruteForceIndex::new(self.config.distance_metric);

        // Wrap as DiskANN variant (using BruteForce backend)
        Ok(IndexWrapper::new(Box::new(bruteforce_index)))
    }

    /// Create BruteForce index
    fn create_bruteforce_index(&self) -> Result<IndexWrapper> {
        use crate::search::BruteForceIndex;

        // Create BruteForce index - simple linear scan
        let bruteforce_index = BruteForceIndex::new(self.config.distance_metric);

        // Wrap for thread-safe access
        Ok(IndexWrapper::new(Box::new(bruteforce_index)))
    }

    /// Validate configuration for index creation
    pub fn validate_config(&self) -> Result<()> {
        // Validate all parameters
        if self.config.distance_metric.to_string().is_empty() {
            return Err(Error::InvalidConfig("Distance metric not set".to_string()));
        }

        // Validate HNSW parameters
        if self.config.hnsw_m < 4 || self.config.hnsw_m > 64 {
            return Err(Error::InvalidConfig(
                format!("HNSW M must be between 4 and 64, got {}", self.config.hnsw_m),
            ));
        }

        if self.config.hnsw_ef_construction < 100 {
            return Err(Error::InvalidConfig(
                "HNSW ef_construction must be at least 100".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creation() {
        let config = AdaptiveConfig::default();
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_ok());
    }

    #[test]
    fn test_invalid_hnsw_m() {
        let mut config = AdaptiveConfig::default();
        config.hnsw_m = 2; // Invalid: too small
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_err());
    }

    #[test]
    fn test_invalid_ef_construction() {
        let mut config = AdaptiveConfig::default();
        config.hnsw_ef_construction = 50; // Invalid: too small
        let factory = IndexFactory::new(config);
        assert!(factory.validate_config().is_err());
    }
}
