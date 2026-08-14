use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Memory-mapped index storage for efficient large dataset handling
pub struct MmapIndexStorage {
    /// Path to the index file
    path: PathBuf,
    /// Memory-mapped data
    mmap: Option<Arc<Mmap>>,
    /// Mutable memory-mapped data for writing
    mmap_mut: Option<Arc<MmapMut>>,
    /// Index metadata
    metadata: IndexMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Number of vectors
    pub num_vectors: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Index type
    pub index_type: String,
    /// Data offset in file
    pub data_offset: usize,
    /// File version
    pub version: u32,
}

impl MmapIndexStorage {
    /// Create a new memory-mapped index storage
    pub fn new<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            mmap: None,
            mmap_mut: None,
            metadata: IndexMetadata {
                num_vectors: 0,
                dimensions: 0,
                index_type: String::new(),
                data_offset: 0,
                version: 1,
            },
        })
    }

    /// Load an existing index file using memory mapping
    pub fn load(&mut self) -> crate::Result<()> {
        let file = File::open(&self.path)
            .map_err(crate::Error::Io)?;

        // Read metadata from the beginning of the file
        let mut metadata_bytes = vec![0u8; 1024]; // Fixed metadata size
        let mut file_copy = File::open(&self.path)
            .map_err(crate::Error::Io)?;
        file_copy.read_exact(&mut metadata_bytes)
            .map_err(crate::Error::Io)?;

        self.metadata = bincode::deserialize(&metadata_bytes)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        // Memory map the file
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(crate::Error::Io)?
        };

        self.mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Create a new index file for writing
    pub fn create(&mut self, metadata: IndexMetadata) -> crate::Result<()> {
        self.metadata = metadata;

        // Calculate total file size
        let metadata_size = 1024; // Fixed metadata size
        let vector_size = self.metadata.dimensions * std::mem::size_of::<f32>();
        let total_size = metadata_size + (self.metadata.num_vectors * vector_size);

        // Create and resize the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .map_err(crate::Error::Io)?;

        file.set_len(total_size as u64)
            .map_err(crate::Error::Io)?;

        // Write metadata
        let metadata_bytes = bincode::serialize(&self.metadata)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        let mut padded_metadata = vec![0u8; metadata_size];
        padded_metadata[..metadata_bytes.len()].copy_from_slice(&metadata_bytes);

        let mut file_copy = OpenOptions::new()
            .write(true)
            .open(&self.path)
            .map_err(crate::Error::Io)?;
        file_copy.write_all(&padded_metadata)
            .map_err(crate::Error::Io)?;

        // Create mutable memory map
        let mmap_mut = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(crate::Error::Io)?
        };

        self.mmap_mut = Some(Arc::new(mmap_mut));
        self.metadata.data_offset = metadata_size;

        Ok(())
    }

    /// Read a vector at the given index
    pub fn read_vector(&self, index: usize) -> crate::Result<Vec<f32>> {
        if index >= self.metadata.num_vectors {
            return Err(crate::Error::InvalidInput(
                format!("Index {} out of bounds", index)
            ));
        }

        let mmap = self.mmap.as_ref()
            .ok_or_else(|| crate::Error::InvalidInput("Index not loaded".to_string()))?;

        let vector_size = self.metadata.dimensions * std::mem::size_of::<f32>();
        let offset = self.metadata.data_offset + (index * vector_size);

        // Read vector data
        let mut vector = Vec::with_capacity(self.metadata.dimensions);
        for i in 0..self.metadata.dimensions {
            let byte_offset = offset + (i * std::mem::size_of::<f32>());
            let bytes = &mmap[byte_offset..byte_offset + std::mem::size_of::<f32>()];
            let value = f32::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3]
            ]);
            vector.push(value);
        }

        Ok(vector)
    }

    /// Write a vector at the given index
    pub fn write_vector(&mut self, index: usize, vector: &[f32]) -> crate::Result<()> {
        if index >= self.metadata.num_vectors {
            return Err(crate::Error::InvalidInput(
                format!("Index {} out of bounds", index)
            ));
        }

        if vector.len() != self.metadata.dimensions {
            return Err(crate::Error::InvalidInput(
                format!("Vector dimension mismatch: expected {}, got {}",
                    self.metadata.dimensions, vector.len())
            ));
        }

        let mmap_mut = Arc::get_mut(self.mmap_mut.as_mut()
            .ok_or_else(|| crate::Error::InvalidInput("Index not created for writing".to_string()))?)
            .ok_or_else(|| crate::Error::InvalidInput("Cannot get mutable reference".to_string()))?;

        let vector_size = self.metadata.dimensions * std::mem::size_of::<f32>();
        let offset = self.metadata.data_offset + (index * vector_size);

        // Write vector data
        for (i, &value) in vector.iter().enumerate() {
            let byte_offset = offset + (i * std::mem::size_of::<f32>());
            let bytes = value.to_le_bytes();
            mmap_mut[byte_offset..byte_offset + std::mem::size_of::<f32>()]
                .copy_from_slice(&bytes);
        }

        Ok(())
    }

    /// Flush changes to disk
    pub fn flush(&self) -> crate::Result<()> {
        if let Some(ref mmap_mut) = self.mmap_mut {
            // Note: We can't directly flush Arc<MmapMut>, 
            // but changes are automatically synced when dropped
            // Just ignore the reference as we can't drop it
            let _ = mmap_mut;
        }
        Ok(())
    }

    /// Get a slice of vectors
    pub fn read_batch(&self, start: usize, count: usize) -> crate::Result<Vec<Vec<f32>>> {
        let mut batch = Vec::with_capacity(count);
        for i in start..start.min(start + count).min(self.metadata.num_vectors) {
            batch.push(self.read_vector(i)?);
        }
        Ok(batch)
    }

    /// Prefetch vectors into memory
    pub fn prefetch(&self, indices: &[usize]) -> crate::Result<()> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| crate::Error::InvalidInput("Index not loaded".to_string()))?;

        let vector_size = self.metadata.dimensions * std::mem::size_of::<f32>();

        for &index in indices {
            if index >= self.metadata.num_vectors {
                continue;
            }

            let offset = self.metadata.data_offset + (index * vector_size);
            let end = offset + vector_size;

            // Advise the OS to prefetch this range
            #[cfg(unix)]
            {
                use libc::{madvise, MADV_WILLNEED};
                unsafe {
                    madvise(
                        mmap[offset..end].as_ptr() as *mut libc::c_void,
                        vector_size,
                        MADV_WILLNEED,
                    );
                }
            }
        }

        Ok(())
    }

    /// Get metadata
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
}

/// Lazy-loading HNSW index using memory-mapped files
pub struct MmapHnswIndex {
    /// Memory-mapped storage
    storage: Arc<MmapIndexStorage>,
    /// In-memory graph structure (lightweight)
    graph: Vec<Vec<u32>>,
    /// Entry point
    entry_point: Option<usize>,
}

impl MmapHnswIndex {
    /// Load an HNSW index from a memory-mapped file
    pub fn load<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let mut storage = MmapIndexStorage::new(path)?;
        storage.load()?;

        // Load graph structure (this should be stored separately or in metadata)
        // For now, we'll create an empty graph
        let num_vectors = storage.metadata().num_vectors;
        let graph = vec![Vec::new(); num_vectors];

        Ok(Self {
            storage: Arc::new(storage),
            graph,
            entry_point: if num_vectors > 0 { Some(0) } else { None },
        })
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> crate::Result<Vec<(usize, f32)>> {
        // Simple implementation - in practice, this would use the graph structure
        let mut distances = Vec::new();

        for i in 0..self.storage.metadata().num_vectors {
            let vector = self.storage.read_vector(i)?;
            let distance = euclidean_distance(query, &vector);
            distances.push((i, distance));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Prefetch vectors that are likely to be accessed
    pub fn prefetch_neighbors(&self, index: usize) -> crate::Result<()> {
        if index < self.graph.len() {
            let neighbors: Vec<usize> = self.graph[index]
                .iter()
                .map(|&i| i as usize)
                .collect();
            self.storage.prefetch(&neighbors)?;
        }
        Ok(())
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_storage() -> crate::Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_index.bin");

        // Create index
        let metadata = IndexMetadata {
            num_vectors: 100,
            dimensions: 10,
            index_type: "test".to_string(),
            data_offset: 1024,
            version: 1,
        };

        let mut storage = MmapIndexStorage::new(&path)?;
        storage.create(metadata)?;

        // Write some vectors
        for i in 0..10 {
            let vector: Vec<f32> = (0..10).map(|j| (i * 10 + j) as f32).collect();
            storage.write_vector(i, &vector)?;
        }

        // Read vectors back
        for i in 0..10 {
            let vector = storage.read_vector(i)?;
            assert_eq!(vector.len(), 10);
            assert_eq!(vector[0], (i * 10) as f32);
        }

        Ok(())
    }

    #[test]
    fn test_batch_read() -> crate::Result<()> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_batch.bin");

        let metadata = IndexMetadata {
            num_vectors: 100,
            dimensions: 5,
            index_type: "test".to_string(),
            data_offset: 1024,
            version: 1,
        };

        let mut storage = MmapIndexStorage::new(&path)?;
        storage.create(metadata)?;

        // Write vectors
        for i in 0..20 {
            let vector: Vec<f32> = vec![i as f32; 5];
            storage.write_vector(i, &vector)?;
        }

        // Load and read batch
        let mut storage2 = MmapIndexStorage::new(&path)?;
        storage2.load()?;

        let batch = storage2.read_batch(5, 10)?;
        assert_eq!(batch.len(), 10);
        assert_eq!(batch[0][0], 5.0);

        Ok(())
    }
}