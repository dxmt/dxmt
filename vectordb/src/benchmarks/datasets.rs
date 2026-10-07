//! Dataset utilities for ANN benchmarks.
//!
//! This module intentionally provides a lightweight, self‑contained
//! implementation so the main crate can compile (e.g. in Docker builds)
//! even if the full benchmarking stack is not being used yet.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

type IoResult<T> = std::result::Result<T, std::io::Error>;

/// In‑memory representation of a benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub dimension: usize,
    pub train_vectors: Vec<Vec<f32>>,
    pub test_queries: Vec<Vec<f32>>,
    /// Ground‑truth neighbors (indices into train_vectors)
    pub ground_truth: Vec<Vec<usize>>,
}

impl Dataset {
    /// Create a small synthetic dataset for tests and demos.
    ///
    /// The vectors and their ground truth are deterministic.  Ground truth is
    /// computed from the generated vectors (cosine distance on the normalized
    /// vectors), rather than sampled independently.  An independently sampled
    /// neighbor list is not a valid ANN benchmark: it makes a perfect index
    /// appear to have poor recall and can contain duplicate neighbors.
    pub fn synthetic(
        name: &str,
        dimension: usize,
        train_size: usize,
        test_size: usize,
        k: usize,
    ) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        // A fixed seed keeps benchmark results reproducible across runs and
        // makes failures debuggable.  The dataset name and shape are mixed in
        // so two named datasets do not accidentally share the same vectors.
        let mut seed = 0x4458_4D54_u64; // "DXMT"
        for byte in name.as_bytes() {
            seed = seed.rotate_left(5) ^ u64::from(*byte);
        }
        seed ^= (dimension as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        seed ^= (train_size as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        seed ^= (test_size as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut train_vectors = Vec::with_capacity(train_size);
        for _ in 0..train_size {
            let mut v = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                v.push(rng.gen::<f32>());
            }
            // simple L2 normalization
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            train_vectors.push(v);
        }

        let mut test_queries = Vec::with_capacity(test_size);
        for _ in 0..test_size {
            let mut v = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                v.push(rng.gen::<f32>());
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            test_queries.push(v);
        }

        // Exact ground truth for cosine similarity.  All generated vectors are
        // normalized, so maximizing the dot product is equivalent to
        // minimizing cosine distance.  Tie-break by index for deterministic
        // ordering.
        let mut ground_truth = Vec::with_capacity(test_size);
        for query in &test_queries {
            let mut scores: Vec<(usize, f32)> = train_vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| {
                    let similarity = query.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
                    (index, similarity)
                })
                .collect();
            scores.sort_by(|(index_a, score_a), (index_b, score_b)| {
                score_b
                    .partial_cmp(score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| index_a.cmp(index_b))
            });
            let neighbors = scores
                .into_iter()
                .take(k.min(train_size))
                .map(|(index, _)| index)
                .collect();
            ground_truth.push(neighbors);
        }

        Self {
            name: name.to_string(),
            dimension,
            train_vectors,
            test_queries,
            ground_truth,
        }
    }

    pub fn train_size(&self) -> usize {
        self.train_vectors.len()
    }

    pub fn test_size(&self) -> usize {
        self.test_queries.len()
    }

    pub fn k(&self) -> usize {
        self.ground_truth.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Print basic statistics – primarily used in tests.
    pub fn print_stats(&self) {
        println!(
            "Dataset \"{}\": dim={}, train={}, test={}, k={}",
            self.name,
            self.dimension,
            self.train_size(),
            self.test_size(),
            self.k()
        );
    }
}

/// High‑level metadata for a dataset in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub dimension: usize,
    pub train_size: usize,
    pub test_size: usize,
    /// Optional human‑readable description.
    pub description: Option<String>,
    /// Optional source URL (e.g. ann-benchmarks or HF datasets).
    pub source_url: Option<String>,
}

/// Size buckets used for simple filtering in the registry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SizeCategory {
    Small,
    Medium,
    Large,
    VeryLarge,
}

/// Dimensionality buckets used for simple filtering in the registry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DimensionCategory {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Simple in‑memory registry of known datasets.
#[derive(Debug, Default)]
pub struct DatasetRegistry {
    by_name: HashMap<String, DatasetMetadata>,
}

impl DatasetRegistry {
    pub fn new() -> Self {
        let mut registry = DatasetRegistry {
            by_name: HashMap::new(),
        };

        // These defaults are chosen to match the expectations in tests/docs.
        registry.insert(DatasetMetadata {
            name: "sift-1k".to_string(),
            dimension: 128,
            train_size: 1_000,
            test_size: 100,
            description: Some("SIFT 1K subset".to_string()),
            source_url: None,
        });
        registry.insert(DatasetMetadata {
            name: "glove-25".to_string(),
            dimension: 25,
            train_size: 1_000_000,
            test_size: 10_000,
            description: Some("GloVe 25D word embeddings".to_string()),
            source_url: None,
        });
        registry.insert(DatasetMetadata {
            name: "fashion-mnist".to_string(),
            dimension: 784,
            train_size: 60_000,
            test_size: 10_000,
            description: Some("Fashion-MNIST images".to_string()),
            source_url: None,
        });
        registry.insert(DatasetMetadata {
            name: "deep-1m".to_string(),
            dimension: 96,
            train_size: 1_000_000,
            test_size: 10_000,
            description: Some("Deep learning embeddings (1M)".to_string()),
            source_url: None,
        });
        registry.insert(DatasetMetadata {
            name: "gist-960".to_string(),
            dimension: 960,
            train_size: 1_000_000,
            test_size: 1_000,
            description: Some("GIST 960D descriptors (1M)".to_string()),
            source_url: None,
        });
        registry.insert(DatasetMetadata {
            name: "lastfm".to_string(),
            dimension: 65,
            train_size: 292_385,
            test_size: 50_000,
            description: Some("Last.fm music recommendations".to_string()),
            source_url: None,
        });

        registry
    }

    fn insert(&mut self, meta: DatasetMetadata) {
        self.by_name.insert(meta.name.clone(), meta);
    }

    pub fn list_datasets(&self) -> Vec<String> {
        let mut names: Vec<String> = self.by_name.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get(&self, name: &str) -> Option<DatasetMetadata> {
        self.by_name.get(name).cloned()
    }

    pub fn by_size(&self, category: SizeCategory) -> Vec<DatasetMetadata> {
        self.by_name
            .values()
            .filter(|m| match category {
                SizeCategory::Small => m.train_size < 10_000,
                SizeCategory::Medium => (10_000..100_000).contains(&m.train_size),
                SizeCategory::Large => (100_000..1_000_000).contains(&m.train_size),
                SizeCategory::VeryLarge => m.train_size >= 1_000_000,
            })
            .cloned()
            .collect()
    }

    pub fn by_dimension(&self, category: DimensionCategory) -> Vec<DatasetMetadata> {
        self.by_name
            .values()
            .filter(|m| match category {
                DimensionCategory::Low => m.dimension < 50,
                DimensionCategory::Medium => (50..256).contains(&m.dimension),
                DimensionCategory::High => (256..512).contains(&m.dimension),
                DimensionCategory::VeryHigh => m.dimension >= 512,
            })
            .cloned()
            .collect()
    }
}

/// Metadata about a single cached entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntryMetadata {
    pub key: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
}

/// Summary statistics for the cache directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub file_count: usize,
    pub total_size_bytes: u64,
}

impl CacheStats {
    pub fn format_size(&self) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;
        const GB: f64 = MB * 1024.0;

        let size = self.total_size_bytes as f64;
        if size >= GB {
            format!("{:.2} GiB", size / GB)
        } else if size >= MB {
            format!("{:.2} MiB", size / MB)
        } else if size >= KB {
            format!("{:.2} KiB", size / KB)
        } else {
            format!("{} B", self.total_size_bytes)
        }
    }
}

/// Very small on‑disk cache with zstd compression.
#[derive(Debug)]
pub struct DatasetCache {
    root: PathBuf,
}

impl DatasetCache {
    /// Create a cache rooted at the given directory (which must already exist).
    pub fn with_directory(dir: &Path) -> IoResult<Self> {
        fs::create_dir_all(dir)?;
        Ok(Self {
            root: dir.to_path_buf(),
        })
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.zst"))
    }

    fn metadata_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.meta.json"))
    }

    pub fn cache_file(&self, source: &Path, key: &str) -> IoResult<()> {
        let data = fs::read(source)?;
        let original_size = data.len() as u64;

        let compressed = zstd::encode_all(&data[..], 0)?;
        let compressed_size = compressed.len() as u64;

        {
            let mut writer = BufWriter::new(File::create(self.entry_path(key))?);
            writer.write_all(&compressed)?;
            writer.flush()?;
        }

        let ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            1.0
        };

        let meta = CacheEntryMetadata {
            key: key.to_string(),
            original_size,
            compressed_size,
            compression_ratio: ratio,
        };

        let meta_bytes = serde_json::to_vec(&meta)?;
        fs::write(self.metadata_path(key), meta_bytes)?;

        Ok(())
    }

    pub fn is_cached(&self, key: &str) -> bool {
        self.entry_path(key).exists()
    }

    pub fn read_cached(&self, key: &str) -> IoResult<Vec<u8>> {
        let mut file = BufReader::new(File::open(self.entry_path(key))?);
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let decompressed = zstd::decode_all(&buf[..])?;
        Ok(decompressed)
    }

    pub fn get_metadata(&self, key: &str) -> IoResult<CacheEntryMetadata> {
        let bytes = fs::read(self.metadata_path(key))?;
        let meta = serde_json::from_slice(&bytes)?;
        Ok(meta)
    }

    pub fn list_cached(&self) -> IoResult<Vec<String>> {
        let mut keys = Vec::new();
        if self.root.exists() {
            for entry in fs::read_dir(&self.root)? {
                let entry = entry?;
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "zst" {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            keys.push(stem.to_string());
                        }
                    }
                }
            }
        }
        Ok(keys)
    }

    pub fn stats(&self) -> IoResult<CacheStats> {
        let mut file_count = 0usize;
        let mut total_size = 0u64;

        if self.root.exists() {
            for entry in fs::read_dir(&self.root)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|e| e == "zst") {
                    file_count += 1;
                    total_size += entry.metadata()?.len();
                }
            }
        }

        Ok(CacheStats {
            file_count,
            total_size_bytes: total_size,
        })
    }
}

/// Minimal fvecs writer used by tests/docs.
pub fn write_fvecs(path: &Path, vectors: &[Vec<f32>]) -> IoResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for v in vectors {
        let dim = v.len() as u32;
        file.write_all(&dim.to_le_bytes())?;
        for &x in v {
            file.write_all(&x.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}

/// Minimal fvecs reader used by tests/docs.
pub fn load_fvecs(path: &Path) -> IoResult<Vec<Vec<f32>>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut result = Vec::new();

    loop {
        let mut dim_bytes = [0u8; 4];
        if file.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        let mut buf = vec![0u8; dim * 4];
        file.read_exact(&mut buf)?;

        let mut vec = Vec::with_capacity(dim);
        for chunk in buf.as_chunks::<4>().0 {
            vec.push(f32::from_le_bytes(*chunk));
        }
        result.push(vec);
    }

    Ok(result)
}
