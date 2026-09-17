use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use tokio::runtime::Runtime;
use uuid::Uuid;

use crate::{
    config::StorageConfig,
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, Filter, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

/// Python wrapper for DXMT VectorDB
#[pyclass]
pub struct DXMTVectorDB {
    storage: Arc<IntelligentIndexedStorage>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl DXMTVectorDB {
    /// Create a new DXMT VectorDB instance
    #[new]
    #[pyo3(signature = (path="/tmp/dxmt_db", **kwargs))]
    fn new(path: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // Create runtime for async operations
        let runtime = Runtime::new().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to create runtime: {}", e))
        })?;

        // Parse configuration from kwargs or use default memory config
        let config = if let Some(dict) = kwargs {
            parse_config(dict)?
        } else {
            // Use a minimal memory-based configuration by default
            use crate::config::MemoryConfig;
            StorageConfig {
                r#type: "memory".to_string(),
                memory: Some(MemoryConfig {
                    max_size_mb: 512,
                    eviction_policy: "lru".to_string(),
                }),
                ssd: None,
                object_storage: None,
            }
        };

        // Create storage
        let storage = runtime
            .block_on(async { IntelligentIndexedStorage::new(&config).await })
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to create storage: {}", e))
            })?;

        Ok(Self {
            storage: Arc::new(storage),
            runtime: Arc::new(runtime),
        })
    }

    /// Create a new collection (table)
    #[pyo3(signature = (name, dimension, distance=None))]
    fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance: Option<&str>,
    ) -> PyResult<()> {
        let distance_metric = parse_distance_metric(distance)?;

        let config = TableConfig {
            name: name.to_string(),
            dimension,
            distance_metric,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
        };

        self.runtime
            .block_on(async { self.storage.create_table(config).await })
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to create collection: {}", e))
            })?;

        Ok(())
    }

    /// Insert vectors with payloads
    #[pyo3(signature = (collection_name, vectors, payloads=None, ids=None))]
    fn insert(
        &self,
        collection_name: &str,
        vectors: Vec<Vec<f32>>,
        payloads: Option<Vec<Bound<'_, PyDict>>>,
        ids: Option<Vec<String>>,
    ) -> PyResult<Vec<String>> {
        // Validate inputs
        let num_vectors = vectors.len();
        if let Some(ref p) = payloads {
            if p.len() != num_vectors {
                return Err(PyErr::new::<PyValueError, _>(
                    "Vectors and payloads must have same length",
                ));
            }
        }
        if let Some(ref i) = ids {
            if i.len() != num_vectors {
                return Err(PyErr::new::<PyValueError, _>(
                    "Vectors and ids must have same length",
                ));
            }
        }

        // Create records
        let mut records = Vec::new();
        for (i, vector) in vectors.into_iter().enumerate() {
            let id = if let Some(ref ids) = ids {
                ids[i].clone()
            } else {
                Uuid::new_v4().to_string()
            };

            let payload = if let Some(ref payloads) = payloads {
                dict_to_json(&payloads[i])?
            } else {
                serde_json::json!({})
            };

            records.push(CreateRecordRequest {
                id,
                vector,
                payload: Some(payload),
                sparse_vector: None,
            });
        }

        // Insert records
        let inserted_ids = self.runtime.block_on(async {
            let results = self
                .storage
                .insert_records(collection_name, records)
                .await
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Failed to insert records: {}", e))
                })?;
            Ok::<Vec<String>, PyErr>(results.into_iter().map(|r| r.id).collect())
        })?;

        Ok(inserted_ids)
    }

    /// Search for similar vectors
    #[pyo3(signature = (collection_name, query_vector, limit=None, filters=None))]
    fn search(
        &self,
        collection_name: &str,
        query_vector: Vec<f32>,
        limit: Option<usize>,
        filters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PyObject>> {
        let filter = if let Some(f) = filters {
            Some(dict_to_filter(f)?)
        } else {
            None
        };

        let query = SearchQuery {
            vector: query_vector,
            top_k: limit.unwrap_or(10),
            limit: limit.unwrap_or(10),
            filter,
            with_payload: true,
            with_vector: false,
        };

        let results = self
            .runtime
            .block_on(async { self.storage.search(collection_name, query).await })
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to search: {}", e)))?;

        // Convert results to Python objects
        Python::with_gil(|py| {
            results
                .hits
                .into_iter()
                .map(|m| {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("id", m.id).unwrap();
                    dict.set_item("score", m.score).unwrap();
                    if let Some(payload) = m.payload {
                        dict.set_item("payload", json_to_py(py, &payload).unwrap())
                            .unwrap();
                    }
                    Ok(dict.into_any().unbind())
                })
                .collect()
        })
    }

    /// Get a vector by ID
    fn get(&self, collection_name: &str, vector_id: &str) -> PyResult<PyObject> {
        let id = vector_id.to_string();

        let record = self
            .runtime
            .block_on(async { self.storage.get_record(collection_name, &id).await })
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to get record: {}", e)))?;

        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &record.id).unwrap();
            dict.set_item("vector", &record.vector).unwrap();
            dict.set_item("payload", json_to_py(py, &record.payload).unwrap())
                .unwrap();
            Ok(dict.into_any().unbind())
        })
    }

    /// Update a vector
    #[pyo3(signature = (collection_name, vector_id, vector=None, payload=None))]
    fn update(
        &self,
        collection_name: &str,
        vector_id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let id = vector_id.to_string();

        self.runtime.block_on(async {
            // Get existing record
            let record = self
                .storage
                .get_record(collection_name, &id)
                .await
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Failed to get record: {}", e))
                })?;

            // Create update request with new values
            let new_vector = vector.unwrap_or(record.vector);
            let new_payload = if let Some(p) = payload {
                Some(dict_to_json(p)?)
            } else {
                Some(record.payload)
            };

            let update_request = CreateRecordRequest {
                id: id.clone(),
                vector: new_vector,
                payload: new_payload,
                sparse_vector: record.sparse_vector,
            };

            // Update record
            self.storage
                .update_record(collection_name, &id, update_request)
                .await
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Failed to update record: {}", e))
                })?;

            Ok(())
        })
    }

    /// Delete a vector
    fn delete(&self, collection_name: &str, vector_id: &str) -> PyResult<()> {
        let id = vector_id.to_string();

        self.runtime
            .block_on(async { self.storage.delete_records(collection_name, vec![id]).await })
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to delete record: {}", e))
            })?;

        Ok(())
    }

    /// List all collections
    fn list_collections(&self) -> PyResult<Vec<String>> {
        let tables = self
            .runtime
            .block_on(async { self.storage.list_tables().await })
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to list collections: {}", e))
            })?;

        Ok(tables.into_iter().map(|t| t.name).collect())
    }

    /// Delete a collection
    fn delete_collection(&self, name: &str) -> PyResult<()> {
        self.runtime
            .block_on(async { self.storage.delete_table(name).await })
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to delete collection: {}", e))
            })?;

        Ok(())
    }

    /// Get collection info
    fn collection_info(&self, name: &str) -> PyResult<PyObject> {
        let table = self
            .runtime
            .block_on(async { self.storage.get_table(name).await })
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to get collection info: {}", e))
            })?;

        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("name", table.name).unwrap();
            dict.set_item("dimension", table.dimension).unwrap();
            dict.set_item("distance_metric", format!("{:?}", table.distance_metric))
                .unwrap();
            dict.set_item("record_count", table.record_count).unwrap();
            Ok(dict.into_any().unbind())
        })
    }

    /// List all vectors in a collection
    #[pyo3(signature = (collection_name, limit=None, filters=None))]
    fn list_vectors(
        &self,
        collection_name: &str,
        limit: Option<usize>,
        filters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PyObject>> {
        let filter = filters.map(dict_to_filter).transpose()?;
        let max_results = limit.unwrap_or(usize::MAX);

        let ids = self
            .runtime
            .block_on(async { self.storage.list_all_record_ids(collection_name).await })
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to list records: {}", e)))?;

        Python::with_gil(|py| {
            let mut results = Vec::new();
            for id in ids.into_iter().take(max_results) {
                let record = self
                    .runtime
                    .block_on(async { self.storage.get_record(collection_name, &id).await })
                    .map_err(|e| {
                        PyErr::new::<PyValueError, _>(format!("Failed to get record: {}", e))
                    })?;

                // Apply filter if provided
                if let Some(ref f) = filter {
                    if !record.matches_filter(f) {
                        continue;
                    }
                }

                let dict = PyDict::new(py);
                dict.set_item("id", &record.id)?;
                dict.set_item(
                    "vector",
                    record.vector.iter().map(|v| *v).collect::<Vec<_>>(),
                )?;
                if let Some(payload) = record.payload.as_ref() {
                    dict.set_item(
                        "metadata",
                        serde_json::to_string(payload).unwrap_or_default(),
                    )?;
                }
                results.push(dict.into());
            }
            Ok(results)
        })
    }

    /// Reset database (delete all collections)
    fn reset(&self) -> PyResult<()> {
        let tables = self
            .runtime
            .block_on(async { self.storage.list_tables().await })
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to list tables: {}", e)))?;

        for table in tables {
            self.runtime
                .block_on(async { self.storage.delete_table(&table.name).await })
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Failed to delete table: {}", e))
                })?;
        }

        Ok(())
    }
}

// Helper functions

fn parse_distance_metric(distance: Option<&str>) -> PyResult<DistanceMetric> {
    match distance {
        None | Some("cosine") => Ok(DistanceMetric::Cosine),
        Some("euclidean") => Ok(DistanceMetric::Euclidean),
        Some("dot") | Some("dot_product") => Ok(DistanceMetric::Dot),
        Some("manhattan") => Ok(DistanceMetric::Manhattan),
        Some(d) => Err(PyErr::new::<PyValueError, _>(format!(
            "Unknown distance metric: {}",
            d
        ))),
    }
}

fn parse_config(dict: &Bound<'_, PyDict>) -> PyResult<StorageConfig> {
    use crate::config::MemoryConfig;

    let storage_type = dict
        .get_item("type")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_else(|| "memory".to_string());

    let max_size_mb = dict
        .get_item("memory_limit_mb")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<u64>().ok())
        .unwrap_or(512);

    Ok(StorageConfig {
        r#type: storage_type,
        memory: Some(MemoryConfig {
            max_size_mb,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    })
}

fn dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        let json_value = py_to_json(&value)?;
        map.insert(key_str, json_value);
    }
    Ok(serde_json::Value::Object(map))
}

fn dict_to_filter(dict: &Bound<'_, PyDict>) -> PyResult<Filter> {
    use crate::types::record::{Condition, FilterOperation};

    // Accepts: {"field": "value"} (equality shorthand) or
    // {"must": [{"key": "...", "op": "eq", "value": ...}], ...}
    let parse_conditions = |items: &Bound<'_, PyAny>| -> PyResult<Vec<Condition>> {
        let mut conditions = Vec::new();
        if let Ok(list) = items.downcast::<PyList>() {
            for item in list.iter() {
                if let Ok(d) = item.downcast::<PyDict>() {
                    let key = d
                        .get_item("key")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<String>().ok())
                        .unwrap_or_default();
                    let op_str = d
                        .get_item("op")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<String>().ok())
                        .unwrap_or_else(|| "eq".to_string());
                    let val = d
                        .get_item("value")
                        .ok()
                        .flatten()
                        .map(|v| py_to_json(&v))
                        .transpose()?
                        .unwrap_or(serde_json::Value::Null);
                    let op = match op_str.as_str() {
                        "eq" | "=" | "==" => FilterOperation::Equal(val),
                        "ne" | "!=" => FilterOperation::NotEqual(val),
                        "gt" | ">" => FilterOperation::GreaterThan(val),
                        "gte" | ">=" => FilterOperation::GreaterThanOrEqual(val),
                        "lt" | "<" => FilterOperation::LessThan(val),
                        "lte" | "<=" => FilterOperation::LessThanOrEqual(val),
                        _ => FilterOperation::Equal(val),
                    };
                    conditions.push(Condition { key, operation: op });
                }
            }
        }
        Ok(conditions)
    };

    let must = dict
        .get_item("must")
        .ok()
        .flatten()
        .map(|v| parse_conditions(&v))
        .transpose()?
        .filter(|v| !v.is_empty());

    let must_not = dict
        .get_item("must_not")
        .ok()
        .flatten()
        .map(|v| parse_conditions(&v))
        .transpose()?
        .filter(|v| !v.is_empty());

    let should = dict
        .get_item("should")
        .ok()
        .flatten()
        .map(|v| parse_conditions(&v))
        .transpose()?
        .filter(|v| !v.is_empty());

    // Equality shorthand: {"field_name": value, ...} for keys that aren't "must"/"must_not"/"should"
    let mut shorthand: Vec<Condition> = Vec::new();
    for (k, v) in dict.iter() {
        let key = k.extract::<String>()?;
        if matches!(key.as_str(), "must" | "must_not" | "should") {
            continue;
        }
        let val = py_to_json(&v)?;
        shorthand.push(Condition {
            key,
            operation: FilterOperation::Equal(val),
        });
    }

    let combined_must = match (must, shorthand.is_empty()) {
        (Some(mut m), false) => {
            m.extend(shorthand);
            Some(m)
        }
        (Some(m), true) => Some(m),
        (None, false) => Some(shorthand),
        (None, true) => None,
    };

    Ok(Filter {
        must: combined_must,
        must_not,
        should,
    })
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)),
        ))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let mut vec = Vec::new();
        for item in list.iter() {
            vec.push(py_to_json(&item)?);
        }
        Ok(serde_json::Value::Array(vec))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        dict_to_json(&dict)
    } else {
        Ok(serde_json::Value::Null)
    }
}

fn json_to_py(py: Python, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.to_object(py))
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (key, value) in map {
                dict.set_item(key, json_to_py(py, value)?)?;
            }
            Ok(dict.to_object(py))
        }
    }
}

/// Python module initialization
#[pymodule]
fn dxmt_vectordb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DXMTVectorDB>()?;
    Ok(())
}
