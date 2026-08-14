//! PyTorch-based CLIP implementation using PyO3 bindings
//!
//! This implementation uses Python's open_clip library through PyO3 FFI
//! to provide working multimodal embeddings while waiting for ONNX Runtime 2.0 to stabilize.
//!
//! **PyO3 v0.22 API**: This code uses PyO3 v0.22 bound API

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::embeddings::EmbeddingError;

/// PyTorch-based CLIP encoder
pub struct ClipPyTorch {
    model_name: String,
    device: String,
    dimension: usize,

    // Python objects cached for reuse
    py_module: Option<PyObject>,
    py_model: Option<PyObject>,
    py_preprocess: Option<PyObject>,
    py_tokenizer: Option<PyObject>,
}

impl ClipPyTorch {
    /// Create a new PyTorch CLIP encoder
    ///
    /// # Arguments
    /// * `model_name` - CLIP model variant (e.g., "ViT-B-32", "ViT-B-16")
    /// * `device` - Device to run on ("cpu", "cuda", "cuda:0", etc.)
    ///
    /// # Returns
    /// Result containing the initialized encoder or an error
    pub fn new(model_name: &str, device: &str) -> Result<Self, EmbeddingError> {
        // Determine embedding dimension from model name
        let dimension = match model_name {
            "ViT-B-32" | "ViT-B-16" => 512,
            "ViT-L-14" => 768,
            "ViT-H-14" => 1024,
            _ => 512, // Default
        };

        Ok(Self {
            model_name: model_name.to_string(),
            device: device.to_string(),
            dimension,
            py_module: None,
            py_model: None,
            py_preprocess: None,
            py_tokenizer: None,
        })
    }

    /// Initialize Python runtime and load the CLIP model
    ///
    /// This should be called once before encoding. It loads:
    /// - open_clip Python module
    /// - CLIP model weights
    /// - Preprocessing transforms
    /// - Text tokenizer
    pub fn initialize(&mut self) -> Result<(), EmbeddingError> {
        Python::with_gil(|py| {
            // Import open_clip using PyO3 v0.22 API
            let open_clip = PyModule::import_bound(py, "open_clip")
                .map_err(|e| EmbeddingError::ModelLoadError(
                    format!("Failed to import open_clip: {}. Please install: pip install open-clip-torch", e)
                ))?;

            // Create model and transforms
            // Returns: (model, _, preprocess)
            let create_result = open_clip.call_method(
                "create_model_and_transforms",
                (&self.model_name,),
                None,
            ).map_err(|e| EmbeddingError::ModelLoadError(
                format!("Failed to create model {}: {}", self.model_name, e)
            ))?;

            // Extract model, preprocess from tuple
            let model = create_result.get_item(0)
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to get model: {}", e)))?;
            let preprocess = create_result.get_item(2)
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to get preprocess: {}", e)))?;

            // Move model to device
            model.call_method1("to", (&self.device,))
                .map_err(|e| EmbeddingError::ModelLoadError(
                    format!("Failed to move model to device {}: {}", self.device, e)
                ))?;

            // Set model to eval mode
            model.call_method0("eval")
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to set eval mode: {}", e)))?;

            // Get tokenizer
            let tokenizer = open_clip.call_method(
                "get_tokenizer",
                (&self.model_name,),
                None,
            ).map_err(|e| EmbeddingError::ModelLoadError(
                format!("Failed to get tokenizer: {}", e)
            ))?;

            // Cache Python objects (convert to PyObject)
            self.py_module = Some(open_clip.unbind().into());
            self.py_model = Some(model.unbind());
            self.py_preprocess = Some(preprocess.unbind());
            self.py_tokenizer = Some(tokenizer.unbind());

            tracing::info!(
                "Initialized CLIP model {} on device {} (dimension: {})",
                self.model_name, self.device, self.dimension
            );

            Ok(())
        })
    }

    /// Encode an image to embedding vector
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image bytes (JPEG, PNG, etc.)
    ///
    /// # Returns
    /// Normalized embedding vector of length `self.dimension`
    pub fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
        if self.py_model.is_none() {
            return Err(EmbeddingError::ModelLoadError(
                "Model not initialized. Call initialize() first.".to_string()
            ));
        }

        Python::with_gil(|py| {
            // Load image from bytes using PIL
            let pil = PyModule::import_bound(py, "PIL.Image")
                .map_err(|e| EmbeddingError::ModelLoadError(
                    format!("Failed to import PIL: {}. Please install: pip install pillow", e)
                ))?;

            let io = PyModule::import_bound(py, "io")
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to import io: {}", e)))?;

            // Create BytesIO object from image bytes
            let bytes_io = io.call_method1("BytesIO", (PyBytes::new_bound(py, image_bytes),))
                .map_err(|e| EmbeddingError::InvalidInput(format!("Failed to create BytesIO: {}", e)))?;

            // Open image
            let image = pil.call_method1("open", (bytes_io,))
                .map_err(|e| EmbeddingError::InvalidInput(format!("Failed to open image: {}", e)))?;

            // Convert to RGB (in case of RGBA, grayscale, etc.)
            let image_rgb = image.call_method0("convert")
                .and_then(|m| m.call1(("RGB",)))
                .map_err(|e| EmbeddingError::InvalidInput(format!("Failed to convert to RGB: {}", e)))?;

            // Preprocess image
            let preprocess = self.py_preprocess.as_ref().unwrap().bind(py);
            let preprocessed = preprocess.call1((image_rgb,))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to preprocess: {}", e)))?;

            // Add batch dimension: unsqueeze(0)
            let batched = preprocessed.call_method1("unsqueeze", (0,))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to unsqueeze: {}", e)))?;

            // Move to device
            let tensor = batched.call_method1("to", (&self.device,))
                .map_err(|e| EmbeddingError::InferenceError(
                    format!("Failed to move tensor to device: {}", e)
                ))?;

            // Encode with torch.no_grad() context
            let torch = PyModule::import_bound(py, "torch")
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to import torch: {}", e)))?;

            let model = self.py_model.as_ref().unwrap().bind(py);

            // Run inference with no_grad
            let features = torch.call_method0("no_grad")
                .and_then(|no_grad_ctx| {
                    // Enter context and encode
                    no_grad_ctx.call_method0("__enter__")?;
                    let result = model.call_method1("encode_image", (tensor,));
                    let _ = no_grad_ctx.call_method("__exit__", (py.None(), py.None(), py.None()), None);
                    result
                })
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to encode image: {}", e)))?;

            // Normalize: features / features.norm(dim=-1, keepdim=True)
            let norm_kwargs = pyo3::types::PyDict::new_bound(py);
            norm_kwargs.set_item("dim", -1)
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to set dim: {}", e)))?;
            norm_kwargs.set_item("keepdim", true)
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to set keepdim: {}", e)))?;

            let norm = features.call_method("norm", (), Some(&norm_kwargs))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to compute norm: {}", e)))?;

            let normalized = features.call_method1("div", (norm,))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to normalize: {}", e)))?;

            // Convert to CPU numpy array
            let cpu_tensor = normalized.call_method0("cpu")
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to move to CPU: {}", e)))?;

            let numpy_array = cpu_tensor.call_method0("numpy")
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to convert to numpy: {}", e)))?;

            // Flatten and extract as Vec<f32>
            let flattened = numpy_array.call_method0("flatten")
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to flatten: {}", e)))?;

            let embedding: Vec<f32> = flattened.call_method0("tolist")
                .and_then(|list| list.extract())
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to extract embedding: {}", e)))?;

            // Validate dimension
            if embedding.len() != self.dimension {
                return Err(EmbeddingError::InferenceError(
                    format!("Expected dimension {}, got {}", self.dimension, embedding.len())
                ));
            }

            Ok(embedding)
        })
    }

    /// Encode text to embedding vector
    ///
    /// # Arguments
    /// * `text` - Input text string
    ///
    /// # Returns
    /// Normalized embedding vector of length `self.dimension`
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if self.py_model.is_none() {
            return Err(EmbeddingError::ModelLoadError(
                "Model not initialized. Call initialize() first.".to_string()
            ));
        }

        Python::with_gil(|py| {
            // Tokenize text
            let tokenizer = self.py_tokenizer.as_ref().unwrap().bind(py);
            let tokens = tokenizer.call1((vec![text],))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to tokenize: {}", e)))?;

            // Move to device
            let tensor = tokens.call_method1("to", (&self.device,))
                .map_err(|e| EmbeddingError::InferenceError(
                    format!("Failed to move tokens to device: {}", e)
                ))?;

            // Encode with torch.no_grad() context
            let torch = PyModule::import_bound(py, "torch")
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to import torch: {}", e)))?;

            let model = self.py_model.as_ref().unwrap().bind(py);

            // Run inference with no_grad
            let features = torch.call_method0("no_grad")
                .and_then(|no_grad_ctx| {
                    no_grad_ctx.call_method0("__enter__")?;
                    let result = model.call_method1("encode_text", (tensor,));
                    let _ = no_grad_ctx.call_method("__exit__", (py.None(), py.None(), py.None()), None);
                    result
                })
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to encode text: {}", e)))?;

            // Normalize
            let norm_kwargs = pyo3::types::PyDict::new_bound(py);
            norm_kwargs.set_item("dim", -1)
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to set dim: {}", e)))?;
            norm_kwargs.set_item("keepdim", true)
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to set keepdim: {}", e)))?;

            let norm = features.call_method("norm", (), Some(&norm_kwargs))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to compute norm: {}", e)))?;

            let normalized = features.call_method1("div", (norm,))
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to normalize: {}", e)))?;

            // Convert to CPU numpy array
            let cpu_tensor = normalized.call_method0("cpu")
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to move to CPU: {}", e)))?;

            let numpy_array = cpu_tensor.call_method0("numpy")
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to convert to numpy: {}", e)))?;

            // Flatten and extract
            let flattened = numpy_array.call_method0("flatten")
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to flatten: {}", e)))?;

            let embedding: Vec<f32> = flattened.call_method0("tolist")
                .and_then(|list| list.extract())
                .map_err(|e| EmbeddingError::InferenceError(format!("Failed to extract embedding: {}", e)))?;

            // Validate dimension
            if embedding.len() != self.dimension {
                return Err(EmbeddingError::InferenceError(
                    format!("Expected dimension {}, got {}", self.dimension, embedding.len())
                ));
            }

            Ok(embedding)
        })
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get device
    pub fn device(&self) -> &str {
        &self.device
    }
}

// Thread-safe wrapper for use in async contexts
#[derive(Clone)]
pub struct ClipPyTorchWrapper {
    inner: Arc<RwLock<ClipPyTorch>>,
}

impl ClipPyTorchWrapper {
    pub fn new(model_name: &str, device: &str) -> Result<Self, EmbeddingError> {
        let mut pytorch = ClipPyTorch::new(model_name, device)?;
        pytorch.initialize()?;

        Ok(Self {
            inner: Arc::new(RwLock::new(pytorch)),
        })
    }

    pub fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
        let pytorch = self.inner.read();
        pytorch.encode_image(image_bytes)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let pytorch = self.inner.read();
        pytorch.encode_text(text)
    }

    pub fn dimension(&self) -> usize {
        let pytorch = self.inner.read();
        pytorch.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Python + open_clip installation
    fn test_clip_pytorch_initialization() {
        let mut clip = ClipPyTorch::new("ViT-B-32", "cpu").unwrap();
        let result = clip.initialize();

        // Will fail if Python/open_clip not installed, which is expected
        match result {
            Ok(_) => println!("Successfully initialized CLIP PyTorch"),
            Err(e) => println!("Failed to initialize (expected if Python not configured): {}", e),
        }
    }

    #[test]
    fn test_clip_pytorch_creation() {
        let clip = ClipPyTorch::new("ViT-B-32", "cpu").unwrap();
        assert_eq!(clip.dimension(), 512);
        assert_eq!(clip.model_name(), "ViT-B-32");
        assert_eq!(clip.device(), "cpu");
    }
}
