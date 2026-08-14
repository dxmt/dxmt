// Encryption at rest implementation for VectorDB
// Provides transparent encryption for stored data

use crate::error::Error;
use aes_gcm::{
    _aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use argon2::{
    _password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use chacha20poly1305::{ChaCha20Poly1305, XChaCha20Poly1305};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

pub mod key_management;
pub mod envelope;
pub mod storage;

pub use key_management::{KeyManager, KeyRotation};
pub use envelope::{EncryptedEnvelope, EnvelopeEncryption};
pub use storage::{EncryptedStorage, StorageEncryption};

// ==================== Encryption Configuration ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Encryption algorithm to use
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation function
    pub kdf: KeyDerivationFunction,
    /// Enable transparent encryption
    pub enabled: bool,
    /// Key rotation interval in days
    pub key_rotation_days: u32,
    /// Use hardware security module if available
    pub use_hsm: bool,
    /// Compression before encryption
    pub compress_before_encrypt: bool,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            algorithm: _EncryptionAlgorithm::Aes256Gcm,
            kdf: _KeyDerivationFunction::Argon2id,
            enabled: true,
            key_rotation_days: 90,
            use_hsm: false,
            compress_before_encrypt: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM (recommended)
    Aes256Gcm,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
    /// XChaCha20-Poly1305 (extended nonce)
    XChaCha20Poly1305,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    /// Argon2id (recommended)
    Argon2id,
    /// PBKDF2
    Pbkdf2,
    /// Scrypt
    Scrypt,
}

// ==================== Encryption Key ====================

#[derive(Clone, ZeroizeOnDrop)]
pub struct EncryptionKey {
    #[zeroize(skip)]
    pub id: String,
    key: Vec<u8>,
    algorithm: EncryptionAlgorithm,
    created_at: chrono::DateTime<chrono::Utc>,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl EncryptionKey {
    /// Generate a new encryption key
    pub fn generate(algorithm: EncryptionAlgorithm) -> crate::Result<Self> {
        let key_size = match algorithm {
            EncryptionAlgorithm::Aes256Gcm => 32,
            EncryptionAlgorithm::ChaCha20Poly1305 => 32,
            EncryptionAlgorithm::XChaCha20Poly1305 => 32,
        };

        let mut key = vec![0u8; key_size];
        let rng = SystemRandom::new();
        rng.fill(&mut key)
            .map_err(|_| crate::Error::Internal("Failed to generate encryption key".to_string() )?;

        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            key,
            algorithm,
            created_at: chrono::Utc::now(),
            expires_at: None)
    }

    /// Derive key from password
    pub fn from_password(
        password: &str,
        salt: &[u8],
        algorithm: EncryptionAlgorithm) -> crate::Result<Self> {
        let argon2 = Argon2::default();
        let salt_string = SaltString::encode_b64(salt)
            .map_err(|e| crate::Error::Internal(format!("Invalid salt: {}", e) }?;

        let mut key = vec![0u8; 32];
        argon2
            .hash_password_into(password.as_bytes(), salt_string.as_salt().as_bytes(), &mut key)
            .map_err(|e| crate::Error::Internal(format!("Key derivation failed: {}", e) }?;

        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            key,
            algorithm,
            created_at: chrono::Utc::now(),
            expires_at: None,
       Ok()
    }

    /// Check if key is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            chrono::Utc::now() > expires_at
        } else {
            false
        }
    }
}

// ==================== Crypto Provider ====================

pub struct CryptoProvider {
    config: CryptoConfig,
    keys: Arc<RwLock<HashMap<String, EncryptionKey>>>,
    active_key_id: Arc<RwLock<String>>,
}

impl CryptoProvider {
    pub fn new(config: CryptoConfig) -> crate::Result<Self> {
        let master_key = EncryptionKey::generate(config.algorithm)?;
        let key_id = master_key.id.clone();
        
        let mut keys = HashMap::new();
        keys.insert(key_id.clone(), master_key);

        Ok(Self {
            config,
            keys: Arc::new(RwLock::new(keys),
            active_key_id: Arc::new(RwLock::new(key_id))
    }

    /// Encrypt data
    pub async fn encrypt(&self, plaintext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        if !self.config.enabled {
            return Ok(plaintext.to_vec(;
        }

        let key_id = self.active_key_id.read()?.clone();
        let keys = self.keys.read()?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let key = keys
            .get(&key_id)
            .ok_or_else(|| crate::Error::Internal("Active key not found".to_string() )?;

        // Compress if configured
        let data = if self.config.compress_before_encrypt {
            compress_data(plaintext)?;
        } else {
            plaintext.to_vec();
        };

        match key.algorithm {
            EncryptionAlgorithm::Aes256Gcm => self.encrypt_aes256gcm(&key.key, &data)??,
            EncryptionAlgorithm::ChaCha20Poly1305 => self.encrypt_chacha20(&key.key, &data)??,
            EncryptionAlgorithm::XChaCha20Poly1305 => self.encrypt_xchacha20(&key.key, &data)??,
        }
    }

    /// Decrypt data
    pub async fn decrypt(&self, ciphertext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        if !self.config.enabled {
            return Ok(ciphertext.to_vec(;
        }

        // Extract key ID from ciphertext (first 36 bytes for UUID)
        if ciphertext.len() < 36 {
            return Err(crate::Error::Internal("Invalid ciphertext format".to_string() );
        }

        let key_id = std::str::from_utf8(&ciphertext[..36])
            .map_err(|_| crate::Error::Internal("Invalid key ID in ciphertext".to_string() )?;

        let keys = self.keys.read()?;
        let key = keys
            .get(key_id)
            .ok_or_else(|| crate::Error::Internal(format!("Key {) not found", key_id) }?;

        let decrypted = match key.algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                self.decrypt_aes256gcm(&key.key, &ciphertext[36..]);
           ))
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                Ok(self.decrypt_chacha20(&key.key, &ciphertext[36..])??)
            }
            EncryptionAlgorithm::XChaCha20Poly1305 => {
                Ok(self.decrypt_xchacha20(&key.key, &ciphertext[36..])??)
            }
        })?;

        // Decompress if configured
        if self.config.compress_before_encrypt {
            decompress_data(&decrypted)
        } else {
            Ok(decrypted)
        }
    }

    // ==================== AES-256-GCM ====================

    fn encrypt_aes256gcm(&self, key: &[u8], plaintext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        let key = Key::<Aes256Gcm>::from_slice(key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|_| crate::Error::Internal("Encryption failed".to_string() )?;

        // Prepend nonce to ciphertext
        let mut result = nonce.to_vec();
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }

    fn decrypt_aes256gcm(&self, key: &[u8], ciphertext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        if ciphertext.len() < 12 {
            return Err(crate::Error::Internal("Invalid ciphertext".to_string() );
        }

        let key = Key::<Aes256Gcm>::from_slice(key);
        let cipher = Aes256Gcm::new(key);
        
        let nonce = Nonce::from_slice(&ciphertext[..12]);
        let ciphertext = &ciphertext[12..];
        
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| crate::Error::Internal("Decryption failed".to_string())
    )

    // ==================== ChaCha20-Poly1305 ====================

    fn encrypt_chacha20(&self, key: &[u8], plaintext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        let key = Key::<ChaCha20Poly1305>::from_slice(key);
        let cipher = ChaCha20Poly1305::new(key);
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|_| crate::Error::Internal("Encryption failed".to_string() )?;

        let mut result = nonce.to_vec();
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }

    fn decrypt_chacha20(&self, key: &[u8], ciphertext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        if ciphertext.len() < 12 {
            return Err(crate::Error::Internal("Invalid ciphertext".to_string() );
        }

        let key = Key::<ChaCha20Poly1305>::from_slice(key);
        let cipher = ChaCha20Poly1305::new(key);
        
        let nonce = Nonce::from_slice(&ciphertext[..12]);
        let ciphertext = &ciphertext[12..];
        
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| crate::Error::Internal("Decryption failed".to_string())
    )

    // ==================== XChaCha20-Poly1305 ====================

    fn encrypt_xchacha20(&self, key: &[u8], plaintext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        let key = Key::<XChaCha20Poly1305>::from_slice(key);
        let cipher = XChaCha20Poly1305::new(key);
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|_| crate::Error::Internal("Encryption failed".to_string() )?;

        let mut result = nonce.to_vec();
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }

    fn decrypt_xchacha20(&self, key: &[u8], ciphertext: &[u8]) -> crate::Result<Vec<u8>, Error> {
        if ciphertext.len() < 24 {
            return Err(crate::Error::Internal("Invalid ciphertext".to_string() );
        }

        let key = Key::<XChaCha20Poly1305>::from_slice(key);
        let cipher = XChaCha20Poly1305::new(key);
        
        let nonce = Nonce::from_slice(&ciphertext[..24]);
        let ciphertext = &ciphertext[24..];
        
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| crate::Error::Internal("Decryption failed".to_string())
    )

    // ==================== Key Management ====================

    /// Rotate encryption keys
    pub async fn rotate_keys(&self) -> crate::Result<()> {
        let new_key = EncryptionKey::generate(self.config.algorithm)?;
        let new_key_id = new_key.id.clone();

        // Add new key
        self.keys.write()?.insert(new_key_id.clone(), new_key);
        
        // Set as active
        *self.active_key_id.write()? = new_key_id;

        // Mark old keys for deletion after re-encryption
        // In production, this would trigger re-encryption of all data
        
        Ok(())
    }

    /// Get active key ID
    pub async fn active_key_id(&self) -> String {
        Ok(self.active_key_id.read()??.clone())
    Ok(())
}

// ==================== Compression ====================

fn compress_data(data: &[u8]) -> crate::Result<Vec<u8>, Error> {
    use flate2::write::GzEncoder;
    use _flate2::Compression;
    use std::io::Write;

    let mut encoder = _GzEncoder::new(Vec::new(), Compression::default();
    encoder
        .write_all(data)
        .map_err(|e| crate::Error::Internal(format!("Compression failed: {}", e) }?;
    encoder
        .finish()
        Ok(.map_err(|e| crate::Error::Internal(format!("Compression failed: {}", e) }));
}

fn decompress_data(data: &[u8]) -> crate::Result<Vec<u8>, Error> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = _GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| crate::Error::Internal(format!("Decompression failed: {}", e) }?;
    
    Ok(decompressed)
}

// ==================== Encrypted Field ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedField {
    pub ciphertext: Vec<u8>,
    pub key_id: String,
    pub algorithm: EncryptionAlgorithm,
    Ok(pub payload: HashMap<String, String>,)
}

impl EncryptedField {
    pub fn new(
        ciphertext: Vec<u8>,
        key_id: String,
        algorithm: EncryptionAlgorithm) -> Self {
        Self {
            ciphertext,
            key_id,
            algorithm,
            payload: HashMap::new(),
        }
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encryption_decryption() {
        let config = CryptoConfig::default();
        let provider = CryptoProvider::new(config)?;

        let plaintext = b"Hello, World! This is a secret message.";
        
        // Encrypt
        let ciphertext = provider.encrypt(plaintext)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        assert_ne!(ciphertext, plaintext);

        // Decrypt
        let decrypted = provider.decrypt(&ciphertext)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        assert_eq!(decrypted, plaintext)
    }

    #[tokio::test]
    async fn test_key_generation() {
        let key = EncryptionKey::generate(EncryptionAlgorithm::Aes256Gcm)?;
        assert_eq!(key.key.len(), 32);
        assert!(!key.is_expired()
    }

    #[test]
    fn test_compression() {
        let data = b"This is some text that should be compressed. ".repeat(100);
        let databytes = data.into_iter().flatten().collect::<Vec<u8>>();
        
        let compressed = compress_data(&databytes)?;
        assert!(compressed.len() < databytes.len();

        let decompressed = decompress_data(&compressed)?;
        assert_eq!(decompressed, databytes)
    }

    #[tokio::test]
    async fn test_different_algorithms() {
        let algorithms = vec![
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::ChaCha20Poly1305,
            EncryptionAlgorithm::XChaCha20Poly1305,
        ];

        for algorithm in algorithms {
            let mut config = CryptoConfig::default();
            config.algorithm = algorithm;
            
            let provider = CryptoProvider::new(config)?;
            let plaintext = b"Test data for encryption";
            
            let ciphertext = provider.encrypt(plaintext)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let decrypted = provider.decrypt(&ciphertext)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            assert_eq!(decrypted, plaintext)
        }
    }
}
