use aes_gcm::aead::rand_core::RngCore;
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use base64::{engine::general_purpose, Engine as _};

/// Encryption manager for data at rest
pub struct EncryptionManager {
    cipher: Aes256Gcm,
    #[allow(dead_code)]
    key: Vec<u8>,
}

impl EncryptionManager {
    pub fn new(key: &[u8]) -> crate::Result<Self> {
        if key.len() != 32 {
            return Err(crate::Error::Storage(
                "Encryption key must be 32 bytes".to_string(),
            ));
        }

        let key_array: [u8; 32] = key
            .try_into()
            .map_err(|_| crate::Error::Storage("Failed to convert key".to_string()))?;
        let cipher = Aes256Gcm::new(&Key::<Aes256Gcm>::from_slice(&key_array).clone());

        Ok(Self {
            cipher,
            key: key.to_vec(),
        })
    }

    pub fn generate_key() -> Vec<u8> {
        let mut key = vec![0u8; 32];
        OsRng.fill_bytes(&mut key);
        key
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> crate::Result<Vec<u8>> {
        let mut noncebytes = [0u8; 12];
        OsRng.fill_bytes(&mut noncebytes);
        let nonce = Nonce::from_slice(&noncebytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| crate::Error::Storage(format!("Encryption failed: {}", e)))?;

        // Prepend nonce to ciphertext
        let mut result = noncebytes.to_vec();
        result.extend(ciphertext);

        Ok(result)
    }

    pub fn decrypt(&self, ciphertext: &[u8]) -> crate::Result<Vec<u8>> {
        if ciphertext.len() < 12 {
            return Err(crate::Error::Storage("Ciphertext too short".to_string()));
        }

        let (noncebytes, encrypted) = ciphertext.split_at(12);
        let nonce = Nonce::from_slice(noncebytes);

        let plaintext = self
            .cipher
            .decrypt(nonce, encrypted)
            .map_err(|e| crate::Error::Storage(format!("Decryption failed: {}", e)))?;

        Ok(plaintext)
    }

    pub fn encrypt_string(&self, plaintext: &str) -> crate::Result<String> {
        let encrypted = self.encrypt(plaintext.as_bytes())?;
        Ok(general_purpose::STANDARD.encode(encrypted))
    }

    pub fn decrypt_string(&self, ciphertext: &str) -> crate::Result<String> {
        let decoded = general_purpose::STANDARD
            .decode(ciphertext)
            .map_err(|e| crate::Error::Storage(format!("Base64 decode failed: {}", e)))?;
        let decrypted = self.decrypt(&decoded)?;
        String::from_utf8(decrypted)
            .map_err(|e| crate::Error::Storage(format!("UTF-8 decode failed: {}", e)))
    }
}

/// Key management for encryption
pub struct KeyManager {
    keys: std::collections::HashMap<String, Vec<u8>>,
    active_key_id: String,
}

impl Default for KeyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyManager {
    pub fn new() -> Self {
        Self {
            keys: std::collections::HashMap::new(),
            active_key_id: String::new(),
        }
    }

    pub fn add_key(&mut self, key_id: String, key: Vec<u8>) -> crate::Result<()> {
        if key.len() != 32 {
            return Err(crate::Error::Storage("Key must be 32 bytes".to_string()));
        }
        self.keys.insert(key_id, key);
        Ok(())
    }

    pub fn set_active_key(&mut self, key_id: String) -> crate::Result<()> {
        if !self.keys.contains_key(&key_id) {
            return Err(crate::Error::Storage(format!("Key {} not found", key_id)));
        }
        self.active_key_id = key_id;
        Ok(())
    }

    pub fn get_active_key(&self) -> crate::Result<&Vec<u8>> {
        self.keys
            .get(&self.active_key_id)
            .ok_or_else(|| crate::Error::Storage("No active key set".to_string()))
    }

    pub fn rotate_key(&mut self) -> crate::Result<String> {
        let new_key = EncryptionManager::generate_key();
        let new_key_id = format!("key_{}", chrono::Utc::now().timestamp());
        self.add_key(new_key_id.clone(), new_key)?;
        self.set_active_key(new_key_id.clone())?;
        Ok(new_key_id)
    }
}

/// Envelope encryption for large data
pub struct EnvelopeEncryption {
    key_encryption_key: Vec<u8>,
}

impl EnvelopeEncryption {
    pub fn new(masterkey: &[u8]) -> crate::Result<Self> {
        if masterkey.len() != 32 {
            return Err(crate::Error::Storage(
                "Master key must be 32 bytes".to_string(),
            ));
        }
        Ok(Self {
            key_encryption_key: masterkey.to_vec(),
        })
    }

    pub fn encrypt_data(&self, plaintext: &[u8]) -> crate::Result<EncryptedData> {
        // Generate data encryption key
        let data_key = EncryptionManager::generate_key();

        // Encrypt data with data key
        let data_encryptor = EncryptionManager::new(&data_key)?;
        let encrypted_data = data_encryptor.encrypt(plaintext)?;

        // Encrypt data key with key encryption key
        let key_encryptor = EncryptionManager::new(&self.key_encryption_key)?;
        let encrypted_key = key_encryptor.encrypt(&data_key)?;

        Ok(EncryptedData {
            encrypted_key,
            encrypted_data,
        })
    }

    pub fn decrypt_data(&self, encrypted: &EncryptedData) -> crate::Result<Vec<u8>> {
        // Decrypt data key
        let key_decryptor = EncryptionManager::new(&self.key_encryption_key)?;
        let data_key = key_decryptor.decrypt(&encrypted.encrypted_key)?;

        // Decrypt data with decrypted data key
        let data_decryptor = EncryptionManager::new(&data_key)?;
        data_decryptor.decrypt(&encrypted.encrypted_data)
    }
}

#[derive(Clone, Debug)]
pub struct EncryptedData {
    pub encrypted_key: Vec<u8>,
    pub encrypted_data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_manager() -> Result<(), Box<dyn std::error::Error>> {
        let key = EncryptionManager::generate_key();
        let manager = EncryptionManager::new(&key)?;

        let plaintext = b"Hello, world!";
        let encrypted = manager.encrypt(plaintext)?;
        let decrypted = manager.decrypt(&encrypted)?;

        assert_eq!(plaintext, &decrypted[..]);
        Ok(())
    }

    #[test]
    fn test_string_encryption() -> Result<(), Box<dyn std::error::Error>> {
        let key = EncryptionManager::generate_key();
        let manager = EncryptionManager::new(&key)?;

        let plaintext = "Hello, world!";
        let encrypted = manager.encrypt_string(plaintext)?;
        let decrypted = manager.decrypt_string(&encrypted)?;

        assert_eq!(plaintext, decrypted);
        Ok(())
    }

    #[test]
    fn test_envelope_encryption() -> Result<(), Box<dyn std::error::Error>> {
        let master_key = EncryptionManager::generate_key();
        let envelope = EnvelopeEncryption::new(&master_key)?;

        let plaintext = b"Large data payload";
        let encrypted = envelope.encrypt_data(plaintext)?;
        let decrypted = envelope.decrypt_data(&encrypted)?;

        assert_eq!(plaintext, &decrypted[..]);
        Ok(())
    }
}
