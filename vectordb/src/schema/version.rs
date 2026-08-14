use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub version: u32,
    pub created_at: DateTime<Utc>,
    pub description: String,
    pub migrations: Vec<Migration>,
    pub compatibility: CompatibilityMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityMode {
    Backward,
    Forward,
    Full,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    pub from_version: u32,
    pub to_version: u32,
    pub operations: Vec<MigrationOperation>,
    pub rollback_operations: Vec<MigrationOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationOperation {
    AddField {
        table: String,
        field: FieldDefinition,
    },
    RemoveField {
        table: String,
        field_name: String,
    },
    RenameField {
        table: String,
        old_name: String,
        new_name: String,
    },
    ChangeFieldType {
        table: String,
        field_name: String,
        new_type: FieldType,
        conversion: TypeConversion,
    },
    AddTable {
        definition: TableDefinition,
    },
    RemoveTable {
        name: String,
    },
    RenameTable {
        old_name: String,
        new_name: String,
    },
    AddIndex {
        table: String,
        index: IndexDefinition,
    },
    RemoveIndex {
        table: String,
        index_name: String,
    },
    Custom {
        function: String,
        parameters: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: FieldType,
    pub nullable: bool,
    pub default_value: Option<serde_json::Value>,
    pub indexed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Vector(usize),
    Text,
    Integer,
    Float,
    Boolean,
    Json,
    Timestamp,
    Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeConversion {
    Cast,
    Custom(String),
    Default(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableDefinition {
    pub name: String,
    pub fields: Vec<FieldDefinition>,
    pub primary_key: Vec<String>,
    pub indices: Vec<IndexDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub fields: Vec<String>,
    pub index_type: IndexType,
    pub unique: bool,
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    Ivf,
    Flat,
    Bm25,
    BTree,
    Hash,
}

pub struct SchemaVersionManager {
    versions: HashMap<u32, SchemaVersion>,
    current_version: u32,
}

impl SchemaVersionManager {
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            current_version: 0,
        }
    }

    pub fn register_version(&mut self, version: SchemaVersion) -> Result<(), SchemaError> {
        if self.versions.contains_key(&version.version) {
            return Err(SchemaError::VersionAlreadyExists(version.version));
        }

        for migration in &version.migrations {
            if migration.from_version >= migration.to_version {
                return Err(SchemaError::InvalidMigration(
                    migration.from_version,
                    migration.to_version,
                ));
            }
        }

        self.versions.insert(version.version, version);
        Ok(())
    }

    pub fn get_version(&self, version: u32) -> Option<&SchemaVersion> {
        self.versions.get(&version)
    }

    pub fn set_current_version(&mut self, version: u32) -> Result<(), SchemaError> {
        if !self.versions.contains_key(&version) {
            return Err(SchemaError::VersionNotFound(version));
        }

        self.current_version = version;
        Ok(())
    }

    pub fn get_current_version(&self) -> u32 {
        self.current_version
    }

    pub fn get_migration_path(&self, from: u32, to: u32) -> Result<Vec<Migration>, SchemaError> {
        if from == to {
            return Ok(vec![]);
        }

        let mut path = vec![];
        let mut current = from;

        if from < to {
            while current < to {
                let next_version = self.find_next_version(current, to)?;
                let version = self.versions.get(&next_version)
                    .ok_or(SchemaError::VersionNotFound(next_version))?;

                if let Some(migration) = version.migrations.iter()
                    .find(|m| m.from_version == current && m.to_version == next_version)
                {
                    path.push(migration.clone());
                    current = next_version;
                } else {
                    return Err(SchemaError::MigrationNotFound(current, next_version));
                }
            }
        } else {
            while current > to {
                let prev_version = self.find_previous_version(current, to)?;
                let version = self.versions.get(&current)
                    .ok_or(SchemaError::VersionNotFound(current))?;

                if let Some(migration) = version.migrations.iter()
                    .find(|m| m.from_version == prev_version && m.to_version == current)
                {
                    let rollback = Migration {
                        from_version: current,
                        to_version: prev_version,
                        operations: migration.rollback_operations.clone(),
                        rollback_operations: migration.operations.clone(),
                    };
                    path.push(rollback);
                    current = prev_version;
                } else {
                    return Err(SchemaError::MigrationNotFound(current, prev_version));
                }
            }
        }

        Ok(path)
    }

    fn find_next_version(&self, current: u32, target: u32) -> Result<u32, SchemaError> {
        let mut candidates: Vec<u32> = self.versions.keys()
            .filter(|&&v| v > current && v <= target)
            .copied()
            .collect();

        candidates.sort();

        candidates.first().copied()
            .ok_or(SchemaError::NoMigrationPath(current, target))
    }

    fn find_previous_version(&self, current: u32, target: u32) -> Result<u32, SchemaError> {
        let mut candidates: Vec<u32> = self.versions.keys()
            .filter(|&&v| v < current && v >= target)
            .copied()
            .collect();

        candidates.sort_by(|a, b| b.cmp(a));

        candidates.first().copied()
            .ok_or(SchemaError::NoMigrationPath(current, target))
    }

    pub fn validate_compatibility(
        &self,
        from_version: u32,
        to_version: u32,
    ) -> Result<bool, SchemaError> {
        let from = self.versions.get(&from_version)
            .ok_or(SchemaError::VersionNotFound(from_version))?;

        let to = self.versions.get(&to_version)
            .ok_or(SchemaError::VersionNotFound(to_version))?;

        match (&from.compatibility, &to.compatibility) {
            (CompatibilityMode::None, _) | (_, CompatibilityMode::None) => Ok(false),
            (CompatibilityMode::Full, _) | (_, CompatibilityMode::Full) => Ok(true),
            (CompatibilityMode::Backward, _) if from_version < to_version => Ok(true),
            (CompatibilityMode::Forward, _) if from_version > to_version => Ok(true),
            _ => Ok(false),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Version {0} already exists")]
    VersionAlreadyExists(u32),

    #[error("Version {0} not found")]
    VersionNotFound(u32),

    #[error("Invalid migration from version {0} to {1}")]
    InvalidMigration(u32, u32),

    #[error("Migration not found from version {0} to {1}")]
    MigrationNotFound(u32, u32),

    #[error("No migration path from version {0} to {1}")]
    NoMigrationPath(u32, u32),

    #[error("Incompatible versions")]
    IncompatibleVersions,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version_manager() {
        let mut manager = SchemaVersionManager::new();

        let v1 = SchemaVersion {
            version: 1,
            created_at: Utc::now(),
            description: "Initial version".to_string(),
            migrations: vec![],
            compatibility: CompatibilityMode::Full,
        };

        let v2 = SchemaVersion {
            version: 2,
            created_at: Utc::now(),
            description: "Add vector field".to_string(),
            migrations: vec![
                Migration {
                    from_version: 1,
                    to_version: 2,
                    operations: vec![
                        MigrationOperation::AddField {
                            table: "documents".to_string(),
                            field: FieldDefinition {
                                name: "embedding".to_string(),
                                field_type: FieldType::Vector(384),
                                nullable: true,
                                default_value: None,
                                indexed: true,
                            },
                        },
                    ],
                    rollback_operations: vec![
                        MigrationOperation::RemoveField {
                            table: "documents".to_string(),
                            field_name: "embedding".to_string(),
                        },
                    ],
                },
            ],
            compatibility: CompatibilityMode::Backward,
        };

        manager.register_version(v1).unwrap();
        manager.register_version(v2).unwrap();

        manager.set_current_version(1).unwrap();

        let path = manager.get_migration_path(1, 2).unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].from_version, 1);
        assert_eq!(path[0].to_version, 2);

        let rollback_path = manager.get_migration_path(2, 1).unwrap();
        assert_eq!(rollback_path.len(), 1);
        assert_eq!(rollback_path[0].from_version, 2);
        assert_eq!(rollback_path[0].to_version, 1);
    }
}