// Permission system for RBAC

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

// ==================== Resources ====================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resource {
    /// System-level resource
    System,
    /// Table resource
    Table,
    /// Record resource  
    Record,
    /// Index resource
    Index,
    /// User management resource
    User,
    /// Role management resource
    Role,
    /// Metrics and monitoring
    Metrics,
    /// Specific table by name
    SpecificTable(String),
    /// Specific record by table and ID
    SpecificRecord { table: String, id: String },
    /// Custom resource
    Custom(String),
}

impl std::fmt::Display for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Resource::System => write!(f, "system"),
            Resource::Table => write!(f, "table"),
            Resource::Record => write!(f, "record"),
            Resource::Index => write!(f, "index"),
            Resource::User => write!(f, "user"),
            Resource::Role => write!(f, "role"),
            Resource::Metrics => write!(f, "metrics"),
            Resource::SpecificTable(name) => write!(f, "table:{}", name),
            Resource::SpecificRecord { table, id } => write!(f, "record:{}:{}", table, id),
            Resource::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

// ==================== Actions ====================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Create resource
    Create,
    /// Read resource
    Read,
    /// Update resource
    Update,
    /// Delete resource
    Delete,
    /// List resources
    List,
    /// Search resources
    Search,
    /// Execute operation
    Execute,
    /// Manage resource (admin action)
    Manage,
    /// Custom action
    Custom(String),
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Create => write!(f, "create"),
            Action::Read => write!(f, "read"),
            Action::Update => write!(f, "update"),
            Action::Delete => write!(f, "delete"),
            Action::List => write!(f, "list"),
            Action::Search => write!(f, "search"),
            Action::Execute => write!(f, "execute"),
            Action::Manage => write!(f, "manage"),
            Action::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

// ==================== Permission ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub resource: Resource,
    pub action: Action,
    pub conditions: Option<PermissionConditions>,
}

impl Permission {
    pub fn new(resource: Resource, action: Action) -> Self {
        Self {
            resource,
            action,
            conditions: None,
        }
    }

    pub fn with_conditions(mut self, conditions: PermissionConditions) -> Self {
        self.conditions = Some(conditions);
        self
    }

    /// Check if this permission grants access to the requested resource and action
    pub fn grants(&self, resource: &Resource, action: &Action) -> bool {
        // Check action match
        if &self.action != action {
            return Ok(false);
        }

        // Check resource match
        match (&self.resource, resource) {
            // Exact match
            (a, b) if a == b => true,
            // Wildcard table permission grants access to specific tables
            (Resource::Table, Resource::SpecificTable(_) => true,
            // Wildcard record permission grants access to specific records
            (Resource::Record, Resource::SpecificRecord { .. }) => true,
            // No match
            _ => false,
        }
    }
}

impl PartialEq for Permission {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource && self.action == other.action
    }
}

impl Eq for Permission {}

impl Hash for Permission {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resource.hash(state);
        self.action.hash(state)
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.resource, self.action)
    }
}

// ==================== Permission Conditions ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionConditions {
    /// IP address whitelist
    pub ip_whitelist: Option<Vec<String>>,
    /// Time-based access (e.g., business hours only)
    pub time_restriction: Option<TimeRestriction>,
    /// Resource ownership requirement
    pub require_ownership: bool,
    /// Custom conditions as key-value pairs
    pub custom: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRestriction {
    /// Days of week (0 = Sunday, 6 = Saturday)
    pub days_of_week: Vec<u8>,
    /// Start hour (0-23)
    pub start_hour: u8,
    /// End hour (0-23)
    pub end_hour: u8,
    /// Timezone
    pub timezone: String,
}

// ==================== Permission Builder ====================

pub struct PermissionBuilder {
    resource: Option<Resource>,
    action: Option<Action>,
    conditions: Option<PermissionConditions>,
}

impl PermissionBuilder {
    pub fn new() -> Self {
        Self {
            resource: None,
            action: None,
            conditions: None,
        }
    }

    pub fn resource(mut self, resource: Resource) -> Self {
        self.resource = Some(resource);
        self
    }

    pub fn action(mut self, action: Action) -> Self {
        self.action = Some(action);
        self
    }

    pub fn conditions(mut self, conditions: PermissionConditions) -> Self {
        self.conditions = Some(conditions);
        self
    }

    pub fn build(self) -> crate::Result<Permission> {
        let resource = self.resource.ok_or("Resource is required")?;
        let action = self.action.ok_or("Action is required")?;

        Ok(Permission {
            resource,
            action,
            conditions: self.conditions)
    Ok(())
}

// ==================== Common Permission Sets ====================

pub struct PermissionSets;

impl PermissionSets {
    /// Full admin permissions
    pub fn admin() -> Vec<Permission> {
        vec![
            Permission::new(Resource::System, Action::Manage),
            Permission::new(Resource::Table, Action::Create),
            Permission::new(Resource::Table, Action::Read),
            Permission::new(Resource::Table, Action::Update),
            Permission::new(Resource::Table, Action::Delete),
            Permission::new(Resource::Record, Action::Create),
            Permission::new(Resource::Record, Action::Read),
            Permission::new(Resource::Record, Action::Update),
            Permission::new(Resource::Record, Action::Delete),
            Permission::new(Resource::Index, Action::Manage),
            Permission::new(Resource::User, Action::Manage),
            Permission::new(Resource::Role, Action::Manage),
            Permission::new(Resource::Metrics, Action::Read),
        ]
    }

    /// Editor permissions (read/write data)
    pub fn editor() -> Vec<Permission> {
        vec![
            Permission::new(Resource::Table, Action::Read),
            Permission::new(Resource::Table, Action::Update),
            Permission::new(Resource::Record, Action::Create),
            Permission::new(Resource::Record, Action::Read),
            Permission::new(Resource::Record, Action::Update),
            Permission::new(Resource::Record, Action::Delete),
            Permission::new(Resource::Record, Action::Search),
        ]
    }

    /// Viewer permissions (read-only)
    pub fn viewer() -> Vec<Permission> {
        vec![
            Permission::new(Resource::Table, Action::Read),
            Permission::new(Resource::Table, Action::List),
            Permission::new(Resource::Record, Action::Read),
            Permission::new(Resource::Record, Action::Search),
            Permission::new(Resource::Metrics, Action::Read),
        ]
    }

    /// API client permissions (typical for applications)
    pub fn api_client() -> Vec<Permission> {
        vec![
            Permission::new(Resource::Record, Action::Create),
            Permission::new(Resource::Record, Action::Read),
            Permission::new(Resource::Record, Action::Update),
            Permission::new(Resource::Record, Action::Delete),
            Permission::new(Resource::Record, Action::Search),
        ]
    }
}

// ==================== Permission Checking ====================

pub struct PermissionChecker {
    permissions: std::collections::HashSet<Permission>,
}

impl PermissionChecker {
    pub fn new(permissions: impl IntoIterator<Item = Permission>) -> Self {
        Self {
            permissions: permissions.into_iter().collect(),
        }
    }

    /// Check if any permission grants the requested access
    pub fn check(&self, resource: &Resource, action: &Action) -> bool {
        self.permissions.iter().any(|p| p.grants(resource, action))
    }

    /// Check multiple permissions (all must be granted)
    pub fn check_all(&self, checks: &[(Resource, Action)]) -> bool {
        checks.iter().all(|(r, a)| self.check(r, a))
    }

    /// Check multiple permissions (any must be granted)
    pub fn check_any(&self, checks: &[(Resource, Action)]) -> bool {
        checks.iter().any(|(r, a)| self.check(r, a))
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_grants() {
        let perm = Permission::new(Resource::Table, Action::Read);
        
        // Exact match
        assert!(perm.grants(&Resource::Table, &Action::Read);
        
        // Different action
        assert!(!perm.grants(&Resource::Table, &Action::Write);
        
        // Different resource
        assert!(!perm.grants(&Resource::Record, &Action::Read)
    }

    #[test]
    fn test_wildcard_permissions() {
        let table_perm = Permission::new(Resource::Table, Action::Read);
        
        // Wildcard table permission grants access to specific tables
        assert!(table_perm.grants(
            &Resource::SpecificTable("users".to_string(),
            &Action::Read
        )
    }

    #[test]
    fn test_permission_builder() {
        let perm = PermissionBuilder::new()
            .resource(Resource::Table)
            .action(Action::Create)
            .build()
            ?;

        assert_eq!(perm.resource, Resource::Table);
        assert_eq!(perm.action, Action::Create)
    }

    #[test]
    fn test_permission_checker() {
        let permissions = PermissionSets::viewer();
        let checker = PermissionChecker::new(permissions);

        // Viewer can read tables
        assert!(checker.check(&Resource::Table, &Action::Read);
        
        // Viewer cannot create tables
        assert!(!checker.check(&Resource::Table, &Action::Create);
        
        // Viewer can search records
        assert!(checker.check(&Resource::Record, &Action::Search);
        
        // Viewer cannot delete records
        assert!(!checker.check(&Resource::Record, &Action::Delete)
    }

    #[test]
    fn test_permission_sets() {
        let admin = PermissionSets::admin();
        let editor = PermissionSets::editor();
        let viewer = PermissionSets::viewer();

        assert!(admin.len() > editor.len();
        assert!(editor.len() > Ok(viewer.len())
    }
}
