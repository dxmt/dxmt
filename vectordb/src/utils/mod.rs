
pub mod cpu_affinity;
pub mod cpu_detection;
pub mod memory_tracking;
pub mod safety_checks;

/// 验证表名是否有效
pub fn validate_table_name(name: &str) -> crate::Result<()> {
    if name.is_empty() {
        return Err(crate::Error::InvalidInput("Table name cannot be empty".to_string()));
    }

    if name.len() > 63 {
        return Err(crate::Error::InvalidInput(
            "Table name too long (max 63 characters)".to_string()
        ));
    }

    // 表名必须以字母开头，只能包含字母、数字和下划线
    let first_char = name.chars().next().ok_or_else(|| crate::Error::InvalidInput("Empty table name".to_string()))?;
    if !first_char.is_ascii_alphabetic() {
        return Err(crate::Error::InvalidInput(
            "Table name must start with a letter".to_string()
        ));
    }

    for ch in name.chars() {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            return Err(crate::Error::InvalidInput(
                "Table name can only contain letters, numbers, and underscores".to_string()
            ));
        }
    }

    Ok(())
}

/// 验证记录ID是否有效
pub fn validate_record_id(id: &str) -> crate::Result<()> {
    if id.is_empty() {
        return Err(crate::Error::InvalidInput("Record ID cannot be empty".to_string()));
    }

    if id.len() > 255 {
        return Err(crate::Error::InvalidInput(
            "Record ID too long (max 255 characters)".to_string()
        ));
    }

    // 记录ID不能包含控制字符
    for ch in id.chars() {
        if ch.is_control() {
            return Err(crate::Error::InvalidInput(
                "Record ID cannot contain control characters".to_string()
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_table_name() {
        // 有效的表名
        assert!(validate_table_name("test_table").is_ok());
        assert!(validate_table_name("table123").is_ok());
        assert!(validate_table_name("my_table_name").is_ok());

        // 无效的表名
        assert!(validate_table_name("").is_err()); // 空名称
        assert!(validate_table_name("123table").is_err()); // 数字开头
        assert!(validate_table_name("test-table").is_err()); // 包含连字符
        assert!(validate_table_name("test.table").is_err()); // 包含点号
        assert!(validate_table_name(&"a".repeat(64)).is_err()); // 太长
    }

    #[test]
    fn test_validate_record_id() {
        // 有效的记录ID
        assert!(validate_record_id("test_id").is_ok());
        assert!(validate_record_id("123").is_ok());
        assert!(validate_record_id("test-id.value").is_ok());

        // 无效的记录ID
        assert!(validate_record_id("").is_err()); // 空ID
        assert!(validate_record_id(&"a".repeat(256)).is_err()); // 太长
        assert!(validate_record_id("test\nid").is_err()); // 包含控制字符
        assert!(validate_record_id("test\tid").is_err()); // 包含制表符
    }
}
