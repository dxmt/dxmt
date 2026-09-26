pub mod cpu_affinity;
pub mod cpu_detection;
pub mod id_hash;
pub mod memory_tracking;
pub mod safety_checks;

/// Whether a table may be called this.
///
/// A table's name becomes a file name — `wal/records/<name>.rec` and its siblings — so the
/// rule has to keep out anything that would escape the data directory or mean something to a
/// filesystem. That is the *whole* justification, and the rule was wider than it:
///
/// * **`-` was refused.** `my-collection` is the most common way anyone names one of these,
///   and a hyphen is an ordinary character in a file name on every filesystem.
/// * **A leading digit or `_` was refused.** `2024_logs` and `_staging` are file names too.
///
/// What stays refused, and why:
///
/// * Anything but `[A-Za-z0-9_-]`, which keeps out `/`, `\` and `.` — the three that make a
///   name a path — and the shell and URL metacharacters with them.
/// * Non-ASCII. A filesystem is entitled to normalise it (macOS stores NFD, Linux stores what
///   it was given), and a table created on one machine then cannot be opened after a copy to
///   another. The name is an identifier here, not a label; put the label in `description`.
/// * A leading `-`, which is a command-line flag everywhere it is later pasted.
pub fn validate_table_name(name: &str) -> crate::Result<()> {
    if name.is_empty() {
        return Err(crate::Error::InvalidInput(
            "Table name cannot be empty".to_string(),
        ));
    }

    if name.len() > 63 {
        return Err(crate::Error::InvalidInput(
            "Table name too long (max 63 characters)".to_string(),
        ));
    }

    if name.starts_with('-') {
        return Err(crate::Error::InvalidInput(
            "Table name cannot start with '-': it is a flag to every command line the name \
             is later pasted into. Letters, digits and '_' may start one."
                .to_string(),
        ));
    }

    for ch in name.chars() {
        if !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-' {
            return Err(crate::Error::InvalidInput(format!(
                "Table name contains '{ch}', which is not allowed. A table's name becomes a \
                 file name, so it may hold letters, digits, '_' and '-' only -- that keeps \
                 out '/', '\\' and '.', and it keeps out non-ASCII, which a filesystem is \
                 entitled to normalise so that the table cannot be opened after a copy to \
                 another machine. Put the human-readable name in 'description'."
            )));
        }
    }

    Ok(())
}

/// 验证记录ID是否有效
pub fn validate_record_id(id: &str) -> crate::Result<()> {
    if id.is_empty() {
        return Err(crate::Error::InvalidInput(
            "Record ID cannot be empty".to_string(),
        ));
    }

    if id.len() > 255 {
        return Err(crate::Error::InvalidInput(
            "Record ID too long (max 255 characters)".to_string(),
        ));
    }

    // 记录ID不能包含控制字符
    for ch in id.chars() {
        if ch.is_control() {
            return Err(crate::Error::InvalidInput(
                "Record ID cannot contain control characters".to_string(),
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

        // A hyphen and a leading digit are ordinary in a file name, which is the only thing
        // this rule exists to protect. `my-collection` is how most people name one of these
        // and used to be refused.
        assert!(validate_table_name("my-collection").is_ok());
        assert!(validate_table_name("2024_logs").is_ok());
        assert!(validate_table_name("_staging").is_ok());

        // 无效的表名
        assert!(validate_table_name("").is_err()); // 空名称
        assert!(validate_table_name("test.table").is_err()); // 点号，会让名字变成路径
        assert!(validate_table_name("a/b").is_err()); // 路径分隔符
        assert!(validate_table_name("-rf").is_err()); // 命令行 flag
        assert!(validate_table_name("表名").is_err()); // 非 ASCII，文件系统可能做归一化
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
