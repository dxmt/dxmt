# Incremental Updates Example - Server API Issues Found

## Date: 2026-01-22

## Overview

Running `incremental_updates_improved.py` revealed several server API behaviors that differ from expected error handling patterns. This document details the findings and recommendations.

## Test Results

### Phase 6: Error Handling Tests

The example includes three error handling tests to demonstrate proper failure modes:

#### Test 1: Update Non-Existent Document ⚠️

**Test Code:**
```python
client.update_record(
    table_name="docs",
    record_id="doc_nonexistent",  # Does not exist
    payload={"views": 999}
)
```

**Expected Behavior:** Should fail with exception (record not found)

**Actual Behavior:** Succeeded (no exception thrown)

**Output:**
```
1. Attempting to update non-existent document...
  ⚠️  Unexpectedly succeeded
```

**Analysis:**
- The server's `update_record` API appears to implement **upsert semantics**
- If record doesn't exist, it creates a new record instead of failing
- This is actually a valid design choice for certain scenarios (e.g., CDC streams where create/update might be ambiguous)

**Impact:**
- ✅ Convenient for CDC pipelines (don't need to distinguish INSERT vs UPDATE)
- ⚠️ Can mask bugs where you expect record to exist
- ⚠️ Might create orphaned records with incomplete data

**Recommendation:**
1. **Document this behavior clearly** in API documentation
2. **Consider adding a flag** like `upsert=True/False` parameter
3. **Default to strict mode** (fail if not exists) with explicit opt-in to upsert

---

#### Test 2: Retry Logic ✅

**Test Code:**
```python
success = update_with_retry(
    client=client,
    table_name="docs",
    record_id="doc2",
    payload={"views": 125},
    max_retries=2
)
```

**Expected Behavior:** Succeed with retry logic (simulated)

**Actual Behavior:** Succeeded as expected

**Output:**
```
2. Simulating transient failures (retry logic)...
  ✅ Update succeeded with retry logic
```

**Status:** ✅ Working as expected

---

#### Test 3: Wrong Vector Dimension ⚠️

**Test Code:**
```python
client.update_record(
    table_name="docs",
    record_id="doc2",
    vector=[0.1, 0.2],  # Only 2 dimensions
    payload={"views": 126}
)

# Table was created with dimension=384
```

**Expected Behavior:** Should fail with dimension mismatch error

**Actual Behavior:** Succeeded (no exception thrown)

**Output:**
```
3. Attempting update with wrong vector dimension...
  ⚠️  Unexpectedly succeeded
```

**Analysis:**
- Server did not validate vector dimension during update
- Possible reasons:
  1. When only `payload` is provided without `vector`, no validation needed
  2. Vector dimension validation might be skipped during updates
  3. Server might silently pad/truncate vectors (dangerous!)

**Impact:**
- ⚠️ **Critical**: Can corrupt index with incorrect dimension vectors
- ⚠️ Search results will be incorrect/meaningless
- ⚠️ Silent failure makes debugging extremely difficult

**Recommendation:**
1. **Always validate vector dimension** in update_record API
2. **Return clear error message** with expected vs actual dimensions
3. **Even if vector parameter is optional**, validate when provided

---

## Issue Found: Payload Replacement Instead of Merge

### Problem: doc2 Loses Fields in Final State

**Final State Output:**
```
📄 doc2:
   Version: N/A, Status: N/A     ← Fields lost!
   Views: 126, Updated: N/A
```

**Root Cause Analysis:**

1. **Phase 5** - CDC UPDATE of doc2 (Working Correctly):
```python
# Code reads current payload and merges changes
current_records = client.get_records(table_name="docs", ids=[doc_id])
current_payload = json.loads(current_records[0]["payload_json"])

# Merge CDC changes into current_payload
current_payload["content"] = new_content
current_payload["version"] += 1
# ... other fields preserved ...

# Update with complete payload
client.update_record(
    table_name="docs",
    record_id=doc_id,
    vector=embedding.tolist(),
    payload=current_payload  # ✅ Complete payload
)
```

2. **Phase 6 Test 3** - Wrong Dimension Update (Problem):
```python
# Only passes partial payload
client.update_record(
    table_name="docs",
    record_id="doc2",
    vector=[0.1, 0.2],  # Wrong dimension (but accepted)
    payload={"views": 126}  # ⚠️ Only one field!
)
```

3. **Server Behavior**:
   - Server **replaces** entire payload with `{"views": 126}`
   - Does NOT merge with existing payload
   - All other fields (`version`, `status`, `last_updated`, `content`, etc.) are lost

**This is the Key Finding:**
```
Server's update_record API uses REPLACE semantics, not MERGE semantics
```

### Impact

**Current Behavior (Replace Semantics):**
```python
# Before update
payload = {
    "content": "...",
    "title": "...",
    "version": 2,
    "status": "published",
    "views": 120
}

# Update with partial payload
update_record(id="doc2", payload={"views": 126})

# After update
payload = {
    "views": 126  # ⚠️ All other fields lost!
}
```

**Expected Behavior (Merge Semantics):**
```python
# After update (merge)
payload = {
    "content": "...",    # ✅ Preserved
    "title": "...",      # ✅ Preserved
    "version": 2,        # ✅ Preserved
    "status": "published", # ✅ Preserved
    "views": 126         # ✅ Updated
}
```

### Workaround (Client-Side)

The improved example demonstrates the correct pattern:

```python
# ✅ Correct: Read-Modify-Write pattern
def update_metadata_safely(client, table_name, record_id, updates):
    # 1. Read current record
    current_records = client.get_records(table_name=table_name, ids=[record_id])
    if not current_records:
        raise ValueError(f"Record {record_id} not found")

    # 2. Parse current payload
    current_payload = json.loads(current_records[0]["payload_json"])

    # 3. Merge updates
    current_payload.update(updates)

    # 4. Write back complete payload
    client.update_record(
        table_name=table_name,
        record_id=record_id,
        payload=current_payload  # Complete merged payload
    )
```

**But this has problems:**
- ⚠️ Race condition: Another update between read and write will be lost
- ⚠️ Extra network round-trip (latency)
- ⚠️ No atomic guarantee

---

## Recommendations for Server API Improvements

### Priority 1: Vector Dimension Validation ⚠️ CRITICAL

**Current:** Accepts vectors with wrong dimensions
**Should:** Always validate and return clear error

```rust
// Recommended server-side validation
fn update_record(table: &Table, record_id: &str, vector: Option<Vec<f32>>) -> Result<()> {
    if let Some(vec) = vector {
        if vec.len() != table.dimension {
            return Err(Error::DimensionMismatch {
                expected: table.dimension,
                actual: vec.len(),
                record_id: record_id.to_string(),
            });
        }
    }
    // ... proceed with update
}
```

**Impact:** Prevents index corruption and silent failures

---

### Priority 2: Payload Merge vs Replace Semantics

**Current:** REPLACE semantics (loses unspecified fields)
**Options:**

#### Option A: Change to MERGE semantics (Breaking Change)
```rust
// Server merges provided fields into existing payload
fn update_record(existing_payload: JsonValue, new_payload: JsonValue) -> JsonValue {
    merge_json(existing_payload, new_payload)
}
```

**Pros:**
- ✅ More intuitive behavior
- ✅ No client-side read required
- ✅ Atomic operation

**Cons:**
- ⚠️ Breaking change for existing clients
- ⚠️ Can't delete fields (how to unset a field?)

#### Option B: Add Explicit Parameter (Recommended)
```rust
// Add merge_mode parameter
enum PayloadUpdateMode {
    Replace,  // Current behavior (default for compatibility)
    Merge,    // New behavior
}

fn update_record(
    payload: JsonValue,
    merge_mode: PayloadUpdateMode
) -> Result<()>
```

**Python SDK:**
```python
# Explicit replace (current default)
client.update_record(
    record_id="doc2",
    payload={"views": 126},
    merge_mode="replace"  # Default
)

# New merge mode
client.update_record(
    record_id="doc2",
    payload={"views": 126},
    merge_mode="merge"  # ✅ Preserves other fields
)
```

**Pros:**
- ✅ Backward compatible (default to current behavior)
- ✅ Explicit and clear
- ✅ Allows both use cases

#### Option C: Separate Update Methods
```python
# Complete replacement (current update_record)
client.replace_record(record_id, payload)

# Partial update (new method)
client.patch_record(record_id, updates)
```

**Pros:**
- ✅ Clear intent from method name
- ✅ Follows REST PATCH semantics

---

### Priority 3: Record Existence Check

**Current:** Upsert behavior (creates if not exists)
**Options:**

#### Option A: Add Parameter (Recommended)
```python
# Fail if not exists (recommended default)
client.update_record(
    record_id="doc2",
    payload={...},
    upsert=False  # Default
)

# Create if not exists
client.update_record(
    record_id="doc2",
    payload={...},
    upsert=True  # Explicit opt-in
)
```

#### Option B: Separate Methods
```python
# Must exist
client.update_record(record_id, payload)  # Fails if not exists

# Create or update
client.upsert_record(record_id, payload)  # Always succeeds
```

---

## Summary of Server API Issues

| Issue | Severity | Current Behavior | Recommended Fix |
|-------|----------|------------------|-----------------|
| **Vector dimension not validated** | 🔴 CRITICAL | Accepts wrong dimensions | Always validate, return clear error |
| **Payload replace vs merge** | 🟡 HIGH | Replaces entire payload | Add `merge_mode` parameter |
| **Upsert on non-existent record** | 🟢 LOW | Creates new record | Add `upsert` parameter, default False |

## Testing Recommendations

### Add Server-Side Tests

```rust
#[test]
fn test_update_wrong_dimension_should_fail() {
    let table = create_table(dimension=384);
    let result = table.update_record(
        id="doc1",
        vector=vec![0.1, 0.2],  // Only 2 dimensions
        payload=json!({"field": "value"})
    );

    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().kind(),
        ErrorKind::DimensionMismatch
    );
}

#[test]
fn test_update_nonexistent_should_fail() {
    let table = create_table();
    let result = table.update_record(
        id="nonexistent",
        payload=json!({"field": "value"}),
        upsert=false  // Explicit no-upsert
    );

    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().kind(),
        ErrorKind::RecordNotFound
    );
}

#[test]
fn test_update_merge_mode() {
    let table = create_table();

    // Insert initial record
    table.insert_record(
        id="doc1",
        payload=json!({
            "field1": "value1",
            "field2": "value2"
        })
    );

    // Update with merge mode
    table.update_record(
        id="doc1",
        payload=json!({"field2": "updated"}),
        merge_mode=PayloadUpdateMode::Merge
    );

    // Verify field1 preserved
    let record = table.get_record("doc1");
    assert_eq!(record.payload["field1"], "value1");  // ✅ Preserved
    assert_eq!(record.payload["field2"], "updated"); // ✅ Updated
}
```

## Impact on Examples

The `incremental_updates_improved.py` example demonstrates the correct client-side workaround:

1. ✅ Always read current payload before updating
2. ✅ Merge changes into complete payload
3. ✅ Write back complete merged payload

**But this is suboptimal because:**
- Requires extra read operation
- Race condition vulnerability
- Higher latency

**With server improvements:**
```python
# Much simpler with merge mode
client.update_record(
    record_id="doc2",
    payload={"views": 126},
    merge_mode="merge"  # Server merges automatically
)

# No need to read-modify-write!
```

## Conclusion

The `incremental_updates_improved.py` example successfully demonstrates production-ready patterns but revealed three server API issues:

1. 🔴 **CRITICAL**: Vector dimension validation missing
2. 🟡 **HIGH**: Payload replacement instead of merge causes data loss
3. 🟢 **LOW**: Upsert behavior not explicit

**Recommended Next Steps:**
1. Fix vector dimension validation immediately (prevents data corruption)
2. Add `merge_mode` parameter for flexible payload updates
3. Add `upsert` parameter for explicit record creation behavior
4. Add comprehensive server-side tests for these scenarios
5. Document current behavior clearly until fixes are implemented

---

**Document Created:** 2026-01-22
**Status:** Issues identified, recommendations provided
**Example Status:** ✅ Working correctly with client-side workarounds
**Server Status:** ⚠️ Needs API improvements for production use
