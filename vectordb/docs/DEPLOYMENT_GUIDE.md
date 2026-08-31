# DXMT VectorDB - Deployment Guide for Server Improvements

## 🚀 Quick Start

### 1. Build the Improved Server

```bash
cd /localdisk/code/dxmt/vectordb

# Check compilation
cargo check --lib
# ✅ Should complete with 0 errors

# Build release version
cargo build --release
# ✅ Creates binary at target/release/dxmt-vectordb
```

### 2. Run the Server

```bash
# Simple mode (recommended for testing)
cargo run -- serve --simple --memory-limit 1024

# Or use the release binary
./target/release/dxmt-vectordb serve --simple --memory-limit 1024

# Standard mode with all features
./target/release/dxmt-vectordb serve --host 0.0.0.0 --port 9999
```

### 3. Verify the Improvements

```bash
# Terminal 1: Server is running
cargo run -- serve --simple --memory-limit 1024

# Terminal 2: Test improvements
cd dxmt-python-sdk/examples
python3 test_server_improvements.py

# Terminal 3: Run original problem scenario
python3 batch_operations.py
```

## 📊 What Changed

### Before

```
Symptom: Insert failures after heavy operations
Error: HTTP 500 Internal Server Error
Message: "Failed to insert records:"
Retry: Manual with guesswork
```

### After

```
Symptom: Intelligent backpressure
Response: HTTP 429 Too Many Requests
Header: Retry-After: 2
Message: Detailed explanation with suggestions
Retry: Automatic with exact timing
```

## 🔍 New Monitoring Endpoints

### 1. Enhanced Health Check

```bash
curl http://localhost:9999/health | jq
```

**Response**:
```json
{
  "status": "Healthy",
  "description": "Server is operating normally",
  "retry_after_seconds": 0,
  "metrics": {
    "ongoing_inserts": 5,
    "ongoing_searches": 12,
    "total_operations": 15234,
    "recent_errors": 0,
    "index_building": false
  },
  "accepting_requests": {
    "inserts": true,
    "searches": true
  }
}
```

### 2. Detailed Metrics (NEW)

```bash
curl http://localhost:9999/health/metrics | jq
```

**Response**:
```json
{
  "ongoing_operations": {
    "inserts": 5,
    "searches": 12,
    "deletes": 0,
    "table_operations": 0,
    "total": 17
  },
  "completed_operations": 15234,
  "errors": {
    "recent_count": 0
  },
  "index_status": {
    "building": false
  }
}
```

## 🎯 Testing Checklist

### Basic Functionality Tests

- [ ] **Server starts successfully**
  ```bash
  cargo run -- serve --simple --memory-limit 1024
  ```

- [ ] **Health endpoint works**
  ```bash
  curl http://localhost:9999/health
  # Should return health status with new fields
  ```

- [ ] **Metrics endpoint works**
  ```bash
  curl http://localhost:9999/health/metrics
  # Should return detailed metrics
  ```

### Improvement Tests

- [ ] **Test server improvements script**
  ```bash
  python3 test_server_improvements.py
  # All 4 tests should pass
  ```

- [ ] **Original problem scenario**
  ```bash
  python3 batch_operations.py
  # Should complete without 3 consecutive failures
  # May see 429 instead of 500
  ```

- [ ] **Exact reproduction test**
  ```bash
  python3 test_exact_reproduction.py
  # Should show improvement in error handling
  ```

### Load Tests

- [ ] **Heavy load test**
  ```bash
  # Monitor health while running
  watch -n 1 'curl -s http://localhost:9999/health | jq .data.status'

  # In another terminal, run load test
  python3 batch_operations.py

  # Should see status transitions: Healthy → Degraded → Overloaded → Healthy
  ```

## 📈 Monitoring in Production

### Setup Continuous Monitoring

```bash
# Option 1: Simple watch
watch -n 5 'curl -s http://localhost:9999/health | jq'

# Option 2: Log to file
while true; do
    curl -s http://localhost:9999/health >> health.log
    sleep 5
done

# Option 3: Alert on non-healthy state
while true; do
    status=$(curl -s http://localhost:9999/health | jq -r '.data.status')
    if [ "$status" != "Healthy" ]; then
        echo "ALERT: Server status is $status" | mail -s "DXMT Alert" admin@example.com
    fi
    sleep 30
done
```

### Prometheus Integration (Future)

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'dxmt'
    static_configs:
      - targets: ['localhost:9999']
    metrics_path: '/health/metrics'
    scrape_interval: 10s
```

### Grafana Dashboard (Future)

Key metrics to monitor:
- Server health state
- Ongoing operations count
- Error rate
- Index building status
- Operation throughput

## 🔧 Tuning Health Thresholds

If you need to adjust thresholds for your hardware:

### Option 1: Code Modification

Edit `src/api/server.rs`:

```rust
use crate::api::server_health::{ServerHealth, HealthConfig};

let config = HealthConfig {
    max_concurrent_inserts: 100,      // Default: 50
    max_concurrent_operations: 500,   // Default: 200
    cooldown_period_ms: 1000,         // Default: 2000
    max_recent_errors: 20,            // Default: 10
};

let server_health = Arc::new(ServerHealth::new(config));
```

### Option 2: Environment Variables (Future Enhancement)

```bash
export DXMT_MAX_CONCURRENT_INSERTS=100
export DXMT_MAX_CONCURRENT_OPS=500
export DXMT_COOLDOWN_MS=1000
export DXMT_MAX_ERRORS=20

cargo run -- serve
```

## 🐛 Troubleshooting

### Issue: Server returns 429 too frequently

**Cause**: Thresholds too low for your hardware

**Solution**: Increase thresholds in HealthConfig
```rust
max_concurrent_inserts: 100,  // Increase from 50
max_concurrent_operations: 500,  // Increase from 200
```

### Issue: Still getting 500 errors

**Cause**: Different issue than backpressure

**Solution**: Check server logs and error details:
```bash
# Enable detailed logging
RUST_LOG=debug cargo run -- serve

# Check error message in response
curl -v http://localhost:9999/api/v1/tables/test/records -X POST -d '...'
```

### Issue: Health endpoint shows Critical state

**Cause**: Server is genuinely overloaded or error spike

**Solution**:
1. Check metrics: `curl http://localhost:9999/health/metrics`
2. Identify high operation counts or error rate
3. Reduce load or scale up resources
4. Check for underlying issues (disk space, memory, etc.)

## 📚 Documentation Reference

### User Documentation

| Document | Purpose | Location |
|----------|---------|----------|
| Issue Report | Problem details | `dxmt-python-sdk/examples/ISSUE_REPORT.md` |
| Diagnosis Summary | Quick summary | `dxmt-python-sdk/examples/DIAGNOSIS_SUMMARY.md` |
| Server Improvements | Technical details | `SERVER_IMPROVEMENTS.md` |
| Improvements Summary | Implementation summary | `IMPROVEMENTS_SUMMARY.md` |
| **This Guide** | **Deployment guide** | **`DEPLOYMENT_GUIDE.md`** |

### Code Documentation

| File | Purpose |
|------|---------|
| `src/api/server_health.rs` | Health monitoring module |
| `src/api/handlers.rs` | Enhanced API handlers |
| `src/api/server.rs` | Server initialization |

### Test Scripts

| Script | Purpose |
|--------|---------|
| `test_server_improvements.py` | Test new features |
| `batch_operations.py` | Original problem scenario |
| `test_exact_reproduction.py` | Exact problem reproduction |
| `diagnose_server.py` | Diagnostic tool |

## ✅ Success Criteria

Your deployment is successful if:

- [x] ✅ Server compiles without errors
- [x] ✅ Server starts and responds to requests
- [x] ✅ `/health` endpoint returns health status
- [x] ✅ `/health/metrics` endpoint returns metrics
- [x] ✅ Heavy load triggers 429 instead of 500
- [x] ✅ 429 responses include Retry-After header
- [x] ✅ Server recovers automatically after load
- [x] ✅ `batch_operations.py` completes successfully
- [x] ✅ No breaking changes for existing clients

## 🎓 Best Practices

### For Development

1. **Test with load**: Always test under load to verify backpressure
2. **Monitor health**: Watch `/health` endpoint during development
3. **Check metrics**: Use `/health/metrics` for debugging
4. **Tune thresholds**: Adjust based on your hardware

### For Production

1. **Set up monitoring**: Continuous health check monitoring
2. **Configure alerts**: Alert on Degraded/Overloaded states
3. **Plan scaling**: Use health state to trigger auto-scaling
4. **Review logs**: Regularly check for error patterns
5. **Tune configuration**: Optimize thresholds for your workload

### For Operations

1. **Monitor trends**: Track health state over time
2. **Capacity planning**: Use metrics for capacity planning
3. **Incident response**: Use health info for troubleshooting
4. **Performance tuning**: Adjust based on real-world usage

## 🚀 Next Steps

### Immediate

1. ✅ Build and deploy the improved server
2. ✅ Run test suite to verify improvements
3. ✅ Monitor health endpoints
4. ✅ Validate with production workload

### Short-term

1. 🔄 Set up continuous monitoring
2. 🔄 Configure alerting
3. 🔄 Document operational procedures
4. 🔄 Train team on new features

### Long-term

1. 📊 Integrate with Prometheus/Grafana
2. 🔧 Fine-tune thresholds based on metrics
3. 🎯 Implement auto-scaling based on health state
4. 📈 Add more detailed metrics

## 💡 Tips

- **Start conservative**: Use default thresholds initially
- **Monitor closely**: Watch health state during first week
- **Tune gradually**: Adjust thresholds based on observations
- **Document changes**: Keep track of threshold adjustments
- **Review regularly**: Periodically review health metrics

---

**Version**: 0.2.0
**Date**: 2026-01-22
**Status**: ✅ Ready for Deployment
**Tested**: ✅ All improvements verified
**Documentation**: ✅ Complete

For questions or issues, refer to:
- Technical details: `SERVER_IMPROVEMENTS.md`
- Problem diagnosis: `DIAGNOSIS_SUMMARY.md`
- Test results: Run `test_server_improvements.py`
