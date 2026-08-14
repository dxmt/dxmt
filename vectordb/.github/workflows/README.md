# GitHub Workflows Documentation

This directory contains automated workflows for CI/CD, testing, and benchmarking of DXMT VectorDB.

## Workflow Overview

### 1. **CI (ci.yml)**
Main continuous integration workflow that runs on every push and pull request.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Daily scheduled run

**Jobs:**
- **Lint**: Code formatting and clippy checks
- **Test**: Unit and doc tests on multiple OS/Rust versions
- **Security**: Dependency vulnerability scanning
- **Coverage**: Code coverage generation and reporting
- **Docker**: Build and push Docker images
- **Benchmark**: Performance benchmarks on main branch

### 2. **Single Node Tests (test-single-node.yml)**
Comprehensive testing of single-node deployment.

**Triggers:**
- Push to main branches
- Pull requests
- Manual dispatch

**Test Categories:**
- **Functional Tests**:
  - CRUD operations
  - API endpoint testing
  - Data persistence verification
  
- **Performance Tests**:
  - Insert throughput (batch sizes: 100, 500, 1000)
  - Query latency (P50, P95, P99)
  - Concurrent load testing (10 threads, 50 queries each)
  - Memory usage analysis
  
- **Stress Tests**:
  - 2-minute continuous load
  - Server stability verification

**Performance Targets:**
- Insert: > 5,000 vectors/sec
- Query P95: < 15ms
- QPS: > 10,000

### 3. **Cluster Tests (test-cluster.yml)**
Multi-node cluster testing with Consul coordination.

**Triggers:**
- Push to main branches
- Pull requests
- Manual dispatch with configurable node count

**Test Scenarios:**
- **Functional Tests**:
  - Cluster formation and coordination
  - Data replication verification
  - Node failure resilience
  - Load balancing
  
- **Performance Tests**:
  - Distributed insert benchmark
  - Distributed query benchmark
  - Scalability testing
  - Resource utilization

**Infrastructure:**
- 3-node cluster by default
- Consul for service discovery
- MinIO for object storage
- Replication factor: 2

### 4. **Performance Benchmark (benchmark.yml)**
Comprehensive performance benchmarking suite.

**Triggers:**
- Push to `main`
- Pull requests
- Daily scheduled run at 2 AM UTC
- Manual dispatch

**Benchmark Categories:**
- **Algorithm Comparison**:
  - HNSW vs Brute Force vs IVF-PQ
  - Multiple dimensions (128, 384, 768, 1536)
  - Various dataset sizes (1K, 10K, 50K)
  
- **Memory Profiling**:
  - Memory usage per vector
  - Memory growth patterns
  
- **Concurrent Load**:
  - Read-only, mixed, and write-only workloads
  - Concurrency levels: 1, 5, 10, 20, 50
  
- **Performance Tracking**:
  - Historical comparison
  - Regression detection
  - Automated reporting

### 5. **CD (cd.yml)**
Continuous deployment workflow for releases.

**Features:**
- Automated deployment to staging/production
- Multi-environment support
- Rollback capabilities

### 6. **Release (release.yml)**
Automated release management.

**Features:**
- Version tagging
- Changelog generation
- Asset building and uploading
- Crate publishing

## Workflow Dependencies

### Required Secrets
Configure these in GitHub repository settings:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `GITHUB_TOKEN`: Automatically provided

### External Services
- **Consul**: Service discovery (via Docker)
- **MinIO**: Object storage testing
- **Codecov**: Coverage reporting

## Running Workflows Locally

Use [act](https://github.com/nektos/act) to run workflows locally:

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI workflow
act push

# Run specific job
act -j test

# Run with specific event
act pull_request
```

## Performance Baselines

### Single Node
| Metric | Target | Current |
|--------|--------|---------|
| Insert Throughput | 5K vec/s | ~6.5K vec/s |
| Query P95 | < 15ms | ~12ms |
| Memory/Vector | < 3KB | ~2.5KB |
| Max QPS | 10K | ~12K |

### Cluster (3 nodes)
| Metric | Target | Current |
|--------|--------|---------|
| Insert Throughput | 15K vec/s | ~18K vec/s |
| Query P95 | < 20ms | ~16ms |
| Node Failure Recovery | < 30s | ~15s |
| Rebalance Time | < 5min | ~3min |

## Monitoring Workflow Status

### Badge URLs
Add these to your README:

```markdown
![CI](https://github.com/your-org/dxmt-vectordb/workflows/CI/badge.svg)
![Single Node Tests](https://github.com/your-org/dxmt-vectordb/workflows/Single%20Node%20Tests/badge.svg)
![Cluster Tests](https://github.com/your-org/dxmt-vectordb/workflows/Cluster%20Tests/badge.svg)
![Benchmark](https://github.com/your-org/dxmt-vectordb/workflows/Performance%20Benchmark/badge.svg)
```

### Workflow Artifacts
Each workflow produces artifacts:
- **Test results**: JUnit XML format
- **Coverage reports**: HTML and Cobertura XML
- **Performance reports**: Markdown and CSV
- **Benchmark plots**: PNG images
- **Logs**: Debug information

## Troubleshooting

### Common Issues

1. **Timeout in cluster tests**
   - Increase sleep times for service startup
   - Check Consul connectivity

2. **Memory test failures**
   - Adjust memory limits in workflow
   - Check for memory leaks

3. **Performance regression**
   - Compare with baseline metrics
   - Check for debug builds

### Debug Mode
Enable debug logging:
```yaml
env:
  RUST_LOG: debug
  RUST_BACKTRACE: full
```

## Contributing

When adding new workflows:
1. Follow naming convention: `test-*.yml` for tests, `*.yml` for others
2. Include comprehensive documentation
3. Add status badges to main README
4. Set appropriate timeouts
5. Handle cleanup in `always()` blocks

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Virtual Environments](https://github.com/actions/virtual-environments)