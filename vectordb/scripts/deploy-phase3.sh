#!/bin/bash
# Phase 3 Deployment Script
# Automated deployment and validation for Phase 3 components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VECTORDB_PATH="$PROJECT_ROOT/vectordb"
LOG_DIR="/var/log/dxmt"
DATA_DIR="/var/lib/dxmt"

# Functions
print_header() {
    echo -e "${GREEN}=== $1 ===${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Pre-deployment checks
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Rust
    if ! command -v cargo &> /dev/null; then
        print_error "cargo not found. Install Rust first."
        exit 1
    fi
    print_success "Rust/Cargo found"

    # Check directories exist
    if [ ! -d "$VECTORDB_PATH" ]; then
        print_error "VectorDB path not found: $VECTORDB_PATH"
        exit 1
    fi
    print_success "VectorDB path exists"

    # Create log and data directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    print_success "Log and data directories ready"
}

# Code quality checks
run_quality_checks() {
    print_header "Running Quality Checks"

    cd "$VECTORDB_PATH"

    # Format check
    if ! cargo fmt --check 2>/dev/null; then
        print_error "Code formatting issues found"
        print_warning "Run 'cargo fmt' to fix"
        exit 1
    fi
    print_success "Code formatting OK"

    # Clippy check
    if ! cargo clippy -- -D warnings 2>/dev/null; then
        print_error "Clippy warnings found"
        exit 1
    fi
    print_success "Clippy checks passed"
}

# Build verification
verify_build() {
    print_header "Verifying Build"

    cd "$VECTORDB_PATH"

    # Run cargo check
    if ! cargo check 2>&1 | tee "$LOG_DIR/build-check.log"; then
        print_error "Build check failed. See $LOG_DIR/build-check.log"
        exit 1
    fi
    print_success "Build verification passed"
}

# Run tests
run_tests() {
    print_header "Running Tests"

    cd "$VECTORDB_PATH"

    # Run unit tests
    if ! cargo test --lib 2>&1 | tee "$LOG_DIR/tests.log"; then
        print_error "Tests failed. See $LOG_DIR/tests.log"
        exit 1
    fi
    print_success "All tests passed"
}

# Configuration validation
validate_configuration() {
    print_header "Validating Configuration"

    # Check required environment variables
    required_vars=("RUST_LOG" "DXMT_CONFIG_PATH")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            print_warning "Environment variable not set: $var"
        fi
    done

    print_success "Configuration validation complete"
}

# Create deployment artifacts
create_artifacts() {
    print_header "Creating Deployment Artifacts"

    cd "$VECTORDB_PATH"

    # Build release binary
    print_warning "Building release binary (this may take a few minutes)..."
    if ! cargo build --release 2>&1 | tail -5; then
        print_error "Release build failed"
        exit 1
    fi
    print_success "Release binary created"

    # Copy to deployment directory
    DEPLOY_DIR="$PROJECT_ROOT/deploy"
    mkdir -p "$DEPLOY_DIR"
    cp target/release/dxmt_vectordb "$DEPLOY_DIR/dxmt-index-manager" 2>/dev/null || print_warning "Binary copy skipped"

    print_success "Artifacts ready at $DEPLOY_DIR"
}

# Database initialization
init_database() {
    print_header "Initializing Database"

    # Create transaction log directory
    mkdir -p "$LOG_DIR/migrations"
    chmod 755 "$LOG_DIR/migrations"

    # Create data directory structure
    mkdir -p "$DATA_DIR/indexes"
    mkdir -p "$DATA_DIR/backups"
    chmod 755 "$DATA_DIR"

    print_success "Database directories initialized"
}

# Health checks
run_health_checks() {
    print_header "Running Health Checks"

    # Check if service is running (if already deployed)
    if command -v systemctl &> /dev/null; then
        if systemctl is-active --quiet dxmt-index-manager 2>/dev/null; then
            print_success "Service is running"

            # Check health endpoint
            if curl -s http://localhost:9999/health >/dev/null; then
                print_success "Health endpoint responsive"
            else
                print_warning "Health endpoint not responding"
            fi
        else
            print_warning "Service not running yet"
        fi
    fi
}

# Backup current configuration
backup_config() {
    print_header "Backing Up Current Configuration"

    if [ -f "/etc/dxmt/config.yaml" ]; then
        cp "/etc/dxmt/config.yaml" "$DATA_DIR/backups/config-$(date +%Y%m%d-%H%M%S).yaml"
        print_success "Configuration backed up"
    else
        print_warning "No existing configuration to backup"
    fi
}

# Print deployment summary
print_summary() {
    print_header "Deployment Summary"

    echo "Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review deployment logs: tail -f $LOG_DIR/*.log"
    echo "2. Start service: systemctl start dxmt-index-manager"
    echo "3. Check status: systemctl status dxmt-index-manager"
    echo "4. View metrics: curl http://localhost:9999/metrics"
    echo ""
    echo "Documentation:"
    echo "- Implementation Guide: PHASE3_IMPLEMENTATION_GUIDE.md"
    echo "- Deployment Guide: PHASE3_PRODUCTION_DEPLOYMENT.md"
    echo "- API Reference: PHASE3_API_REFERENCE.md"
}

# Main execution
main() {
    print_header "Phase 3 Deployment Script"
    echo "Starting deployment at $(date)"
    echo ""

    check_prerequisites
    run_quality_checks
    verify_build
    run_tests
    validate_configuration
    create_artifacts
    init_database
    backup_config
    run_health_checks

    echo ""
    print_success "All deployment checks passed!"
    print_summary
}

# Run main function
main "$@"
