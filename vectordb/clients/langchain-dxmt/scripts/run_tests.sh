#!/bin/bash
# Script to run all tests for langchain-dxmt
# Usage: ./scripts/run_tests.sh [unit|integration|standard|performance|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed. Run: pip install -e \".[dev]\""
    exit 1
fi

# Parse command line argument
TEST_TYPE=${1:-all}

print_info "Running tests: $TEST_TYPE"
echo ""

# Run unit tests
run_unit_tests() {
    print_info "Running unit tests..."
    pytest tests/unit_tests/ -v --cov=langchain_dxmt --cov-report=term-missing
    echo ""
}

# Run integration tests
run_integration_tests() {
    print_info "Running integration tests..."

    # Check if DXMT server is running
    if ! nc -z localhost 9999 2>/dev/null; then
        print_warning "DXMT server not detected on localhost:9999"
        print_warning "Integration tests may fail or be skipped"
    fi

    # Check for OpenAI API key
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY not set"
        print_warning "Some integration tests will be skipped"
    fi

    pytest tests/integration_tests/ -v
    echo ""
}

# Run LangChain standard tests
run_standard_tests() {
    print_info "Running LangChain standard compliance tests..."

    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY not set"
        print_warning "Standard tests will be skipped"
        return
    fi

    pytest tests/langchain_standard/ -v
    echo ""
}

# Run performance benchmarks
run_performance_tests() {
    print_info "Running performance benchmarks..."

    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY not set"
        print_warning "Performance tests will be skipped"
        return
    fi

    python tests/performance/bench_dxmt.py
    echo ""
}

# Main execution
case $TEST_TYPE in
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    standard)
        run_standard_tests
        ;;
    performance)
        run_performance_tests
        ;;
    all)
        print_info "Running all tests..."
        echo ""
        run_unit_tests
        run_integration_tests
        run_standard_tests
        run_performance_tests
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        echo "Usage: $0 [unit|integration|standard|performance|all]"
        exit 1
        ;;
esac

print_info "Tests completed!"
