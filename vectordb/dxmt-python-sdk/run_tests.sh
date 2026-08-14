#!/bin/bash
# DXMT Python SDK - Test Runner Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate virtual environment
if [ -d "venv/bin" ]; then
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -e '.[dev]'"
    exit 1
fi

# Function to print section headers
print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

# Parse command line arguments
TEST_TYPE="${1:-all}"

case "$TEST_TYPE" in
    unit)
        print_header "Running Unit Tests"
        pytest tests/test_basic.py tests/test_client_complete.py -v
        ;;

    integration)
        print_header "Running Integration Tests"
        echo -e "${YELLOW}Note: Integration tests require a running DXMT server at localhost:9999${NC}\n"
        pytest tests/test_integration.py -v -m integration
        ;;

    basic)
        print_header "Running Basic Tests Only"
        pytest tests/test_basic.py -v
        ;;

    coverage)
        print_header "Running Tests with Coverage"
        pytest tests/test_basic.py tests/test_client_complete.py \
            --cov=dxmt \
            --cov-report=term-missing \
            --cov-report=html
        echo -e "\n${GREEN}Coverage report generated in: htmlcov/index.html${NC}"
        ;;

    full-coverage)
        print_header "Running All Tests with Coverage"
        echo -e "${YELLOW}Note: This requires a running DXMT server${NC}\n"
        pytest tests/ \
            --cov=dxmt \
            --cov-report=term-missing \
            --cov-report=html
        echo -e "\n${GREEN}Coverage report generated in: htmlcov/index.html${NC}"
        ;;

    all)
        print_header "Running All Tests (Unit + Integration)"
        echo -e "${YELLOW}Note: Integration tests require a running DXMT server at localhost:9999${NC}\n"

        echo -e "${GREEN}1. Running Unit Tests...${NC}"
        pytest tests/test_basic.py tests/test_client_complete.py -v

        echo -e "\n${GREEN}2. Running Integration Tests...${NC}"
        pytest tests/test_integration.py -v -m integration || {
            echo -e "\n${YELLOW}Integration tests skipped (server not available)${NC}"
        }
        ;;

    quick)
        print_header "Running Quick Tests (Unit Only)"
        pytest tests/test_basic.py tests/test_client_complete.py -v --tb=line
        ;;

    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo ""
        echo "Usage: $0 [test_type]"
        echo ""
        echo "Test types:"
        echo "  unit         - Run unit tests only (no server required)"
        echo "  integration  - Run integration tests (requires DXMT server)"
        echo "  basic        - Run basic functionality tests only"
        echo "  coverage     - Run unit tests with coverage report"
        echo "  full-coverage- Run all tests with coverage (requires server)"
        echo "  all          - Run all tests (default)"
        echo "  quick        - Run quick unit tests with minimal output"
        echo ""
        exit 1
        ;;
esac

echo -e "\n${GREEN}✓ Tests completed${NC}\n"
