#!/bin/bash
# Script to run code quality checks
# Usage: ./scripts/check_quality.sh [lint|format|type|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if tools are installed
check_dependencies() {
    local missing=()

    if ! command -v ruff &> /dev/null; then
        missing+=("ruff")
    fi

    if ! command -v mypy &> /dev/null; then
        missing+=("mypy")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        print_info "Run: pip install -e \".[dev]\""
        exit 1
    fi
}

# Run linting
run_linting() {
    print_header "Running Ruff Linter"
    print_info "Checking code style and potential bugs..."
    echo ""

    if ruff check langchain_dxmt; then
        print_info "✅ No linting errors found!"
    else
        print_error "❌ Linting errors found"
        print_info "Run: ruff check --fix langchain_dxmt (to auto-fix)"
        return 1
    fi
    echo ""
}

# Run formatting check
run_format_check() {
    print_header "Running Ruff Formatter"
    print_info "Checking code formatting..."
    echo ""

    if ruff format --check langchain_dxmt; then
        print_info "✅ Code is properly formatted!"
    else
        print_error "❌ Code formatting issues found"
        print_info "Run: ruff format langchain_dxmt (to auto-format)"
        return 1
    fi
    echo ""
}

# Run type checking
run_type_checking() {
    print_header "Running MyPy Type Checker"
    print_info "Checking type hints..."
    echo ""

    if mypy langchain_dxmt; then
        print_info "✅ No type errors found!"
    else
        print_error "❌ Type errors found"
        return 1
    fi
    echo ""
}

# Run all checks
run_all_checks() {
    local failed=0

    run_linting || failed=$((failed + 1))
    run_format_check || failed=$((failed + 1))
    run_type_checking || failed=$((failed + 1))

    if [ $failed -eq 0 ]; then
        print_header "All Checks Passed! ✅"
        print_info "Code quality is excellent!"
        return 0
    else
        print_error "❌ $failed check(s) failed"
        return 1
    fi
}

# Parse command line argument
CHECK_TYPE=${1:-all}

# Check dependencies
check_dependencies

# Main execution
case $CHECK_TYPE in
    lint)
        run_linting
        ;;
    format)
        run_format_check
        ;;
    type)
        run_type_checking
        ;;
    all)
        run_all_checks
        ;;
    *)
        print_error "Unknown check type: $CHECK_TYPE"
        echo "Usage: $0 [lint|format|type|all]"
        exit 1
        ;;
esac
