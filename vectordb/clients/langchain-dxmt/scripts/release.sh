#!/bin/bash
# Script to prepare and publish a release
# Usage: ./scripts/release.sh [check|build|test-pypi|pypi]

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

# Check dependencies
check_dependencies() {
    local missing=()

    if ! command -v python &> /dev/null; then
        missing+=("python")
    fi

    if ! python -m pip show build &> /dev/null; then
        missing+=("build")
    fi

    if ! python -m pip show twine &> /dev/null; then
        missing+=("twine")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        print_info "Run: pip install build twine"
        exit 1
    fi
}

# Get current version from pyproject.toml
get_version() {
    python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || \
    grep -oP 'version = "\K[^"]+' pyproject.toml | head -1
}

# Run pre-release checks
run_checks() {
    print_header "Pre-Release Checks"

    # Check if on main branch
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ]; then
        print_warning "Not on main branch (current: $current_branch)"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_error "Uncommitted changes detected"
        print_info "Please commit or stash changes before release"
        exit 1
    fi

    # Run quality checks
    print_info "Running code quality checks..."
    if ! ./scripts/check_quality.sh all; then
        print_error "Quality checks failed"
        exit 1
    fi

    # Run tests
    print_info "Running unit tests..."
    if ! pytest tests/unit_tests/ -v; then
        print_error "Tests failed"
        exit 1
    fi

    # Get version
    version=$(get_version)
    print_info "Current version: $version"

    # Check if tag exists
    if git rev-parse "v$version" >/dev/null 2>&1; then
        print_error "Tag v$version already exists"
        print_info "Update version in pyproject.toml before release"
        exit 1
    fi

    print_info "✅ All pre-release checks passed!"
    echo ""
}

# Build package
build_package() {
    print_header "Building Package"

    # Clean previous builds
    print_info "Cleaning previous builds..."
    rm -rf dist/ build/ *.egg-info
    echo ""

    # Build package
    print_info "Building package..."
    python -m build
    echo ""

    # Check package
    print_info "Checking package..."
    twine check dist/*
    echo ""

    # List built files
    print_info "Built files:"
    ls -lh dist/
    echo ""

    version=$(get_version)
    print_info "✅ Package v$version built successfully!"
    echo ""
}

# Upload to Test PyPI
upload_test_pypi() {
    print_header "Uploading to Test PyPI"

    if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
        print_error "No distribution files found"
        print_info "Run: ./scripts/release.sh build"
        exit 1
    fi

    print_warning "This will upload to Test PyPI"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    print_info "Uploading to Test PyPI..."
    twine upload --repository testpypi dist/*
    echo ""

    version=$(get_version)
    print_info "✅ Uploaded to Test PyPI!"
    print_info "Test install with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ langchain-dxmt==$version"
    echo ""
}

# Upload to PyPI
upload_pypi() {
    print_header "Uploading to PyPI"

    if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
        print_error "No distribution files found"
        print_info "Run: ./scripts/release.sh build"
        exit 1
    fi

    version=$(get_version)

    print_warning "⚠️  WARNING: This will publish v$version to PyPI!"
    print_warning "This action cannot be undone!"
    echo ""
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    print_info "Uploading to PyPI..."
    twine upload dist/*
    echo ""

    print_info "✅ Published to PyPI!"
    print_info "Install with: pip install langchain-dxmt==$version"
    print_info "View at: https://pypi.org/project/langchain-dxmt/$version/"
    echo ""

    # Suggest next steps
    print_header "Next Steps"
    print_info "1. Create Git tag:"
    echo "     git tag -a v$version -m \"Release v$version\""
    echo "     git push origin v$version"
    echo ""
    print_info "2. Create GitHub release:"
    echo "     Go to: https://github.com/dxmt/dxmt/releases/new"
    echo "     Tag: v$version"
    echo "     Copy CHANGELOG.md content for this version"
    echo ""
    print_info "3. Announce release:"
    echo "     - LangChain Discord"
    echo "     - Reddit r/LangChain"
    echo "     - Twitter/X"
    echo ""
}

# Full release workflow
full_release() {
    print_header "Full Release Workflow"

    print_info "This will:"
    echo "  1. Run all pre-release checks"
    echo "  2. Build the package"
    echo "  3. Upload to Test PyPI"
    echo "  4. Prompt for PyPI upload"
    echo ""

    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    # Run checks
    run_checks

    # Build
    build_package

    # Test PyPI
    print_info "Uploading to Test PyPI first..."
    upload_test_pypi

    # Verify Test PyPI
    print_warning "Please verify the Test PyPI package before proceeding"
    print_info "Test installation command shown above"
    echo ""
    read -p "Verified and ready to upload to production PyPI? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Stopping before production upload"
        exit 0
    fi

    # Production PyPI
    upload_pypi
}

# Parse command line argument
ACTION=${1:-check}

# Check dependencies
check_dependencies

# Main execution
case $ACTION in
    check)
        run_checks
        ;;
    build)
        build_package
        ;;
    test-pypi)
        upload_test_pypi
        ;;
    pypi)
        upload_pypi
        ;;
    full)
        full_release
        ;;
    *)
        print_error "Unknown action: $ACTION"
        echo "Usage: $0 [check|build|test-pypi|pypi|full]"
        echo ""
        echo "Actions:"
        echo "  check      - Run pre-release checks"
        echo "  build      - Build package"
        echo "  test-pypi  - Upload to Test PyPI"
        echo "  pypi       - Upload to production PyPI"
        echo "  full       - Run complete release workflow"
        exit 1
        ;;
esac
