#!/bin/bash
# Script to set up development environment
# Usage: ./scripts/setup_dev.sh

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

print_header "LangChain-DXMT Development Setup"
echo ""

# Check Python version
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

print_info "Python version: $python_version"

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]); then
    print_error "Python 3.9 or higher required"
    print_info "Current version: $python_version"
    exit 1
fi

print_info "✅ Python version OK"
echo ""

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python -m venv venv
    print_info "✅ Virtual environment created"
else
    print_info "Virtual environment already exists"
fi
echo ""

# Activate virtual environment message
print_warning "Please activate the virtual environment:"
echo "  source venv/bin/activate    # Linux/macOS"
echo "  venv\\Scripts\\activate       # Windows"
echo ""
read -p "Press Enter after activating the virtual environment..."
echo ""

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install package with dev dependencies
print_info "Installing langchain-dxmt with dev dependencies..."
pip install -e ".[dev]"
echo ""

# Install additional tools
print_info "Installing additional development tools..."
pip install pre-commit pip-audit
echo ""

# Make scripts executable
print_info "Making scripts executable..."
chmod +x scripts/*.sh
print_info "✅ Scripts are now executable"
echo ""

# Set up pre-commit hooks (optional)
read -p "Install pre-commit hooks? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Installing pre-commit hooks..."
    pre-commit install
    print_info "✅ Pre-commit hooks installed"
else
    print_info "Skipping pre-commit hooks"
fi
echo ""

# Check DXMT server
print_info "Checking for DXMT server..."
if nc -z localhost 9999 2>/dev/null; then
    print_info "✅ DXMT server detected on localhost:9999"
else
    print_warning "DXMT server not detected on localhost:9999"
    print_info "You'll need a running DXMT server for integration tests"
    echo ""
    print_info "To start DXMT server:"
    echo "  cd ../../"
    echo "  cargo run --release -- serve"
fi
echo ""

# Check environment variables
print_info "Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    print_warning "OPENAI_API_KEY not set"
    print_info "Set it for integration tests:"
    echo "  export OPENAI_API_KEY=\"your-key-here\""
else
    print_info "✅ OPENAI_API_KEY is set"
fi
echo ""

# Run initial checks
print_info "Running initial quality checks..."
if ./scripts/check_quality.sh all; then
    print_info "✅ Code quality checks passed"
else
    print_warning "Some quality checks failed (expected for first-time setup)"
fi
echo ""

# Run unit tests
print_info "Running unit tests..."
if pytest tests/unit_tests/ -v; then
    print_info "✅ Unit tests passed"
else
    print_error "❌ Unit tests failed"
    print_warning "Please investigate test failures"
fi
echo ""

# Setup complete
print_header "Setup Complete! 🎉"
echo ""
print_info "Development environment is ready!"
echo ""
print_info "Quick commands:"
echo "  ./scripts/check_quality.sh all    # Run code quality checks"
echo "  ./scripts/run_tests.sh all        # Run all tests"
echo "  ./scripts/release.sh check        # Check release readiness"
echo ""
print_info "Documentation:"
echo "  README.md                         # Getting started"
echo "  CONTRIBUTING.md                   # Contribution guidelines"
echo "  docs/API_REFERENCE.md             # API documentation"
echo ""
print_info "Happy coding! 🚀"
