#!/bin/bash

# Setup script for DXMT LlamaIndex examples
# This script installs all required dependencies for running the examples

set -e  # Exit on error

echo "=========================================="
echo "DXMT LlamaIndex Examples Setup"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

required_major=3
required_minor=9

version_check=$(python3 -c "import sys; print(1 if sys.version_info >= ($required_major, $required_minor) else 0)")
if [ "$version_check" != "1" ]; then
    echo "   ❌ Error: Python 3.9+ required"
    exit 1
fi
echo "   ✓ Python version OK"
echo ""

# Check if virtual environment exists
echo "2. Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment exists"
fi
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "4. Upgrading pip..."
pip install --upgrade pip -q
echo "   ✓ pip upgraded"
echo ""

# Install core dependencies
echo "5. Installing core dependencies..."
pip install -q llama-index-core llama-index-vector-stores-dxmt
echo "   ✓ Core dependencies installed"
echo ""

# Install OpenAI dependencies
echo "6. Installing OpenAI dependencies..."
pip install -q llama-index-embeddings-openai llama-index-llms-openai
echo "   ✓ OpenAI dependencies installed"
echo ""

# Install RAGAS (optional)
echo "7. Installing RAGAS for evaluation (optional)..."
read -p "   Install RAGAS? (y/n) [y]: " install_ragas
install_ragas=${install_ragas:-y}

if [ "$install_ragas" = "y" ] || [ "$install_ragas" = "Y" ]; then
    pip install -q ragas datasets langchain-openai
    echo "   ✓ RAGAS installed (with LangChain support)"
else
    echo "   ⊘ RAGAS installation skipped"
fi
echo ""

# Check OpenAI API key
echo "8. Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠ OPENAI_API_KEY not set"
    read -p "   Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "export OPENAI_API_KEY='$api_key'" >> venv/bin/activate
        echo "   ✓ API key set (saved to venv/bin/activate)"
    else
        echo "   ⊘ API key setup skipped"
        echo "   You can set it later with: export OPENAI_API_KEY='your-key'"
    fi
else
    echo "   ✓ OPENAI_API_KEY is set"
fi
echo ""

# Check DXMT server
echo "9. Checking DXMT server..."
if curl -s http://localhost:9999/health > /dev/null 2>&1; then
    echo "   ✓ DXMT server is running on localhost:9999"
else
    echo "   ⚠ DXMT server is not running"
    echo ""
    echo "   To start the DXMT server:"
    echo "   $ cd ../../  # Go to vectordb directory"
    echo "   $ cargo run -- serve --simple --memory-limit 1024"
    echo ""
fi
echo ""

# List available examples
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Available examples:"
echo "  1. basic_usage.py         - Basic vector store operations"
echo "  2. advanced_filters.py    - Complex metadata filtering"
echo "  3. rag_example.py         - Complete RAG pipeline"
echo "  4. hybrid_search.py       - Vector + keyword search"
echo "  5. ragas_evaluation.py    - RAG evaluation with RAGAS"
echo ""
echo "To run an example:"
echo "  $ source venv/bin/activate"
echo "  $ python basic_usage.py"
echo ""
echo "For more information, see README.md"
echo ""
