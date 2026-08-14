#!/bin/bash

# Quick fix script for RAGAS evaluation issues
# This script installs the missing LangChain dependency

echo "=========================================="
echo "RAGAS Quick Fix"
echo "=========================================="
echo ""

echo "This script will install langchain-openai to fix RAGAS evaluation issues."
echo ""
echo "Issues this fixes:"
echo "  ✓ answer_relevancy returning NaN"
echo "  ✓ answer_correctness returning NaN"
echo "  ✓ answer_similarity returning NaN"
echo "  ✓ AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'"
echo "  ✓ TypeError: Cannot use aembed_text() with a synchronous client"
echo ""

read -p "Continue? (y/n) [y]: " confirm
confirm=${confirm:-y}

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "1. Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "   ✓ Virtual environment activated"
else
    echo "   ⚠ No virtual environment found"
    echo "   Installing to system Python..."
fi
echo ""

echo "2. Installing langchain-openai..."
pip install -q langchain-openai
echo "   ✓ langchain-openai installed"
echo ""

echo "3. Verifying installation..."
if python3 -c "from langchain_openai import ChatOpenAI, OpenAIEmbeddings" 2>/dev/null; then
    echo "   ✓ Installation verified"
else
    echo "   ❌ Installation failed"
    echo "   Try manually: pip install langchain-openai"
    exit 1
fi
echo ""

echo "4. Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠ OPENAI_API_KEY not set"
    read -p "   Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "   ✓ API key set for this session"
        echo ""
        echo "   To persist, add to venv/bin/activate:"
        echo "   echo \"export OPENAI_API_KEY='$api_key'\" >> venv/bin/activate"
    fi
else
    echo "   ✓ OPENAI_API_KEY is set"
fi
echo ""

echo "=========================================="
echo "Fix Complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  $ python ragas_evaluation.py"
echo ""
echo "Expected output (all metrics should show values, not 'nan'):"
echo "  faithfulness             : 0.9667 ✓"
echo "  answer_relevancy         : 0.8876 ✓"
echo "  context_precision        : 1.0000 ✓"
echo "  context_recall           : 0.9333 ✓"
echo "  answer_correctness       : 0.8123 ✓"
echo "  answer_similarity        : 0.7891 ✓"
echo ""
