#!/bin/bash

echo "=========================================="
echo "DXMT JavaScript SDK - Verification Script"
echo "=========================================="

# Check Node.js version
echo ""
echo "📊 Checking Node.js version..."
node_version=$(node -v)
echo "   Node.js: $node_version"

if command -v npm &> /dev/null; then
    npm_version=$(npm -v)
    echo "   npm: $npm_version"
fi

# Install dependencies
echo ""
echo "📊 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"

# Build the SDK
echo ""
echo "📊 Building SDK..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ SDK built successfully"

# Check build output
echo ""
echo "📊 Checking build output..."
if [ -d "dist" ]; then
    echo "✅ dist/ directory created"
    echo "   Files:"
    ls -lh dist/ | grep -E '\.(js|mjs|d\.ts)$' | awk '{print "   -", $9, "(" $5 ")"}'
else
    echo "❌ dist/ directory not found"
    exit 1
fi

# Run tests
echo ""
echo "📊 Running tests..."
npm test

if [ $? -ne 0 ]; then
    echo "⚠️  Some tests failed"
else
    echo "✅ All tests passed"
fi

# Summary
echo ""
echo "=========================================="
echo "✅ DXMT JavaScript SDK Verification Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999"
echo "2. Run example: node dist/examples/basic-usage.js"
echo "3. Or use TypeScript: npx ts-node examples/basic-usage.ts"
echo ""
