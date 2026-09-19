#!/bin/bash
#
# Diagnose Storage Configuration
#
# This script helps diagnose why data might not be appearing in GCS
#

echo "=========================================="
echo "Storage Configuration Diagnostic"
echo "=========================================="
echo ""

echo "=== Environment Variables ==="
echo "DXMT_STORAGE_TYPE: ${DXMT_STORAGE_TYPE:-NOT SET}"
echo "DXMT_OBJECT_STORAGE_PROVIDER: ${DXMT_OBJECT_STORAGE_PROVIDER:-NOT SET}"
echo "DXMT_S3_BUCKET: ${DXMT_S3_BUCKET:-NOT SET}"
echo "DXMT_S3_ENDPOINT: ${DXMT_S3_ENDPOINT:-NOT SET}"
echo "DXMT_S3_REGION: ${DXMT_S3_REGION:-NOT SET}"
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:+SET (${#AWS_ACCESS_KEY_ID} chars)}"
echo "AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:+SET (${#AWS_SECRET_ACCESS_KEY} chars)}"
echo "DXMT_AWS_ACCESS_KEY_ID: ${DXMT_AWS_ACCESS_KEY_ID:+SET (${#DXMT_AWS_ACCESS_KEY_ID} chars)}"
echo "DXMT_AWS_SECRET_ACCESS_KEY: ${DXMT_AWS_SECRET_ACCESS_KEY:+SET (${#DXMT_AWS_SECRET_ACCESS_KEY} chars)}"
echo "DXMT_MEMORY_CACHE_SIZE: ${DXMT_MEMORY_CACHE_SIZE:-NOT SET}"
echo "DXMT_SSD_CACHE_DIR: ${DXMT_SSD_CACHE_DIR:-NOT SET}"
echo "DXMT_SSD_CACHE_SIZE: ${DXMT_SSD_CACHE_SIZE:-NOT SET}"
echo ""

echo "=== SSD Cache Directory ==="
if [ -d "${DXMT_SSD_CACHE_DIR}" ]; then
    echo "✓ Directory exists: ${DXMT_SSD_CACHE_DIR}"
    echo ""
    echo "Contents:"
    ls -lah "${DXMT_SSD_CACHE_DIR}"
    echo ""
    echo "Total size:"
    du -sh "${DXMT_SSD_CACHE_DIR}"
    echo ""
    echo "File count:"
    find "${DXMT_SSD_CACHE_DIR}" -type f | wc -l
else
    echo "✗ Directory does not exist: ${DXMT_SSD_CACHE_DIR}"
fi
echo ""

echo "=== Configuration Issues ==="
ISSUES=0

if [ -z "${DXMT_STORAGE_TYPE}" ]; then
    echo "✗ DXMT_STORAGE_TYPE is not set (should be 'tiered')"
    ISSUES=$((ISSUES + 1))
elif [ "${DXMT_STORAGE_TYPE}" != "tiered" ]; then
    echo "✗ DXMT_STORAGE_TYPE='${DXMT_STORAGE_TYPE}' (should be 'tiered')"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DXMT_STORAGE_TYPE is set correctly"
fi

if [ -z "${DXMT_OBJECT_STORAGE_PROVIDER}" ]; then
    echo "✗ DXMT_OBJECT_STORAGE_PROVIDER is not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DXMT_OBJECT_STORAGE_PROVIDER is set: ${DXMT_OBJECT_STORAGE_PROVIDER}"
fi

if [ -z "${DXMT_S3_BUCKET}" ]; then
    echo "✗ DXMT_S3_BUCKET is not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DXMT_S3_BUCKET is set: ${DXMT_S3_BUCKET}"
fi

if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "✗ AWS credentials (AWS_*) are not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ AWS credentials (AWS_*) are set"
fi

if [ -z "${DXMT_AWS_ACCESS_KEY_ID}" ] || [ -z "${DXMT_AWS_SECRET_ACCESS_KEY}" ]; then
    echo "✗ DB AWS credentials (DXMT_AWS_*) are not set - these are REQUIRED for VectorDB config!"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DB AWS credentials (DXMT_AWS_*) are set"
fi

echo ""
if [ ${ISSUES} -eq 0 ]; then
    echo "✓ No configuration issues found"
else
    echo "✗ Found ${ISSUES} configuration issue(s)"
    echo ""
    echo "To fix, export the object storage settings before starting the server:"
    echo "  export DXMT_OBJECT_STORAGE_PROVIDER=s3"
    echo "  export DXMT_S3_BUCKET=<your-bucket>"
    echo "  export DXMT_S3_ENDPOINT=<your-endpoint>"
    echo "  export DXMT_AWS_ACCESS_KEY_ID=<your-access-key>"
    echo "  export DXMT_AWS_SECRET_ACCESS_KEY=<your-secret-key>"
fi

echo ""
echo "=== Recommendations ==="
echo "1. Make sure server was started AFTER setting environment variables"
echo "2. Check server logs for any S3/GCS connection errors"
echo "3. Verify that test created enough data to trigger GCS write"
echo "   (tiered storage may keep hot data in memory/SSD)"
echo "4. Try inserting more records to force promotion to object storage"
echo ""
echo "=========================================="
