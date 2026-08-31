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
echo "DB_STORAGE_TYPE: ${DB_STORAGE_TYPE:-NOT SET}"
echo "DB_OBJECT_STORAGE_PROVIDER: ${DB_OBJECT_STORAGE_PROVIDER:-NOT SET}"
echo "DB_S3_BUCKET: ${DB_S3_BUCKET:-NOT SET}"
echo "DB_S3_ENDPOINT: ${DB_S3_ENDPOINT:-NOT SET}"
echo "DB_S3_REGION: ${DB_S3_REGION:-NOT SET}"
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:+SET (${#AWS_ACCESS_KEY_ID} chars)}"
echo "AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:+SET (${#AWS_SECRET_ACCESS_KEY} chars)}"
echo "DB_AWS_ACCESS_KEY_ID: ${DB_AWS_ACCESS_KEY_ID:+SET (${#DB_AWS_ACCESS_KEY_ID} chars)}"
echo "DB_AWS_SECRET_ACCESS_KEY: ${DB_AWS_SECRET_ACCESS_KEY:+SET (${#DB_AWS_SECRET_ACCESS_KEY} chars)}"
echo "DB_MEMORY_CACHE_SIZE: ${DB_MEMORY_CACHE_SIZE:-NOT SET}"
echo "DB_SSD_CACHE_DIR: ${DB_SSD_CACHE_DIR:-NOT SET}"
echo "DB_SSD_CACHE_SIZE: ${DB_SSD_CACHE_SIZE:-NOT SET}"
echo ""

echo "=== SSD Cache Directory ==="
if [ -d "${DB_SSD_CACHE_DIR}" ]; then
    echo "✓ Directory exists: ${DB_SSD_CACHE_DIR}"
    echo ""
    echo "Contents:"
    ls -lah "${DB_SSD_CACHE_DIR}"
    echo ""
    echo "Total size:"
    du -sh "${DB_SSD_CACHE_DIR}"
    echo ""
    echo "File count:"
    find "${DB_SSD_CACHE_DIR}" -type f | wc -l
else
    echo "✗ Directory does not exist: ${DB_SSD_CACHE_DIR}"
fi
echo ""

echo "=== Configuration Issues ==="
ISSUES=0

if [ -z "${DB_STORAGE_TYPE}" ]; then
    echo "✗ DB_STORAGE_TYPE is not set (should be 'tiered')"
    ISSUES=$((ISSUES + 1))
elif [ "${DB_STORAGE_TYPE}" != "tiered" ]; then
    echo "✗ DB_STORAGE_TYPE='${DB_STORAGE_TYPE}' (should be 'tiered')"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DB_STORAGE_TYPE is set correctly"
fi

if [ -z "${DB_OBJECT_STORAGE_PROVIDER}" ]; then
    echo "✗ DB_OBJECT_STORAGE_PROVIDER is not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DB_OBJECT_STORAGE_PROVIDER is set: ${DB_OBJECT_STORAGE_PROVIDER}"
fi

if [ -z "${DB_S3_BUCKET}" ]; then
    echo "✗ DB_S3_BUCKET is not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DB_S3_BUCKET is set: ${DB_S3_BUCKET}"
fi

if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "✗ AWS credentials (AWS_*) are not set"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ AWS credentials (AWS_*) are set"
fi

if [ -z "${DB_AWS_ACCESS_KEY_ID}" ] || [ -z "${DB_AWS_SECRET_ACCESS_KEY}" ]; then
    echo "✗ DB AWS credentials (DB_AWS_*) are not set - these are REQUIRED for VectorDB config!"
    ISSUES=$((ISSUES + 1))
else
    echo "✓ DB AWS credentials (DB_AWS_*) are set"
fi

echo ""
if [ ${ISSUES} -eq 0 ]; then
    echo "✓ No configuration issues found"
else
    echo "✗ Found ${ISSUES} configuration issue(s)"
    echo ""
    echo "To fix, export the object storage settings before starting the server:"
    echo "  export DB_OBJECT_STORAGE_PROVIDER=s3"
    echo "  export DB_S3_BUCKET=<your-bucket>"
    echo "  export DB_S3_ENDPOINT=<your-endpoint>"
    echo "  export DB_AWS_ACCESS_KEY_ID=<your-access-key>"
    echo "  export DB_AWS_SECRET_ACCESS_KEY=<your-secret-key>"
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
