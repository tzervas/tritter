#!/usr/bin/env bash
# Security Scanner Wrapper - Proper failure handling
# Only passes when: scanner unavailable OR no secrets found
# Fails when: secrets detected (regardless of other errors)

set -euo pipefail

# Try to run the scanner
if ! python -m tritter.curation.secrets scan . 2>&1; then
    exit_code=$?

    # Check if failure was due to module not found (scanner not installed)
    if [[ $exit_code -eq 127 ]] || python -c "import tritter.curation.secrets" 2>&1 | grep -q "ModuleNotFoundError"; then
        echo "ℹ️  Security scanner not installed (acceptable for development)"
        exit 0
    else
        # Scanner is installed but found secrets or had other error
        echo "❌ Security scan failed - secrets may have been detected"
        exit 1
    fi
fi

# Scanner ran successfully and found no secrets
echo "✅ No secrets detected"
exit 0
