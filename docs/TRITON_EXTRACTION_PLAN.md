# Triton Kernel Dataset Extraction Plan

**Issue**: #58
**Purpose**: Executable plan for extracting Triton kernels from permissively-licensed repositories
**Status**: Ready for implementation

---

## Phase 1: Repository Setup (Execute Once)

### 1.1 Create Workspace

```bash
#!/bin/bash
set -e

WORKSPACE="${HOME}/triton_dataset_workspace"
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

echo "Workspace created at: ${WORKSPACE}"
```

### 1.2 Clone Repositories with Sparse Checkout

**Script: `01_clone_repositories.sh`**

```bash
#!/bin/bash
set -e

WORKSPACE="${1:-.}"
cd "${WORKSPACE}"

echo "=== Phase 1: Cloning Repositories ==="

# Repository 1: Triton (full clone, small)
echo "1/5 Cloning triton-lang/triton..."
git clone --depth 1 https://github.com/triton-lang/triton.git triton
echo "✓ Triton cloned (Size: ~500MB with history)"

# Repository 2: PyTorch (sparse checkout)
echo "2/5 Cloning pytorch/pytorch (sparse)..."
git clone --filter=blob:none --sparse https://github.com/pytorch/pytorch.git pytorch
cd pytorch
git sparse-checkout set --cone torch/_inductor
git checkout main
cd ..
echo "✓ PyTorch cloned with sparse checkout"

# Repository 3: FlashAttention (full clone, small)
echo "3/5 Cloning Dao-AILab/flash-attention..."
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git flash-attention
echo "✓ FlashAttention cloned"

# Repository 4: xFormers (sparse checkout)
echo "4/5 Cloning facebookresearch/xformers (sparse)..."
git clone --filter=blob:none --sparse https://github.com/facebookresearch/xformers.git xformers
cd xformers
git sparse-checkout set --cone xformers/ops
git checkout main
cd ..
echo "✓ xFormers cloned with sparse checkout"

# Repository 5: JAX (sparse checkout)
echo "5/5 Cloning jax-ml/jax (sparse)..."
git clone --filter=blob:none --sparse https://github.com/jax-ml/jax.git jax
cd jax
git sparse-checkout set --cone jax/experimental/pallas
git checkout main
cd ..
echo "✓ JAX cloned with sparse checkout"

echo ""
echo "=== Clone Complete ==="
du -sh . 2>/dev/null || echo "(Size info unavailable)"
```

**Usage**:
```bash
chmod +x 01_clone_repositories.sh
./01_clone_repositories.sh
```

---

## Phase 2: Extract Triton Kernels

### 2.1 Identify Kernel Files

**Script: `02_identify_kernels.sh`**

```bash
#!/bin/bash
set -e

WORKSPACE="${1:-.}"
cd "${WORKSPACE}"

echo "=== Phase 2: Identifying Triton Kernels ==="
mkdir -p extraction_logs

# Helper function to search for Triton kernels
find_triton_kernels() {
    local repo_path="$1"
    local output_file="$2"

    if [ -d "${repo_path}" ]; then
        echo "Searching ${repo_path}..."
        find "${repo_path}" -name "*.py" -type f -exec grep -l "@triton.jit" {} \; | sort > "${output_file}"
        echo "  Found: $(wc -l < ${output_file}) files"
    else
        echo "WARNING: ${repo_path} not found"
        touch "${output_file}"
    fi
}

# Search each repository
find_triton_kernels "triton/python/tutorials" "extraction_logs/triton_kernels.txt"
find_triton_kernels "pytorch/torch/_inductor" "extraction_logs/pytorch_kernels.txt"
find_triton_kernels "flash-attention" "extraction_logs/flashattn_kernels.txt"
find_triton_kernels "xformers/xformers/ops" "extraction_logs/xformers_kernels.txt"
find_triton_kernels "jax/jax/experimental/pallas" "extraction_logs/jax_kernels.txt"

# Create combined list
echo ""
echo "=== Summary ==="
cat extraction_logs/*_kernels.txt | sort -u > extraction_logs/all_triton_kernels.txt
echo "Total unique kernel files: $(wc -l < extraction_logs/all_triton_kernels.txt)"

echo ""
echo "Kernel file lists:"
ls -lh extraction_logs/*_kernels.txt
```

**Usage**:
```bash
chmod +x 02_identify_kernels.sh
./02_identify_kernels.sh
```

### 2.2 Generate Statistics

**Script: `03_statistics.sh`**

```bash
#!/bin/bash
set -e

WORKSPACE="${1:-.}"
cd "${WORKSPACE}"

echo "=== Phase 3: Generate Statistics ==="
mkdir -p extraction_logs/statistics

# Repository 1: Triton
echo "Triton Repository:"
find triton/python/tutorials -name "*.py" -type f | wc -l | xargs echo "  Python files:"
find triton/python/tutorials -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print "  Total lines:", $1}'

# Repository 2: PyTorch
echo ""
echo "PyTorch Repository:"
find pytorch/torch/_inductor -name "*triton*.py" -type f | wc -l | xargs echo "  Triton files:"
find pytorch/torch/_inductor -name "*triton*.py" -type f -exec wc -l {} + | tail -1 | awk '{print "  Total lines:", $1}'

# Repository 3: FlashAttention
echo ""
echo "FlashAttention Repository:"
find flash-attention -name "*triton*.py" -type f | wc -l | xargs echo "  Triton files:"
find flash-attention -name "*triton*.py" -type f -exec wc -l {} + | tail -1 | awk '{print "  Total lines:", $1}'

# Repository 4: xFormers
echo ""
echo "xFormers Repository:"
find xformers/xformers/ops -path "*_triton*" -name "*.py" -type f | wc -l | xargs echo "  Triton files:"
find xformers/xformers/ops -path "*_triton*" -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print "  Total lines:", $1}'

# Repository 5: JAX
echo ""
echo "JAX Repository:"
find jax/jax/experimental/pallas/triton -name "*.py" -type f 2>/dev/null | wc -l | xargs echo "  Triton files:"
find jax/jax/experimental/pallas/triton -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print "  Total lines:", $1}'

# Total
echo ""
echo "=== TOTAL ==="
find . -path "*/extraction_logs" -prune -o -name "*triton*.py" -type f -print | grep -v extraction_logs | xargs wc -l | tail -1
```

**Usage**:
```bash
chmod +x 03_statistics.sh
./03_statistics.sh
```

---

## Phase 3: Extract and Organize Kernels

### 3.1 Copy Kernels to Organized Structure

**Script: `04_organize_kernels.sh`**

```bash
#!/bin/bash
set -e

WORKSPACE="${1:-.}"
EXTRACT_DIR="${WORKSPACE}/triton_kernels_organized"

echo "=== Phase 4: Organizing Kernels ==="

# Create organized directory structure
mkdir -p "${EXTRACT_DIR}"/{tutorial,inductor,flashattention,xformers,jax,utils}

echo "Created output directory: ${EXTRACT_DIR}"

# Copy Triton tutorials
echo ""
echo "Copying Triton tutorials..."
find "${WORKSPACE}/triton/python/tutorials" -maxdepth 1 -name "*.py" -type f -exec cp {} "${EXTRACT_DIR}/tutorial/" \;
echo "✓ $(ls -1 ${EXTRACT_DIR}/tutorial | wc -l) files copied"

# Copy PyTorch Inductor kernels
echo ""
echo "Copying PyTorch Inductor kernels..."
find "${WORKSPACE}/pytorch/torch/_inductor" -name "*triton*.py" -type f | while read f; do
    rel_path="${f#${WORKSPACE}/pytorch/}"
    mkdir -p "${EXTRACT_DIR}/inductor/$(dirname ${rel_path})"
    cp "$f" "${EXTRACT_DIR}/inductor/${rel_path}"
done
echo "✓ $(find ${EXTRACT_DIR}/inductor -name "*.py" | wc -l) files copied"

# Copy FlashAttention Triton kernels
echo ""
echo "Copying FlashAttention kernels..."
find "${WORKSPACE}/flash-attention" -name "*triton*.py" -type f -exec cp {} "${EXTRACT_DIR}/flashattention/" \;
echo "✓ $(ls -1 ${EXTRACT_DIR}/flashattention 2>/dev/null | wc -l) files copied"

# Copy xFormers Triton kernels
echo ""
echo "Copying xFormers kernels..."
find "${WORKSPACE}/xformers/xformers/ops" -path "*_triton*" -name "*.py" -type f | while read f; do
    rel_path="${f#${WORKSPACE}/xformers/}"
    mkdir -p "${EXTRACT_DIR}/xformers/$(dirname ${rel_path})"
    cp "$f" "${EXTRACT_DIR}/xformers/${rel_path}"
done
echo "✓ $(find ${EXTRACT_DIR}/xformers -name "*.py" 2>/dev/null | wc -l) files copied"

# Copy JAX Pallas Triton kernels
echo ""
echo "Copying JAX Pallas kernels..."
find "${WORKSPACE}/jax/jax/experimental/pallas" -path "*triton*" -name "*.py" -type f | while read f; do
    rel_path="${f#${WORKSPACE}/jax/}"
    mkdir -p "${EXTRACT_DIR}/jax/$(dirname ${rel_path})"
    cp "$f" "${EXTRACT_DIR}/jax/${rel_path}"
done
echo "✓ $(find ${EXTRACT_DIR}/jax -name "*.py" 2>/dev/null | wc -l) files copied"

echo ""
echo "=== Extraction Complete ==="
echo "Total Triton kernel files: $(find ${EXTRACT_DIR} -name "*.py" | wc -l)"
du -sh "${EXTRACT_DIR}"
```

**Usage**:
```bash
chmod +x 04_organize_kernels.sh
./04_organize_kernels.sh
```

---

## Phase 4: Quality Validation

### 4.1 License Verification

**Script: `05_verify_licenses.py`**

```python
#!/usr/bin/env python3
"""
Verify that all extracted Triton kernels have permissive licenses.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

WORKSPACE = os.environ.get("WORKSPACE", ".")
EXTRACT_DIR = os.path.join(WORKSPACE, "triton_kernels_organized")

PERMISSIVE_LICENSES = [
    "MIT",
    "BSD",
    "BSD-3",
    "Apache-2.0",
    "Apache 2.0",
    "ISC",
]

def extract_license_header(filepath):
    """Extract first 20 lines to check for license."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [next(f).strip() for _ in range(20) if f.readline()]
    return '\n'.join(lines)

def check_permissive_license(header):
    """Check if header contains permissive license."""
    for license_name in PERMISSIVE_LICENSES:
        if license_name.upper() in header.upper():
            return True, license_name
    return False, None

def main():
    print("=== License Verification ===")

    results = defaultdict(lambda: {"valid": [], "unknown": [], "missing": []})

    for root, dirs, files in os.walk(EXTRACT_DIR):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, EXTRACT_DIR)
                category = rel_path.split(os.sep)[0]

                header = extract_license_header(filepath)
                has_license, license_type = check_permissive_license(header)

                if has_license:
                    results[category]["valid"].append((rel_path, license_type))
                else:
                    # Check if it's part of a licensed repository
                    if any(pattern in filepath for pattern in ["pytorch", "triton", "flash", "xformers", "jax"]):
                        results[category]["unknown"].append(rel_path)
                    else:
                        results[category]["missing"].append(rel_path)

    # Report
    print("\nResults by category:")
    for category in sorted(results.keys()):
        data = results[category]
        print(f"\n{category.upper()}:")
        print(f"  ✓ Valid license headers: {len(data['valid'])}")
        if len(data['valid']) > 0:
            print(f"    Licenses: {set(lic for _, lic in data['valid'])}")
        print(f"  ? Unknown (repository licensed): {len(data['unknown'])}")
        print(f"  ✗ Missing license info: {len(data['missing'])}")

    # Total
    total_valid = sum(len(data['valid']) + len(data['unknown']) for data in results.values())
    total_files = sum(len(data['valid']) + len(data['unknown']) + len(data['missing']) for data in results.values())
    print(f"\n=== TOTAL ===")
    print(f"Total files: {total_files}")
    print(f"License coverage: {total_valid}/{total_files} ({100*total_valid//total_files}%)")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
chmod +x 05_verify_licenses.py
WORKSPACE=. ./05_verify_licenses.py
```

### 4.2 Triton Quality Checks

**Script: `06_quality_checks.py`**

```python
#!/usr/bin/env python3
"""
Validate quality of extracted Triton kernels.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

WORKSPACE = os.environ.get("WORKSPACE", ".")
EXTRACT_DIR = os.path.join(WORKSPACE, "triton_kernels_organized")

class TritonKernelValidator:
    def __init__(self):
        self.results = defaultdict(lambda: {
            "has_triton_jit": 0,
            "has_bounds_masking": 0,
            "has_program_id": 0,
            "has_block_size": 0,
            "has_load_store": 0,
            "has_docstring": 0,
            "total_files": 0,
        })

    def validate_kernel(self, filepath):
        """Validate a single Triton kernel file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        category = os.path.relpath(filepath, EXTRACT_DIR).split(os.sep)[0]
        self.results[category]["total_files"] += 1

        checks = {
            "has_triton_jit": "@triton.jit" in content,
            "has_bounds_masking": ("tl.where" in content or
                                 "mask=" in content or
                                 "masked_load" in content),
            "has_program_id": "tl.program_id" in content,
            "has_block_size": any(pattern in content for pattern in
                                ["BLOCK_SIZE", "block_", "BLOCK_"]),
            "has_load_store": ("tl.load" in content or "tl.store" in content),
            "has_docstring": ('"""' in content or "'''" in content or
                            re.search(r'def \w+.*:\s+"""', content)),
        }

        for check, result in checks.items():
            if result:
                self.results[category][check] += 1

        return checks

    def run(self):
        """Run validation on all kernel files."""
        print("=== Triton Quality Validation ===\n")

        kernel_count = 0
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for filename in files:
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    self.validate_kernel(filepath)
                    kernel_count += 1

        print(f"Total kernel files analyzed: {kernel_count}\n")

        # Report by category
        for category in sorted(self.results.keys()):
            data = self.results[category]
            total = data["total_files"]
            print(f"{category.upper()} ({total} files):")

            checks = [
                ("@triton.jit decorator", "has_triton_jit"),
                ("Bounds masking", "has_bounds_masking"),
                ("Program ID usage", "has_program_id"),
                ("Block size config", "has_block_size"),
                ("Load/Store ops", "has_load_store"),
                ("Documentation", "has_docstring"),
            ]

            for check_name, check_key in checks:
                count = data[check_key]
                pct = (100 * count // total) if total > 0 else 0
                status = "✓" if pct > 80 else "~" if pct > 50 else "✗"
                print(f"  {status} {check_name}: {count}/{total} ({pct}%)")

            print()

if __name__ == "__main__":
    validator = TritonKernelValidator()
    validator.run()
```

**Usage**:
```bash
chmod +x 06_quality_checks.py
WORKSPACE=. ./06_quality_checks.py
```

---

## Phase 5: Generate Dataset Manifest

### 5.1 Create Manifest File

**Script: `07_create_manifest.py`**

```python
#!/usr/bin/env python3
"""
Create a manifest of extracted Triton kernels for tracking.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

WORKSPACE = os.environ.get("WORKSPACE", ".")
EXTRACT_DIR = os.path.join(WORKSPACE, "triton_kernels_organized")
MANIFEST_FILE = os.path.join(EXTRACT_DIR, "MANIFEST.json")

def compute_file_hash(filepath):
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_file_stats(filepath):
    """Get file statistics."""
    stat = os.stat(filepath)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = len(f.readlines())
        f.seek(0)
        has_triton_jit = "@triton.jit" in f.read()

    return {
        "path": os.path.relpath(filepath, EXTRACT_DIR),
        "size_bytes": stat.st_size,
        "lines": lines,
        "has_triton_jit": has_triton_jit,
        "hash_sha256": compute_file_hash(filepath),
    }

def main():
    print("=== Creating Dataset Manifest ===\n")

    manifest = {
        "created": datetime.now().isoformat(),
        "extraction_dir": EXTRACT_DIR,
        "categories": {},
    }

    total_files = 0
    total_size = 0
    total_lines = 0

    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "MANIFEST.json" in files:
            files.remove("MANIFEST.json")

        for filename in sorted(files):
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                category = os.path.relpath(filepath, EXTRACT_DIR).split(os.sep)[0]

                if category not in manifest["categories"]:
                    manifest["categories"][category] = {
                        "files": [],
                        "total_files": 0,
                        "total_size_bytes": 0,
                        "total_lines": 0,
                    }

                stats = get_file_stats(filepath)
                manifest["categories"][category]["files"].append(stats)
                manifest["categories"][category]["total_files"] += 1
                manifest["categories"][category]["total_size_bytes"] += stats["size_bytes"]
                manifest["categories"][category]["total_lines"] += stats["lines"]

                total_files += 1
                total_size += stats["size_bytes"]
                total_lines += stats["lines"]

    # Add totals
    manifest["total_files"] = total_files
    manifest["total_size_bytes"] = total_size
    manifest["total_lines"] = total_lines

    # Write manifest
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest created: {MANIFEST_FILE}")
    print(f"\n=== Dataset Summary ===")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")
    print(f"Total lines: {total_lines:,}")

    print(f"\nBy category:")
    for category in sorted(manifest["categories"].keys()):
        data = manifest["categories"][category]
        print(f"  {category}:")
        print(f"    Files: {data['total_files']}")
        print(f"    Size: {data['total_size_bytes'] / (1024):.1f} KB")
        print(f"    Lines: {data['total_lines']:,}")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
chmod +x 07_create_manifest.py
WORKSPACE=. ./07_create_manifest.py
```

---

## Master Execution Script

**Script: `run_extraction.sh`** (Orchestrates all phases)

```bash
#!/bin/bash
set -e

WORKSPACE="${1:-.}"
cd "${WORKSPACE}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Triton Kernel Dataset Extraction - Complete Pipeline        ║"
echo "║   Issue #58 - GPU Kernel Curation                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 1: Clone repositories
echo "PHASE 1: Repository Cloning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./01_clone_repositories.sh "${WORKSPACE}" || exit 1
echo ""

# Phase 2: Identify kernels
echo "PHASE 2: Identifying Triton Kernels"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./02_identify_kernels.sh "${WORKSPACE}" || exit 1
echo ""

# Phase 3: Statistics
echo "PHASE 3: Generating Statistics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./03_statistics.sh "${WORKSPACE}" || exit 1
echo ""

# Phase 4: Organize
echo "PHASE 4: Organizing Kernels"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./04_organize_kernels.sh "${WORKSPACE}" || exit 1
echo ""

# Phase 5: Verify licenses
echo "PHASE 5: License Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
WORKSPACE="${WORKSPACE}" python3 ./05_verify_licenses.py || exit 1
echo ""

# Phase 6: Quality checks
echo "PHASE 6: Quality Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
WORKSPACE="${WORKSPACE}" python3 ./06_quality_checks.py || exit 1
echo ""

# Phase 7: Manifest
echo "PHASE 7: Creating Manifest"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
WORKSPACE="${WORKSPACE}" python3 ./07_create_manifest.py || exit 1
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   EXTRACTION COMPLETE                          ║"
echo "║                                                                ║"
echo "║  Next steps:                                                   ║"
echo "║  1. Review MANIFEST.json for dataset contents                 ║"
echo "║  2. Tokenize kernels for training                             ║"
echo "║  3. Apply 5-10x upsample factor relative to Python            ║"
echo "║  4. Implement deduplication if needed                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
```

**Usage**:
```bash
chmod +x run_extraction.sh *.sh *.py
./run_extraction.sh /path/to/workspace
```

---

## Quick Start (One Command)

```bash
# Download and execute extraction in one go
mkdir -p triton_extraction
cd triton_extraction

# Copy scripts to directory
# (scripts should be in current directory)

# Run complete pipeline
./run_extraction.sh $(pwd)

# Results in: triton_extraction/triton_kernels_organized/
```

---

## Expected Output Structure

```
triton_extraction/
├── triton/                          # Cloned repositories
├── pytorch/
├── flash-attention/
├── xformers/
├── jax/
├── triton_kernels_organized/        # Extracted kernels
│   ├── tutorial/                    # Triton tutorials
│   ├── inductor/                    # PyTorch inductor
│   ├── flashattention/              # FlashAttention
│   ├── xformers/                    # xFormers
│   ├── jax/                         # JAX Pallas
│   └── MANIFEST.json                # Dataset manifest
└── extraction_logs/                 # Processing logs
    ├── *_kernels.txt
    └── statistics/
```

---

## Next Steps After Extraction

1. **Tokenization**: Convert Python files to tokens using model tokenizer
2. **Deduplication**: Apply MinHash to remove duplicates
3. **Validation**: Run against quality gates (SPEC-007)
4. **Upsampling**: Apply 5-10x weight multiplier for Triton vs Python
5. **Integration**: Combine with other training data sources

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Git sparse checkout fails | Ensure Git 2.25+: `git --version` |
| Large repository size | Use `--filter=blob:none` flag (already in scripts) |
| Permission errors | Run with appropriate permissions or use `sudo` if needed |
| Missing Python packages | Install: `pip install pathspec click` |

---

## References

- Full documentation: [`TRITON_KERNEL_CURATION.md`](./TRITON_KERNEL_CURATION.md)
- Quick reference: [`TRITON_QUICK_REFERENCE.md`](./TRITON_QUICK_REFERENCE.md)
- Dataset quality gates: [`specs/SPEC-007-dataset-quality-gates.md`](../specs/SPEC-007-dataset-quality-gates.md)
- Related: [`clean-datasets.md`](./clean-datasets.md#triton-gpu-kernel-datasets)
