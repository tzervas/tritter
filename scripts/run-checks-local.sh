#!/usr/bin/env bash
# Local Quality Checks - Exact CI Parity
# Run this before pushing to verify all checks pass
# This is the SAME script that CI uses, ensuring 100% parity
#
# Usage:
#   bash scripts/run-checks-local.sh              # Auto-detect strictness from branch
#   STRICTNESS_OVERRIDE=0 bash scripts/run-checks-local.sh  # Force FEATURE level
#   STRICTNESS_OVERRIDE=2 bash scripts/run-checks-local.sh  # Force RELEASE level
#   STRICTNESS_OVERRIDE=3 bash scripts/run-checks-local.sh  # Force PRODUCTION level
#
# Strictness levels:
#   0 = FEATURE     (warnings only, 50% coverage)
#   1 = DEVELOPMENT (must format, 70% coverage)
#   2 = RELEASE     (strict linting, 80% coverage)
#   3 = PRODUCTION  (zero tolerance, 85% coverage)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║  Tritter Local Quality Checks                        ║${NC}"
echo -e "${BOLD}${BLUE}║  (Exact parity with GitHub Actions CI)               ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in project root
if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo -e "${RED}Error: Must run from project root${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

# Pre-flight checks
echo -e "${CYAN}Pre-flight checks...${NC}"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "  Python: ${GREEN}${PYTHON_VERSION}${NC}"

# Check if dependencies installed
if ! python -c "import torch" 2>/dev/null; then
    echo -e "  ${YELLOW}⚠️  PyTorch not installed - some checks will be skipped${NC}"
    echo -e "     Install: ${CYAN}uv pip install -e '.[dev,training,curation]'${NC}"
else
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "  PyTorch: ${GREEN}${TORCH_VERSION}${NC}"
fi

if ! python -c "from tritter import TritterConfig" 2>/dev/null; then
    echo -e "  ${YELLOW}⚠️  Tritter not installed${NC}"
    echo -e "     Install: ${CYAN}uv pip install -e '.[dev]'${NC}"
else
    echo -e "  Tritter: ${GREEN}Installed${NC}"
fi

# Show override if set
if [[ -n "${STRICTNESS_OVERRIDE:-}" ]]; then
    echo -e "  Override: ${YELLOW}Level ${STRICTNESS_OVERRIDE}${NC}"
fi

echo ""
echo -e "${BOLD}${CYAN}Running CI checks (this is EXACTLY what GitHub Actions runs)...${NC}"
echo ""

# Run the exact same script CI uses
if [[ -f ".github/scripts/run-ci.sh" ]]; then
    bash .github/scripts/run-ci.sh
    EXIT_CODE=$?
else
    echo -e "${RED}Error: CI script not found at .github/scripts/run-ci.sh${NC}"
    exit 1
fi

echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${BOLD}${GREEN}  ✅ Local checks passed!${NC}"
    echo -e ""
    echo -e "  Safe to push. CI (when manually triggered) will produce"
    echo -e "  identical results since it uses the same script."
else
    echo -e "${BOLD}${YELLOW}  ⚠️  Some checks failed or warned${NC}"
    echo -e ""
    echo -e "  Fix issues before pushing to avoid CI failures."
    echo -e "  Note: Warnings are OK on feature branches."
fi
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"

exit $EXIT_CODE
