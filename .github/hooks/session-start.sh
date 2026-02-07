#!/usr/bin/env bash
# Claude Code Session Start Hook for Tritter
# Runs at the beginning of each Claude Code session
# Verifies environment, dependencies, and provides helpful context

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Tritter - Claude Code Session Starting${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "Python: ${GREEN}${PYTHON_VERSION}${NC}"

# Check if in virtual environment
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo -e "Virtual env: ${GREEN}${VIRTUAL_ENV}${NC}"
else
    echo -e "${YELLOW}⚠️  Not in a virtual environment (consider using venv or uv)${NC}"
fi

# Check if uv is installed (recommended)
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1 | awk '{print $2}')
    echo -e "uv: ${GREEN}${UV_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠️  uv not installed (recommended for faster package management)${NC}"
    echo -e "   Install: ${BLUE}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
fi

# Check if dependencies are installed
echo ""
echo -e "${BLUE}Checking dependencies...${NC}"

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "PyTorch: ${GREEN}${TORCH_VERSION}${NC}"

    # Check CUDA availability
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        echo -e "CUDA: ${GREEN}${CUDA_VERSION}${NC}"
        echo -e "GPU: ${GREEN}${CUDA_DEVICE}${NC}"
    else
        echo -e "${YELLOW}⚠️  CUDA not available (CPU-only mode)${NC}"
    fi
else
    echo -e "${RED}❌ PyTorch not installed${NC}"
    echo -e "   Install: ${BLUE}uv pip install torch --index-url https://download.pytorch.org/whl/cu128${NC}"
fi

# Check Tritter installation
echo ""
if python -c "from tritter import TritterConfig" 2>/dev/null; then
    echo -e "Tritter: ${GREEN}Installed${NC}"
else
    echo -e "${YELLOW}⚠️  Tritter not installed or importable${NC}"
    echo -e "   Install: ${BLUE}uv pip install -e '.[dev]'${NC}"
fi

# Detect current branch
echo ""
echo -e "${BLUE}Git status...${NC}"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo -e "Branch: ${GREEN}${CURRENT_BRANCH}${NC}"

# Detect CI strictness level
if [[ -f ".github/scripts/detect-branch-strictness.sh" ]]; then
    eval "$(.github/scripts/detect-branch-strictness.sh 2>/dev/null || echo 'STRICTNESS_NAME=UNKNOWN')"
    echo -e "CI Strictness: ${YELLOW}${STRICTNESS_NAME:-UNKNOWN}${NC}"
fi

# Check for uncommitted changes
UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l)
if [[ $UNCOMMITTED -gt 0 ]]; then
    echo -e "Uncommitted changes: ${YELLOW}${UNCOMMITTED} files${NC}"
else
    echo -e "Working tree: ${GREEN}Clean${NC}"
fi

# Provide helpful shortcuts
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Quick Commands${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Format:       ${BLUE}ruff format .${NC}"
echo -e "Lint:         ${BLUE}ruff check .${NC}"
echo -e "Type check:   ${BLUE}mypy src/tritter${NC}"
echo -e "Test:         ${BLUE}pytest${NC}"
echo -e "Coverage:     ${BLUE}pytest --cov=src/tritter${NC}"
echo -e "CI check:     ${BLUE}.github/scripts/run-ci.sh${NC}"
echo -e "Validate:     ${BLUE}python devtools/validate.py${NC}"
echo ""
echo -e "${BLUE}Model specs:  ${BLUE}python scripts/show_model_specs.py --model 7B${NC}"
echo -e "Feasibility:  ${BLUE}python scripts/rtx5080_feasibility.py${NC}"
echo ""
echo -e "${GREEN}✅ Session ready! Review CLAUDE.md for project context.${NC}"
echo ""
