#!/usr/bin/env bash
#
# install-minimal.sh - Minimal Python-only installation for Tritter
#
# This script assumes Python 3.12+ and CUDA (if needed) are already installed.
# Use this for:
# - CI/CD environments
# - Docker containers
# - Systems where CUDA is pre-installed
# - Quick reinstallation of Python packages
#
# Usage:
#   ./install-minimal.sh [--no-venv] [--no-cuda] [--blackwell]
#
# Options:
#   --no-venv       Don't create virtual environment (use system/existing env)
#   --no-cuda       Install CPU-only PyTorch
#   --blackwell     Install PyTorch nightly for RTX 50-series support

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

fatal() {
    error "$*"
    exit 1
}

# Parse arguments
CREATE_VENV=true
INSTALL_CUDA=true
INSTALL_BLACKWELL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            CREATE_VENV=false
            shift
            ;;
        --no-cuda)
            INSTALL_CUDA=false
            shift
            ;;
        --blackwell)
            INSTALL_BLACKWELL=true
            shift
            ;;
        -h|--help)
            grep '^#' "$0" | grep -v '#!/' | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            fatal "Unknown option: $1 (use --help for usage)"
            ;;
    esac
done

# Check Python version
check_python() {
    info "Checking Python version..."

    PYTHON_CMD=""

    # Check for Python 3.13
    if command -v python3.13 &> /dev/null; then
        PYTHON_CMD="python3.13"
    # Check for Python 3.12
    elif command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    # Check default python3
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

        if [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -ge 12 ]] && [[ "$PYTHON_MINOR" -lt 14 ]]; then
            PYTHON_CMD="python3"
        fi
    fi

    if [[ -z "$PYTHON_CMD" ]]; then
        fatal "Python 3.12 or 3.13 not found. Please install it first."
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
    info "Found Python: $PYTHON_VERSION"

    # Check for venv module
    if [[ "$CREATE_VENV" == "true" ]]; then
        if ! $PYTHON_CMD -m venv --help &> /dev/null; then
            fatal "Python venv module not available. Install python3-venv package."
        fi
    fi

    export PYTHON_CMD
}

# Create virtual environment
create_venv() {
    if [[ "$CREATE_VENV" != "true" ]]; then
        info "Using existing Python environment (--no-venv specified)"
        return
    fi

    VENV_PATH="venv"

    if [[ -d "$VENV_PATH" ]]; then
        info "Virtual environment already exists at $VENV_PATH"
        info "Using existing environment..."
    else
        info "Creating virtual environment at $VENV_PATH..."
        $PYTHON_CMD -m venv "$VENV_PATH"
        success "Virtual environment created"
    fi

    # shellcheck source=/dev/null
    source "$VENV_PATH/bin/activate"
}

# Install PyTorch
install_pytorch() {
    info "Installing PyTorch..."

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    if [[ "$INSTALL_BLACKWELL" == "true" ]]; then
        BLACKWELL_TORCH_VERSION="${TRITTER_BLACKWELL_TORCH_VERSION:-2.11.0.dev20260123+cu128}"
        BLACKWELL_TRITON_VERSION="${TRITTER_BLACKWELL_TRITON_VERSION:-3.6.0+git9844da95}"
        info "Installing pinned PyTorch nightly for RTX 50-series (Blackwell) support: $BLACKWELL_TORCH_VERSION"
        pip install "torch==${BLACKWELL_TORCH_VERSION}" --pre --index-url https://download.pytorch.org/whl/nightly/cu128
        info "Installing pinned Triton for Blackwell: $BLACKWELL_TRITON_VERSION"
        pip install "triton==${BLACKWELL_TRITON_VERSION}" --pre
    elif [[ "$INSTALL_CUDA" == "true" ]]; then
        info "Installing PyTorch with CUDA 12.x support..."
        pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu124
    else
        info "Installing PyTorch (CPU-only)..."
        pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cpu
    fi

    success "PyTorch installed"
}

# Install Triton
install_triton() {
    if [[ "$INSTALL_BLACKWELL" == "true" ]]; then
        info "Triton already installed with PyTorch nightly"
        return
    fi

    info "Installing Triton..."
    pip install triton>=3.0.0
    success "Triton installed"
}

# Install project dependencies
install_project() {
    info "Installing Tritter in editable mode with dev dependencies..."

    # Navigate to project root (assuming script is in scripts/install/)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    cd "$PROJECT_ROOT"

    # Install in editable mode with dev dependencies
    pip install -e ".[dev]"

    success "Tritter installed"
}

# Verify installation
verify_installation() {
    info "Verifying installation..."

    # Test basic imports
    python -c "
import sys
import torch
import triton

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
print(f'Triton: {triton.__version__}')
" || fatal "Basic imports failed"

    # Test Tritter imports
    python -c "from tritter import *; print('Tritter imports OK')" || fatal "Tritter import failed"

    success "Installation verified"
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "      Tritter Installation Complete      "
    echo "=========================================="
    echo ""

    if [[ "$CREATE_VENV" == "true" ]]; then
        echo "Virtual environment: venv/"
        echo "Activate with: source venv/bin/activate"
        echo ""
    fi

    echo "Quick test:"
    echo "  python -c 'from tritter import *; print(\"OK\")'"
    echo ""
    echo "Run tests:"
    echo "  pytest"
    echo ""
    echo "See full installation script for more features:"
    echo "  ./install-dev-debian.sh --help"
    echo ""
}

# Main execution
main() {
    info "Starting minimal Tritter installation..."
    echo ""

    check_python
    create_venv
    install_pytorch
    install_triton
    install_project
    verify_installation

    echo ""
    success "Installation complete!"
    echo ""

    print_summary
}

# Run main function
main
