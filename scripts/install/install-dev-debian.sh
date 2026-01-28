#!/usr/bin/env bash
#
# install-dev-debian.sh - Full development environment setup for Tritter on Debian/Ubuntu
#
# This script sets up everything needed for Tritter development:
# - System dependencies (build tools, Python dev headers)
# - Python 3.12+ (from deadsnakes PPA if needed)
# - CUDA toolkit 12.x (with driver detection)
# - Virtual environment with all dependencies
# - Verification tests
#
# Usage:
#   ./install-dev-debian.sh [--no-cuda] [--venv-path PATH] [--blackwell]
#
# Options:
#   --no-cuda       Skip CUDA installation (CPU-only development)
#   --venv-path     Custom virtual environment path (default: ./venv)
#   --blackwell     Install PyTorch nightly for RTX 50-series support
#   --skip-venv     Don't create virtual environment (use system/existing env)

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
INSTALL_CUDA=true
VENV_PATH="venv"
INSTALL_BLACKWELL=false
SKIP_VENV=false
BLACKWELL_TORCH_VERSION="${TRITTER_BLACKWELL_TORCH_VERSION:-2.11.0.dev20260123+cu128}"
BLACKWELL_TRITON_VERSION="${TRITTER_BLACKWELL_TRITON_VERSION:-3.6.0+git9844da95}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cuda)
            INSTALL_CUDA=false
            shift
            ;;
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --blackwell)
            INSTALL_BLACKWELL=true
            shift
            ;;
        --skip-venv)
            SKIP_VENV=true
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

# Check for Debian/Ubuntu
check_os() {
    info "Checking operating system..."

    if [[ ! -f /etc/os-release ]]; then
        fatal "Cannot detect OS (missing /etc/os-release)"
    fi

    . /etc/os-release

    case "$ID" in
        debian|ubuntu|linuxmint|pop)
            info "Detected $PRETTY_NAME"
            ;;
        *)
            error "This script is designed for Debian/Ubuntu-based systems."
            error "Detected: $PRETTY_NAME"
            error ""
            error "For other distributions, please install dependencies manually:"
            error "  - Python 3.12 or 3.13"
            error "  - CUDA 12.x (optional, for GPU support)"
            error "  - build-essential, git, python3-dev, python3-venv"
            error ""
            error "Then use: ./install-minimal.sh"
            exit 1
            ;;
    esac
}

# Check sudo availability
check_sudo() {
    if ! command -v sudo &> /dev/null; then
        fatal "sudo is not available. Please install sudo or run with root privileges."
    fi

    if ! sudo -n true 2>/dev/null; then
        warn "This script requires sudo privileges for system package installation."
        info "You will be prompted for your password."
        sudo -v || fatal "Failed to obtain sudo privileges"
    fi
}

# Install system dependencies
install_system_deps() {
    info "Installing system dependencies..."

    # Update package lists
    sudo apt-get update

    # Install build essentials and development headers
    sudo apt-get install -y \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        gnupg \
        software-properties-common

    success "System dependencies installed"
}

# Install Python 3.12+
install_python() {
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

    if [[ -n "$PYTHON_CMD" ]]; then
        PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
        info "Found compatible Python: $PYTHON_VERSION"
    else
        warn "Python 3.12+ not found, installing from deadsnakes PPA..."

        # Add deadsnakes PPA
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update

        # Install Python 3.12 and dependencies
        sudo apt-get install -y \
            python3.12 \
            python3.12-dev \
            python3.12-venv \
            python3.12-distutils

        PYTHON_CMD="python3.12"
        success "Python 3.12 installed"
    fi

    # Export for later use
    export PYTHON_CMD
}

# Detect and check NVIDIA drivers
check_nvidia_drivers() {
    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found. No NVIDIA drivers detected."
        return 1
    fi

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    info "NVIDIA driver version: $DRIVER_VERSION"

    # Check if driver supports CUDA 12.x (requires >= 525.60.13)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
    if [[ "$DRIVER_MAJOR" -lt 525 ]]; then
        warn "NVIDIA driver $DRIVER_VERSION may not support CUDA 12.x"
        warn "Recommended: driver >= 525.60.13"
        return 1
    fi

    return 0
}

# Install CUDA toolkit
install_cuda() {
    if [[ "$INSTALL_CUDA" != "true" ]]; then
        info "Skipping CUDA installation (--no-cuda specified)"
        return
    fi

    info "Checking CUDA installation..."

    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)

        if [[ "$CUDA_MAJOR" -eq 12 ]]; then
            info "CUDA $CUDA_VERSION already installed"
            return
        else
            warn "Found CUDA $CUDA_VERSION, but CUDA 12.x is recommended"
        fi
    fi

    # Check NVIDIA drivers
    if ! check_nvidia_drivers; then
        warn "NVIDIA drivers not found or incompatible."
        warn "Please install NVIDIA drivers >= 525.60.13 before installing CUDA."
        warn ""
        warn "Continuing without CUDA installation..."
        warn "You can run this script again after installing drivers."
        INSTALL_CUDA=false
        return
    fi

    info "Installing CUDA 12.x..."

    # Detect Ubuntu/Debian version for CUDA repo
    . /etc/os-release

    case "$ID" in
        ubuntu)
            CUDA_DISTRO="ubuntu${VERSION_ID//./}"
            ;;
        debian)
            CUDA_DISTRO="debian${VERSION_ID}"
            ;;
        *)
            CUDA_DISTRO="ubuntu2204"  # Default fallback
            ;;
    esac

    # Add NVIDIA CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    sudo apt-get update

    # Install CUDA toolkit (without replacing drivers)
    sudo apt-get install -y cuda-toolkit-12-6

    # Add CUDA to PATH
    if ! grep -q "cuda-12" ~/.bashrc; then
        echo '' >> ~/.bashrc
        echo '# CUDA 12.x' >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda-12/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi

    export PATH=/usr/local/cuda-12/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

    success "CUDA toolkit installed"
}

# Create virtual environment
create_venv() {
    if [[ "$SKIP_VENV" == "true" ]]; then
        info "Skipping virtual environment creation (--skip-venv specified)"
        return
    fi

    info "Creating virtual environment at $VENV_PATH..."

    if [[ -d "$VENV_PATH" ]]; then
        warn "Virtual environment already exists at $VENV_PATH"
        read -p "Remove and recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
        else
            info "Using existing virtual environment"
            return
        fi
    fi

    $PYTHON_CMD -m venv "$VENV_PATH"

    success "Virtual environment created"
}

# Install PyTorch
install_pytorch() {
    info "Installing PyTorch..."

    # Activate venv if not skipped
    if [[ "$SKIP_VENV" != "true" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_PATH/bin/activate"
    fi

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    if [[ "$INSTALL_BLACKWELL" == "true" ]]; then
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
    info "Installing Triton..."

    # Activate venv if not skipped
    if [[ "$SKIP_VENV" != "true" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_PATH/bin/activate"
    fi

    if [[ "$INSTALL_BLACKWELL" != "true" ]]; then
        # Regular Triton installation (already installed with PyTorch, but ensure version)
        pip install triton>=3.0.0
    fi

    success "Triton installed"
}

# Install project dependencies
install_project() {
    info "Installing Tritter in editable mode with dev dependencies..."

    # Activate venv if not skipped
    if [[ "$SKIP_VENV" != "true" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_PATH/bin/activate"
    fi

    # Navigate to project root (assuming script is in scripts/install/)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    cd "$PROJECT_ROOT"

    # Install in editable mode with dev dependencies
    pip install -e ".[dev]"

    success "Tritter installed"
}

# Run verification tests
verify_installation() {
    info "Running verification tests..."

    # Activate venv if not skipped
    if [[ "$SKIP_VENV" != "true" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_PATH/bin/activate"
    fi

    # Test imports
    info "Testing imports..."
    python -c "
import sys
import torch
import triton
import numpy as np
import transformers
import einops

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Triton: {triton.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Einops: {einops.__version__}')
" || fatal "Import verification failed"

    # Test Tritter imports
    info "Testing Tritter imports..."
    python -c "from tritter import *; print('Tritter imports OK')" || fatal "Tritter import failed"

    # Run quick test suite (skip GPU tests if no CUDA)
    if [[ "$INSTALL_CUDA" == "true" ]] && command -v nvidia-smi &> /dev/null; then
        info "Running test suite (including GPU tests)..."
        pytest tests/ -v --tb=short -m "not slow" || warn "Some tests failed"
    else
        info "Running test suite (CPU only)..."
        pytest tests/ -v --tb=short -m "not slow and not gpu" || warn "Some tests failed"
    fi

    success "Verification complete"
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  Tritter Development Environment Ready  "
    echo "=========================================="
    echo ""

    if [[ "$SKIP_VENV" != "true" ]]; then
        echo "Activate the virtual environment:"
        echo "  source $VENV_PATH/bin/activate"
        echo ""
    fi

    echo "Useful commands:"
    echo "  pytest                    # Run tests"
    echo "  pytest --cov              # Run with coverage"
    echo "  ruff check .              # Lint code"
    echo "  ruff format .             # Format code"
    echo "  mypy src/tritter          # Type check"
    echo ""
    echo "Hardware info:"
    echo "  python scripts/hardware_profile.py --check 7B"
    echo ""
    echo "Model specs:"
    echo "  python scripts/show_model_specs.py --model 7B"
    echo ""
    echo "RTX 5080 feasibility:"
    echo "  python scripts/rtx5080_feasibility.py"
    echo ""

    if [[ "$INSTALL_CUDA" == "true" ]]; then
        echo "GPU support: Enabled"
        echo "CUDA version: $(nvcc --version | grep release | sed -n 's/.*release \([0-9.]*\).*/\1/p' || echo 'N/A')"
    else
        echo "GPU support: Disabled (use --no-cuda for CPU-only)"
    fi

    if [[ "$INSTALL_BLACKWELL" == "true" ]]; then
        echo "PyTorch: Nightly (Blackwell/RTX 50-series support)"
    fi

    echo ""
    echo "Documentation: docs/DEVELOPMENT_STANDARDS.md"
    echo "Architecture: docs/project-plan.md"
    echo ""
}

# Main execution
main() {
    info "Starting Tritter development environment setup..."
    echo ""

    check_os
    check_sudo
    install_system_deps
    install_python
    install_cuda
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
