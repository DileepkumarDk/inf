#!/bin/bash

# Install PyTorch with CUDA 12.6 support on H100 server
# Run this on your H100 machine if PyTorch doesn't have CUDA support

set -e

echo "=========================================="
echo "PyTorch CUDA Installation Script"
echo "=========================================="
echo ""

# Check for Python development headers
echo "[INFO] Checking for Python development headers..."
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "")
if [ -z "$PYTHON_INCLUDE" ] || [ ! -f "$PYTHON_INCLUDE/Python.h" ]; then
    echo "[WARNING] Python development headers not found!"
    echo "[INFO] Attempting to install python-dev..."
    
    # Try conda first (since you're using miniconda)
    if command -v conda &> /dev/null; then
        echo "[INFO] Installing via conda..."
        conda install -y python-dev || true
    fi
    
    # Verify installation
    PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "")
    if [ -z "$PYTHON_INCLUDE" ] || [ ! -f "$PYTHON_INCLUDE/Python.h" ]; then
        echo "[WARNING] Could not install python-dev via conda"
        echo "[INFO] Please install manually:"
        echo "  conda install -c conda-forge python-dev"
        echo "  OR: sudo apt-get install python3-dev"
    else
        echo "[SUCCESS] Python development headers installed!"
    fi
else
    echo "[SUCCESS] Python development headers found at: $PYTHON_INCLUDE"
fi
echo ""

# Check current PyTorch installation
echo "[INFO] Checking current PyTorch installation..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "Current PyTorch version: $TORCH_VERSION"
echo "CUDA available: $CUDA_AVAILABLE"
echo ""

if [ "$CUDA_AVAILABLE" == "True" ]; then
    echo "[SUCCESS] PyTorch with CUDA is already installed!"
    echo ""
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
    exit 0
fi

echo "[WARNING] PyTorch does not have CUDA support!"
echo "[INFO] Installing PyTorch with CUDA 12.6 support..."
echo ""

# Uninstall CPU-only version if present
if [ "$TORCH_VERSION" != "not installed" ]; then
    echo "[INFO] Uninstalling CPU-only PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    echo ""
fi

# Install CUDA-enabled PyTorch
echo "[INFO] Installing PyTorch 2.5.1 with CUDA 12.6..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "[INFO] Verifying installation..."
python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ Number of GPUs: {torch.cuda.device_count()}')
    print(f'✓ GPU 0: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=========================================="
echo "[SUCCESS] PyTorch with CUDA installed!"
echo "=========================================="
echo ""
echo "You can now run: bash build.sh"
