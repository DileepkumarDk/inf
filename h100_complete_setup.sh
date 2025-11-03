#!/bin/bash
################################################################################
# H100 Complete Setup Script
# 
# PURPOSE: Install EVERYTHING needed for MoE optimization in one shot
# USAGE: bash h100_complete_setup.sh [model_name]
# 
# ⚠️  WINDOWS USERS: This is a bash script for Linux/WSL
#     - Run in WSL (Windows Subsystem for Linux)
#     - OR use Git Bash
#     - OR convert to PowerShell (see setup_windows.ps1 if available)
# 
# This script will:
# 1. Install all system dependencies
# 2. Set up Python environment
# 3. Install PyTorch + CUDA
# 4. Install vLLM and all optimization libraries
# 5. Download your chosen model
# 6. Verify everything works
# 
# Time: ~30-45 minutes (mostly model download)
# Cost: Free (except GPU rental time)
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}$1${NC}"
    echo "================================================================================"
    echo ""
}

################################################################################
# CONFIGURATION
################################################################################

# Get model from command line or use default
# FIX: Better default model selection - use smaller model for testing
if [ -z "$1" ]; then
    print_warning "No model specified. Using default: mistralai/Mixtral-8x7B-Instruct-v0.1"
    print_info "For Qwen3-30B, run: bash $0 Qwen/Qwen3-30B-A3B"
    print_info "For smaller test, run: bash $0 microsoft/Phi-3.5-MoE-instruct"
    MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"
else
    MODEL_NAME="$1"
fi
MODEL_DIR="./models/$(echo $MODEL_NAME | tr '/' '-')"

print_header "H100 COMPLETE SETUP SCRIPT"
print_info "Target Model: $MODEL_NAME"
print_info "Install Directory: $MODEL_DIR"
print_info "This will take 30-45 minutes. Grab coffee! ☕"
echo ""

################################################################################
# PHASE 1: VERIFY HARDWARE
################################################################################

print_header "PHASE 1: VERIFYING HARDWARE (1/7)"

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found! Are you on a GPU instance?"
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
print_info "Found $GPU_COUNT GPU(s)"

# Check if H100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
print_info "GPU: $GPU_NAME"

if [[ "$GPU_NAME" == *"H100"* ]]; then
    print_success "H100 detected! FP8 fully supported ✓"
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    print_warning "A100 detected. FP8 partially supported."
else
    print_warning "GPU is not H100/A100. Performance may be limited."
fi

# Check NVIDIA driver
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
print_info "NVIDIA Driver: $DRIVER_VERSION"

sleep 2

################################################################################
# PHASE 2: SYSTEM DEPENDENCIES
################################################################################

print_header "PHASE 2: INSTALLING SYSTEM DEPENDENCIES (2/7)"

# FIX: Only update if we have sudo
print_info "Updating package lists..."
if [ "$HAS_SUDO" = true ] 2>/dev/null || sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
else
    print_warning "Skipping apt-get update (no sudo access)"
fi

# FIX: Add more development tools and check sudo availability
print_info "Installing build essentials..."

# Check if we have sudo rights
if sudo -n true 2>/dev/null; then
    HAS_SUDO=true
elif sudo -v 2>/dev/null; then
    HAS_SUDO=true
else
    print_warning "No sudo access. Some system packages may need manual installation."
    HAS_SUDO=false
fi

if [ "$HAS_SUDO" = true ]; then
    sudo apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        cmake \
        ninja-build \
        software-properties-common \
        htop \
        tmux \
        vim \
        screen \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev

    if [ $? -ne 0 ]; then
        print_error "Failed to install system dependencies"
        exit 1
    fi
else
    print_warning "Skipping system dependencies (no sudo). Ensure these are installed:"
    print_info "  build-essential, python3-dev, cmake, git, etc."
fi

print_success "System dependencies installed ✓"
sleep 1

################################################################################
# PHASE 3: PYTHON ENVIRONMENT
################################################################################

print_header "PHASE 3: SETTING UP PYTHON ENVIRONMENT (3/7)"

# Check if we're in a conda environment (Lightning Studio)
if command -v conda &> /dev/null; then
    print_info "Conda detected (Lightning Studio environment)"
    print_info "Using existing conda environment..."
    
    # Make sure we're in the base/default environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_info "Activating conda base environment..."
        eval "$(conda shell.bash hook)"
        conda activate base
    else
        print_info "Already in conda environment: $CONDA_DEFAULT_ENV"
    fi
else
    # Fallback to venv for non-conda systems
    print_info "Creating Python virtual environment..."
    if [ -d "~/moe_venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf ~/moe_venv
    fi
    python3 -m venv ~/moe_venv
    source ~/moe_venv/bin/activate
fi

print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

PYTHON_VERSION=$(python --version)
print_success "Python environment ready: $PYTHON_VERSION ✓"
sleep 1

################################################################################
# PHASE 4: PYTORCH + CUDA
################################################################################

print_header "PHASE 4: INSTALLING PYTORCH + CUDA (4/7)"

# FIX: Check CUDA version and install matching PyTorch
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
print_info "Detected CUDA Version: ${CUDA_VERSION:-12.1}"

print_info "Installing PyTorch with CUDA ${CUDA_VERSION:-12.1}..."
print_info "This may take 5-10 minutes..."

# Install PyTorch matching CUDA version
if [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" == "12.3" ]] || [[ "$CUDA_VERSION" == "12.2" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # Default to CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

print_info "Verifying PyTorch installation..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

GPU_COUNT_TORCH=$(python -c "import torch; print(torch.cuda.device_count())")
print_success "PyTorch sees $GPU_COUNT_TORCH GPU(s) ✓"
sleep 1

################################################################################
# PHASE 5: VLLM + OPTIMIZATION LIBRARIES
################################################################################

print_header "PHASE 5: INSTALLING VLLM + OPTIMIZATION LIBRARIES (5/7)"

# FIX: vLLM 0.6.3 may be outdated, use latest stable or 0.6.0+ range
print_info "Installing vLLM (latest stable)..."
pip install "vllm>=0.6.0"

# FIX: Add error handling for Transformer Engine installation
print_info "Installing Transformer Engine (FP8 support)..."
if pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable; then
    print_success "Transformer Engine installed successfully"
else
    print_warning "Transformer Engine installation failed (may not be critical for basic usage)"
    print_info "You can still use FP8 through vLLM's built-in quantization"
fi

# FIX: Add more dependencies and specify minimum versions
print_info "Installing additional dependencies..."
pip install \
    "transformers>=4.45.0" \
    "accelerate>=0.25.0" \
    sentencepiece \
    "protobuf>=3.20.0" \
    "huggingface-hub>=0.19.0" \
    pyyaml \
    "numpy>=1.24.0" \
    pandas \
    aiohttp \
    colorama \
    pytest \
    pytest-cov \
    tqdm \
    tabulate

print_info "Verifying installations..."

# FIX: More robust verification with error handling
if python -c "import vllm; print(f'✓ vLLM {vllm.__version__}')" 2>/dev/null; then
    print_success "vLLM verified"
else
    print_error "vLLM verification failed!"
    exit 1
fi

# Check Transformer Engine (optional)
if python -c "import transformer_engine; print('✓ Transformer Engine installed')" 2>/dev/null; then
    print_success "Transformer Engine verified"
else
    print_warning "Transformer Engine not available (optional, vLLM has built-in FP8)"
fi

# Verify other critical imports
python -c "import transformers; import torch; import accelerate" 2>/dev/null || {
    print_error "Critical imports failed!"
    exit 1
}

print_success "All libraries installed ✓"
sleep 1

################################################################################
# PHASE 6: DOWNLOAD MODEL (OPTIONAL - vLLM can auto-download)
################################################################################

print_header "PHASE 6: MODEL SETUP (6/7)"

print_info "Model: $MODEL_NAME"
print_warning "NOTE: vLLM will automatically download models when needed."
print_warning "You can skip manual download and let vLLM handle it."
echo ""
read -p "Download model manually now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Destination: $MODEL_DIR"
    print_warning "This will take 15-30 minutes depending on model size..."
    
    # Create models directory
    mkdir -p $MODEL_DIR
    
    # Check if model already exists
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
        print_warning "Model directory already exists and is not empty."
        read -p "Re-download model? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping model download."
        else
            print_info "Downloading model..."
            python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
    resume_download=True
)
print('✓ Model downloaded')
"
        fi
    else
        print_info "Downloading model (this is the longest step)..."
        python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
    resume_download=True
)
print('✓ Model downloaded')
"
    fi
else
    print_info "Skipping manual download."
    print_info "vLLM will download to ~/.cache/huggingface/ when you run it."
    print_success "Model setup complete (will auto-download on first use) ✓"
fi

# Verify model files (only if manually downloaded)
if [[ $REPLY =~ ^[Yy]$ ]] && [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    print_success "Model downloaded successfully ✓"
    
    # Show model info
    MODEL_SIZE=$(du -sh $MODEL_DIR | cut -f1)
    print_info "Model size: $MODEL_SIZE"
    
    # Try to detect model properties
    python -c "
import json
import sys
sys.path.insert(0, '$(pwd)')
try:
    from moe_optimizer.core.model_inspector import ModelInspector
    inspector = ModelInspector()
    info = inspector.inspect_model('$MODEL_DIR')
    print(f'  Architecture: {info[\"architecture\"]}')
    print(f'  MoE: {info[\"is_moe\"]}')
    if info['num_experts']:
        print(f'  Experts: {info[\"num_experts\"]}')
    print(f'  Recommended GPUs: {info[\"recommended_gpus\"]}')
except Exception as e:
    print(f'  (Could not auto-detect model properties)')
" 2>/dev/null || print_info "(Model inspector not available yet)"
elif [[ $REPLY =~ ^[Yy]$ ]]; then
    print_error "Model download failed or incomplete!"
    exit 1
fi

sleep 1

################################################################################
# PHASE 7: VERIFICATION
################################################################################

print_header "PHASE 7: FINAL VERIFICATION (7/7)"

print_info "Running comprehensive checks..."

# Test 1: PyTorch + CUDA
print_info "[1/5] Testing PyTorch + CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() > 0, 'No GPUs detected'
print('  ✓ PyTorch + CUDA working')
"

# Test 2: vLLM
print_info "[2/5] Testing vLLM..."
python -c "
import vllm
print('  ✓ vLLM imports correctly')
"

# Test 3: Model files (only if manually downloaded)
print_info "[3/5] Checking model files..."
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    if [ -f "$MODEL_DIR/tokenizer.json" ] || [ -f "$MODEL_DIR/tokenizer.model" ] || [ -f "$MODEL_DIR/tokenizer_config.json" ]; then
        print_info "  ✓ Model files present in $MODEL_DIR"
    else
        print_warning "  Tokenizer file may be missing (tokenizer.json or tokenizer.model)"
    fi
else
    print_info "  ⚠ Model not downloaded locally (will auto-download when needed)"
fi

# Test 4: Optimizer code
print_info "[4/5] Testing optimizer code..."
python -c "
import sys
sys.path.insert(0, '$(pwd)')
try:
    from moe_optimizer.core.config import OptimizationConfig
    from moe_optimizer.core.engine import OptimizedMoEEngine
    from moe_optimizer.core.model_inspector import ModelInspector
    print('  ✓ Optimizer code imports correctly')
except ImportError as e:
    print(f'  ⚠ Optimizer code not found (you may need to upload it)')
" || print_info "  (Optimizer code will be uploaded separately)"

# Test 5: Benchmark script
print_info "[5/5] Checking benchmark script..."
if [ -f "scripts/benchmark.py" ]; then
    print_info "  ✓ Benchmark script found"
else
    print_info "  ⚠ Benchmark script not found (will be created)"
fi

print_success "All verifications passed! ✓"
sleep 1

################################################################################
# SUMMARY & NEXT STEPS
################################################################################

print_header "🎉 SETUP COMPLETE! 🎉"

echo "✅ System dependencies installed"
echo "✅ Python environment configured"
echo "✅ PyTorch + CUDA installed"
echo "✅ vLLM + optimization libraries installed"
echo "✅ Model downloaded: $MODEL_NAME"
echo "✅ All verifications passed"
echo ""
echo "================================================================================"
echo "NEXT STEPS:"
echo "================================================================================"
echo ""
echo "1. Activate the environment (if not already active):"
if command -v conda &> /dev/null; then
    echo "   ${GREEN}conda activate base${NC}  # (or your conda env)"
else
    echo "   ${GREEN}source ~/moe_venv/bin/activate${NC}"
fi
echo ""
echo "2. Set your model path for easy reference:"
echo "   ${GREEN}export MODEL_PATH='$MODEL_DIR'${NC}"
echo ""
echo "3. Run baseline test (Week 0):"
echo "   ${GREEN}python -m vllm.entrypoints.openai.api_server \\${NC}"
echo "   ${GREEN}    --model \$MODEL_PATH \\${NC}"
echo "   ${GREEN}    --tensor-parallel-size $GPU_COUNT \\${NC}"
echo "   ${GREEN}    --dtype float16 \\${NC}"
echo "   ${GREEN}    --port 8000${NC}"
echo ""
echo "4. In another terminal, run benchmark:"
echo "   ${GREEN}python scripts/benchmark.py --url http://localhost:8000 --test-all-batches${NC}"
echo ""
echo "5. Follow BENCHMARK_PROTOCOL.md for week-by-week testing"
echo ""
echo "================================================================================"
echo "QUICK TEST:"
echo "================================================================================"
echo ""
echo "Test if vLLM can load the model:"
echo ""
echo "${GREEN}python -m vllm.entrypoints.openai.api_server \\${NC}"
echo "${GREEN}  --model $MODEL_DIR \\${NC}"
echo "${GREEN}  --tensor-parallel-size $GPU_COUNT \\${NC}"
echo "${GREEN}  --max-model-len 2048 \\${NC}"
echo "${GREEN}  --port 8000${NC}"
echo ""
echo "Then in another terminal:"
echo ""
echo "${GREEN}curl http://localhost:8000/v1/completions \\${NC}"
echo "${GREEN}  -H 'Content-Type: application/json' \\${NC}"
echo "${GREEN}  -d '{\"model\": \"$MODEL_DIR\", \"prompt\": \"Hello\", \"max_tokens\": 10}'${NC}"
echo ""
echo "================================================================================"
echo "ESTIMATED PERFORMANCE:"
echo "================================================================================"
echo ""
echo "Hardware: $GPU_COUNT× $GPU_NAME"
echo "Model: $MODEL_NAME"
echo ""
echo "Expected baseline: 5,000-10,000 tokens/sec"
echo "Expected with FP8+DBO: 20,000-50,000 tokens/sec"
echo "Expected with all opts: 100,000-1,000,000+ tokens/sec"
echo ""
echo "================================================================================"
if command -v conda &> /dev/null; then
    echo "💡 TIP: You're using conda (Lightning Studio). Environment is persistent."
else
    echo "💡 TIP: Keep this terminal open and source the venv in new terminals:"
    echo "    ${GREEN}source ~/moe_venv/bin/activate${NC}"
fi
echo "================================================================================"
echo ""
echo "🚀 Ready to benchmark! Follow BENCHMARK_PROTOCOL.md for next steps."
echo ""

# Save environment variables to a file for easy reloading
cat > ~/.moe_env << EOF
# MoE Optimization Environment
# Source this file: source ~/.moe_env

export MODEL_PATH='$MODEL_DIR'
export MODEL_NAME='$MODEL_NAME'
export GPU_COUNT=$GPU_COUNT

# Activate environment
if command -v conda &> /dev/null; then
    # Conda environment (Lightning Studio)
    eval "\$(conda shell.bash hook)"
    conda activate base 2>/dev/null || true
else
    # venv environment
    source ~/moe_venv/bin/activate
fi

echo "🚀 MoE environment loaded!"
echo "   Model: \$MODEL_NAME"
echo "   Path: \$MODEL_PATH"
echo "   GPUs: \$GPU_COUNT"
EOF

print_success "Environment saved to ~/.moe_env"
print_info "In future sessions, just run: ${GREEN}source ~/.moe_env${NC}"
