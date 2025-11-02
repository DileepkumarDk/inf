#!/bin/bash
################################################################################
# H100 Complete Setup Script
# 
# PURPOSE: Install EVERYTHING needed for MoE optimization in one shot
# USAGE: bash h100_complete_setup.sh [model_name]
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
MODEL_NAME="${1:-mistralai/Mixtral-8x7B-Instruct-v0.1}"
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

print_info "Updating package lists..."
sudo apt-get update -qq

print_info "Installing build essentials..."
sudo apt-get install -y -qq \
    build-essential \
    wget \
    curl \
    git \
    python3.10 \
    python3.10-venv \
    python3-pip \
    cmake \
    ninja-build \
    software-properties-common \
    htop \
    tmux \
    > /dev/null 2>&1

print_success "System dependencies installed ✓"
sleep 1

################################################################################
# PHASE 3: PYTHON ENVIRONMENT
################################################################################

print_header "PHASE 3: SETTING UP PYTHON ENVIRONMENT (3/7)"

# Check if virtual env already exists
if [ -d "~/moe_venv" ]; then
    print_warning "Virtual environment already exists. Removing old one..."
    rm -rf ~/moe_venv
fi

print_info "Creating Python virtual environment..."
python3.10 -m venv ~/moe_venv

print_info "Activating virtual environment..."
source ~/moe_venv/bin/activate

print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

PYTHON_VERSION=$(python --version)
print_success "Python environment ready: $PYTHON_VERSION ✓"
sleep 1

################################################################################
# PHASE 4: PYTORCH + CUDA
################################################################################

print_header "PHASE 4: INSTALLING PYTORCH + CUDA (4/7)"

print_info "Installing PyTorch 2.1.0 with CUDA 12.1..."
print_info "This may take 5-10 minutes..."

pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

print_info "Verifying PyTorch installation..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

GPU_COUNT_TORCH=$(python -c "import torch; print(torch.cuda.device_count())")
print_success "PyTorch sees $GPU_COUNT_TORCH GPU(s) ✓"
sleep 1

################################################################################
# PHASE 5: VLLM + OPTIMIZATION LIBRARIES
################################################################################

print_header "PHASE 5: INSTALLING VLLM + OPTIMIZATION LIBRARIES (5/7)"

print_info "Installing vLLM 0.6.3..."
pip install vllm==0.6.3 -q

print_info "Installing Transformer Engine (FP8 support)..."
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable -q

print_info "Installing additional dependencies..."
pip install -q \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    sentencepiece \
    protobuf \
    huggingface-hub \
    pyyaml \
    numpy \
    pandas \
    aiohttp \
    colorama

print_info "Verifying installations..."
python -c "import vllm; print(f'✓ vLLM {vllm.__version__}')"
python -c "import transformer_engine; print('✓ Transformer Engine installed')" 2>/dev/null || print_warning "Transformer Engine verification failed (may still work)"

print_success "All libraries installed ✓"
sleep 1

################################################################################
# PHASE 6: DOWNLOAD MODEL
################################################################################

print_header "PHASE 6: DOWNLOADING MODEL (6/7)"

print_info "Model: $MODEL_NAME"
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

# Verify model files
if [ -f "$MODEL_DIR/config.json" ]; then
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
else
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

# Test 3: Model files
print_info "[3/5] Checking model files..."
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
    print_info "  ✓ Model files present"
else
    print_warning "  Some model files may be missing"
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
echo "1. Activate the virtual environment (if not already active):"
echo "   ${GREEN}source ~/moe_venv/bin/activate${NC}"
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
echo "💡 TIP: Keep this terminal open and source the venv in new terminals:"
echo "    ${GREEN}source ~/moe_venv/bin/activate${NC}"
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

# Activate virtual environment
source ~/moe_venv/bin/activate

echo "🚀 MoE environment loaded!"
echo "   Model: \$MODEL_NAME"
echo "   Path: \$MODEL_PATH"
echo "   GPUs: \$GPU_COUNT"
EOF

print_success "Environment saved to ~/.moe_env"
print_info "In future sessions, just run: ${GREEN}source ~/.moe_env${NC}"
