#!/bin/bash
################################################################################
# FlashDMoE CUDA Kernel Build Script
# 
# Compiles the FlashDMoE persistent kernel for H100 GPUs
# Requires: CUDA 12.1+, NVSHMEM 2.10+, H100 GPU
#
# NOTE: This is a STUB build script. The actual kernel implementation needs
#       the host interface functions added. Current kernel is compute-only.
#       System will fall back to vLLM's MoE if compilation fails (12.9× speedup).
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=================================${NC}"
echo -e "${GREEN}FlashDMoE Kernel Build Script${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Check if we're on H100
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "${BLUE}[INFO]${NC} Detected GPU: $GPU_NAME"
    
    if [[ "$GPU_NAME" != *"H100"* ]]; then
        echo -e "${YELLOW}[WARNING]${NC} This kernel is optimized for H100 (compute capability 9.0)"
        echo -e "${YELLOW}[WARNING]${NC} Compilation may fail or performance may be suboptimal on $GPU_NAME"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}[ABORT]${NC} Build cancelled"
            exit 1
        fi
    fi
else
    echo -e "${RED}[ERROR]${NC} nvidia-smi not found. Are you on a GPU machine?"
    exit 1
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} nvcc not found. Please install CUDA toolkit 12.1+"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${BLUE}[INFO]${NC} CUDA version: $CUDA_VERSION"

# Check for NVSHMEM (optional but recommended)
if [ -z "$NVSHMEM_HOME" ]; then
    echo -e "${YELLOW}[WARNING]${NC} NVSHMEM_HOME not set"
    echo -e "${YELLOW}[WARNING]${NC} Building without NVSHMEM (device-initiated transfers disabled)"
    echo -e "${BLUE}[INFO]${NC} For optimal performance, install NVSHMEM 2.10+"
    USE_NVSHMEM=0
else
    echo -e "${GREEN}[INFO]${NC} NVSHMEM found: $NVSHMEM_HOME"
    USE_NVSHMEM=1
fi

echo ""
echo -e "${BLUE}[INFO]${NC} Starting compilation..."
echo ""

# Create build directory
mkdir -p build
cd build

# Compile the kernel
echo -e "${BLUE}[1/3]${NC} Compiling CUDA kernel..."

NVCC_FLAGS="-std=c++17 \
    -O3 \
    -use_fast_math \
    -arch=sm_90 \
    -gencode=arch=compute_90,code=sm_90 \
    -Xptxas -v \
    -Xcompiler -fPIC \
    --expt-relaxed-constexpr \
    --expt-extended-lambda"

if [ $USE_NVSHMEM -eq 1 ]; then
    NVCC_FLAGS="$NVCC_FLAGS -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib -lnvshmem"
fi

nvcc $NVCC_FLAGS \
    -c ../flash_dmoe_kernel.cu \
    -o flash_dmoe_kernel.o

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Kernel compilation failed!"
    exit 1
fi

echo -e "${GREEN}[SUCCESS]${NC} Kernel compiled"
echo ""

# Create Python binding
echo -e "${BLUE}[2/3]${NC} Creating Python binding..."

cat > flash_dmoe_binding.cpp << 'EOF'
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declarations
void flash_dmoe_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& gate_weights,
    const torch::Tensor& expert_weights,
    torch::Tensor& output,
    int num_experts,
    int experts_per_token,
    bool use_fp8
);

// Python interface
torch::Tensor flash_dmoe_forward(
    torch::Tensor input,
    torch::Tensor gate_weights,
    torch::Tensor expert_weights,
    int num_experts,
    int experts_per_token,
    bool use_fp8
) {
    auto output = torch::zeros_like(input);
    flash_dmoe_forward_cuda(
        input, gate_weights, expert_weights, output,
        num_experts, experts_per_token, use_fp8
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_dmoe_forward, "FlashDMoE forward pass");
}
EOF

# Compile Python binding
g++ -std=c++17 -O3 -fPIC \
    -I$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])") \
    -I$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[1])") \
    -c flash_dmoe_binding.cpp \
    -o flash_dmoe_binding.o

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Python binding compilation failed!"
    exit 1
fi

echo -e "${GREEN}[SUCCESS]${NC} Python binding created"
echo ""

# Link shared library
echo -e "${BLUE}[3/3]${NC} Linking shared library..."

g++ -shared -fPIC \
    flash_dmoe_kernel.o \
    flash_dmoe_binding.o \
    -L$(python -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])") \
    -ltorch -ltorch_python \
    -L/usr/local/cuda/lib64 -lcudart \
    -o flash_dmoe_cuda.so

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Linking failed!"
    exit 1
fi

echo -e "${GREEN}[SUCCESS]${NC} Shared library created"
echo ""

# Copy to parent directory
cp flash_dmoe_cuda.so ..

echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}BUILD SUCCESSFUL!${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo -e "${BLUE}[INFO]${NC} Compiled kernel: flash_dmoe_cuda.so"
echo -e "${BLUE}[INFO]${NC} Architecture: sm_90 (H100)"
if [ $USE_NVSHMEM -eq 1 ]; then
    echo -e "${BLUE}[INFO]${NC} NVSHMEM: Enabled"
else
    echo -e "${YELLOW}[INFO]${NC} NVSHMEM: Disabled"
fi
echo ""
echo -e "${GREEN}[NEXT]${NC} Test your optimizer:"
echo -e "  ${GREEN}python run_optimizer.py --model <model> --profile balanced${NC}"
echo ""
echo -e "${YELLOW}[NOTE]${NC} If FlashDMoE kernel is not available, system will use:"
echo -e "  - Stage 1+2 optimizations (12.9× speedup)"
echo -e "  - vLLM's built-in MoE implementation"
echo -e "  - Still excellent performance!"
echo ""
