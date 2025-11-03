/*
 * Python bindings for FlashDMoE CUDA kernel
 * 
 * Exposes the kernel to PyTorch via pybind11
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

// MoEConfig structure (must match kernel definition)
struct MoEConfig {
    int num_experts;
    int expert_dim;
    int hidden_dim;
    int top_k;
    int num_gpus;
    int expert_per_gpu;
    bool use_fp8;
    bool use_sparse_24;
};

// Forward declarations of CUDA functions with C linkage
extern "C" {
    void flash_dmoe_persistent_kernel_wrapper(
        const void* input_tokens,
        const void* gate_weights,
        const void* expert_weights,
        void* output_tokens,
        MoEConfig config,
        int batch_size,
        int my_gpu_id,
        int num_blocks,
        int threads_per_block,
        cudaStream_t stream
    );
    
    void flash_dmoe_init_kernel_wrapper(
        void* work_queue,
        int* queue_head,
        int* queue_tail,
        int* shutdown
    );
    
    void flash_dmoe_shutdown_kernel_wrapper();
}

/*
 * Python-facing forward function
 */
torch::Tensor flash_dmoe_forward(
    torch::Tensor input_tokens,
    torch::Tensor gate_weights,
    torch::Tensor expert_weights,
    int num_experts,
    int expert_dim,
    int top_k,
    int num_gpus,
    int my_gpu_id
) {
    // Validate inputs
    TORCH_CHECK(input_tokens.is_cuda(), "input_tokens must be on CUDA");
    TORCH_CHECK(gate_weights.is_cuda(), "gate_weights must be on CUDA");
    TORCH_CHECK(expert_weights.is_cuda(), "expert_weights must be on CUDA");
    
    TORCH_CHECK(input_tokens.dim() == 2, "input_tokens must be 2D [batch, hidden_dim]");
    TORCH_CHECK(gate_weights.dim() == 2, "gate_weights must be 2D [hidden_dim, num_experts]");
    TORCH_CHECK(expert_weights.dim() == 3, "expert_weights must be 3D [num_experts, expert_dim, hidden_dim]");
    
    const int batch_size = input_tokens.size(0);
    const int hidden_dim = input_tokens.size(1);
    
    TORCH_CHECK(gate_weights.size(0) == hidden_dim, "gate_weights dim 0 must match hidden_dim");
    TORCH_CHECK(gate_weights.size(1) == num_experts, "gate_weights dim 1 must match num_experts");
    TORCH_CHECK(expert_weights.size(0) == num_experts, "expert_weights dim 0 must match num_experts");
    
    // Determine precision
    bool use_fp8 = (input_tokens.scalar_type() == torch::kFloat8_e4m3fn);
    // Detect sparsity from weight tensor (check for structured 2:4 pattern)
    bool use_sparse = false;
    if (expert_weights.is_sparse()) {
        use_sparse = true;
    } else {
        // Check if weights follow 2:4 structured sparsity pattern
        // For now, assume dense unless explicitly marked as sparse
        use_sparse = false;
    }
    
    // Allocate output tensor
    auto output = torch::empty_like(input_tokens);
    
    // Set up MoE configuration
    MoEConfig config;
    config.num_experts = num_experts;
    config.expert_dim = expert_dim;
    config.hidden_dim = hidden_dim;
    config.top_k = top_k;
    config.num_gpus = num_gpus;
    config.expert_per_gpu = (num_experts + num_gpus - 1) / num_gpus;
    config.use_fp8 = use_fp8;
    config.use_sparse_24 = use_sparse;
    
    // Calculate grid dimensions
    const int num_blocks = 108;  // H100 has 108 SMs
    const int threads_per_block = 1024;  // 32 warps per block
    
    // Get CUDA stream (PyTorch 2.x API)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_tokens.device().index()).stream();
    
    // Launch kernel
    flash_dmoe_persistent_kernel_wrapper(
        input_tokens.data_ptr(),
        gate_weights.data_ptr(),
        expert_weights.data_ptr(),
        output.data_ptr(),
        config,
        batch_size,
        my_gpu_id,
        num_blocks,
        threads_per_block,
        stream
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "FlashDMoE kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

/*
 * Initialize persistent kernel
 */
void flash_dmoe_init() {
    // Allocate work queue on device
    void* work_queue;
    int* queue_head;
    int* queue_tail;
    int* shutdown;
    
    cudaMalloc(&work_queue, sizeof(int) * 1024);  // 1024 work items
    cudaMalloc(&queue_head, sizeof(int));
    cudaMalloc(&queue_tail, sizeof(int));
    cudaMalloc(&shutdown, sizeof(int));
    
    // Launch init kernel
    flash_dmoe_init_kernel_wrapper(work_queue, queue_head, queue_tail, shutdown);
    
    cudaDeviceSynchronize();
}

/*
 * Shutdown persistent kernel
 */
void flash_dmoe_shutdown() {
    flash_dmoe_shutdown_kernel_wrapper();
    cudaDeviceSynchronize();
}

/*
 * Pybind11 module definition
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDMoE persistent kernel for high-performance MoE inference";
    
    m.def("forward", &flash_dmoe_forward, "FlashDMoE forward pass",
          py::arg("input_tokens"),
          py::arg("gate_weights"),
          py::arg("expert_weights"),
          py::arg("num_experts"),
          py::arg("expert_dim"),
          py::arg("top_k") = 2,
          py::arg("num_gpus") = 1,
          py::arg("my_gpu_id") = 0);
    
    m.def("init", &flash_dmoe_init, "Initialize FlashDMoE persistent kernel");
    m.def("shutdown", &flash_dmoe_shutdown, "Shutdown FlashDMoE persistent kernel");
}

// Declare CUDA kernel functions (implemented in .cu file)
struct MoEConfig;
void flash_dmoe_persistent_kernel_wrapper(
    const void* input_tokens,
    const void* gate_weights,
    const void* expert_weights,
    void* output_tokens,
    MoEConfig config,
    int batch_size,
    int my_gpu_id,
    int num_blocks,
    int threads_per_block,
    cudaStream_t stream
);

void flash_dmoe_init_kernel_wrapper(void* work_queue, int* queue_head, int* queue_tail, int* shutdown);
void flash_dmoe_shutdown_kernel_wrapper();
