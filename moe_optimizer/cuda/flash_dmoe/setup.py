"""
Setup script for FlashDMoE CUDA kernel

Compiles the persistent MoE kernel for H100 GPUs.

Usage:
    python setup.py install
    
Requirements:
    - CUDA 12.1+
    - PyTorch 2.1.0+
    - H100 GPU (compute capability 9.0)
    - NVSHMEM 2.10+ (optional, for device-initiated transfers)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available - cannot build FlashDMoE kernel")

# Check compute capability
device_cap = torch.cuda.get_device_capability(0)
if device_cap[0] < 9:
    print(f"Warning: FlashDMoE is optimized for H100 (compute 9.0+), found {device_cap[0]}.{device_cap[1]}")
    print("Kernel will still compile but may not achieve full performance.")

# CUDA compilation flags
cuda_flags = [
    '-O3',
    '-use_fast_math',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-Xptxas=-v',  # Verbose register usage
    f'-arch=sm_{device_cap[0]}{device_cap[1]}',
    '-std=c++17',
]

# Check for NVSHMEM
nvshmem_home = os.environ.get('NVSHMEM_HOME', '/usr/local/nvshmem')
nvshmem_available = os.path.exists(os.path.join(nvshmem_home, 'include', 'nvshmem.h'))

if nvshmem_available:
    print(f"Found NVSHMEM at {nvshmem_home}")
    cuda_flags.append('-DUSE_NVSHMEM')
    include_dirs = [nvshmem_home + '/include']
    library_dirs = [nvshmem_home + '/lib']
    libraries = ['nvshmem']
else:
    print("NVSHMEM not found - device-initiated transfers will be disabled")
    print("For best performance, install NVSHMEM 2.10+ and set NVSHMEM_HOME")
    include_dirs = []
    library_dirs = []
    libraries = []

# Build extension
ext_modules = [
    CUDAExtension(
        name='flash_dmoe_cuda',
        sources=[
            'flash_dmoe_binding.cpp',
            'flash_dmoe_kernel.cu',
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': cuda_flags,
        },
    )
]

setup(
    name='flash_dmoe_cuda',
    version='0.1.0',
    description='FlashDMoE persistent kernel for H100',
    author='MoE Optimizer Team',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.1.0',
    ],
)
