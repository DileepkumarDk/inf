"""
Setup Validation Script
Verifies all components are installed correctly before starting optimization work
"""

import sys
import subprocess
import platform
from typing import Tuple, List

# Check if we're on Windows and if colorama is available
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    try:
        import colorama
        colorama.init()  # Enable ANSI colors on Windows
        COLORS_AVAILABLE = True
    except ImportError:
        COLORS_AVAILABLE = False
else:
    COLORS_AVAILABLE = True

class Colors:
    if COLORS_AVAILABLE:
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        END = '\033[0m'
    else:
        # No colors on Windows without colorama
        GREEN = ''
        RED = ''
        YELLOW = ''
        BLUE = ''
        END = ''

def check(name: str) -> Tuple[bool, str]:
    """Check if a package is installed"""
    try:
        __import__(name)
        return True, f"✓ {name}"
    except ImportError:
        return False, f"✗ {name} not found"

def main():
    print("=" * 70)
    print("MoE OPTIMIZATION - SETUP VALIDATION")
    print("=" * 70)
    
    all_passed = True
    
    # 1. Check Python packages
    print(f"\n{Colors.BLUE}[1/7] Checking Core Python Packages...{Colors.END}")
    core_packages = [
        'torch',
        'transformers',
        'accelerate',
        'datasets',
        'numpy',
        'tqdm'
    ]
    
    for pkg in core_packages:
        passed, msg = check(pkg)
        color = Colors.GREEN if passed else Colors.RED
        print(f"  {color}{msg}{Colors.END}")
        if not passed:
            all_passed = False
    
    # 2. Check CUDA
    print(f"\n{Colors.BLUE}[2/7] Checking CUDA Support...{Colors.END}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  {Colors.GREEN}✓ CUDA available{Colors.END}")
            print(f"    CUDA Version: {torch.version.cuda}")
            print(f"    PyTorch Version: {torch.__version__}")
            
            gpu_count = torch.cuda.device_count()
            print(f"    GPU Count: {gpu_count}")
            
            for i in range(min(gpu_count, 3)):  # Check first 3 GPUs
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024**3
                print(f"    GPU {i}: {props.name} ({mem_gb:.1f} GB)")
            
            if gpu_count < 3:
                print(f"  {Colors.YELLOW}⚠️  Warning: Need 3 GPUs for full plan, have {gpu_count}{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ CUDA not available{Colors.END}")
            all_passed = False
    except Exception as e:
        print(f"  {Colors.RED}✗ Error checking CUDA: {e}{Colors.END}")
        all_passed = False
    
    # 3. Check vLLM with version validation (FIX #12: Enhanced version check)
    print(f"\n{Colors.BLUE}[3/7] Checking vLLM...{Colors.END}")
    try:
        import vllm
        print(f"  {Colors.GREEN}✓ vLLM installed{Colors.END}")
        
        vllm_version = vllm.__version__
        print(f"    Version: {vllm_version}")
        
        # FIX #12: Check minimum version (0.6.0+ recommended for all features)
        try:
            # Parse version more robustly
            version_str = vllm_version.split('+')[0]  # Remove commit hash if present
            version_parts = version_str.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            patch = int(version_parts[2]) if len(version_parts) > 2 else 0
            
            if major == 0 and minor < 3:
                print(f"  {Colors.RED}✗ vLLM version {vllm_version} is too old{Colors.END}")
                print(f"    Required: 0.3.0+ (minimum)")
                print(f"    Recommended: 0.6.0+ (for full features)")
                print(f"    Install: pip install --upgrade vllm")
                all_passed = False
            elif major == 0 and minor < 6:
                print(f"  {Colors.YELLOW}⚠️  vLLM version {vllm_version} is functional but old{Colors.END}")
                print(f"    Recommended: 0.6.0+ for all features")
                print(f"    Some optimizations may not be available")
            else:
                print(f"  {Colors.GREEN}✓ vLLM version is compatible{Colors.END}")
        except (ValueError, IndexError) as e:
            print(f"  {Colors.YELLOW}⚠️  Could not parse vLLM version: {e}{Colors.END}")
            print(f"    Assuming compatible, but verify manually")
            
    except ImportError:
        print(f"  {Colors.RED}✗ vLLM not found{Colors.END}")
        print(f"    Install: pip install vllm")
        all_passed = False
    
    # 4. Check Transformer Engine with version validation
    print(f"\n{Colors.BLUE}[4/7] Checking Transformer Engine (FP8)...{Colors.END}")
    try:
        import transformer_engine
        print(f"  {Colors.GREEN}✓ Transformer Engine installed{Colors.END}")
        
        # Version check (FIX: Add version validation)
        try:
            te_version = transformer_engine.__version__
            print(f"    Version: {te_version}")
            
            # Check minimum version (1.0.0+)
            major_version = int(te_version.split('.')[0])
            if major_version < 1:
                print(f"  {Colors.YELLOW}⚠️  Warning: TransformerEngine version {te_version} is old{Colors.END}")
                print(f"    Recommended: 1.0.0+")
                all_passed = False
        except (AttributeError, ValueError):
            print(f"  {Colors.YELLOW}⚠️  Could not determine TransformerEngine version{Colors.END}")
        
        # Check if H100 available for FP8
        import torch
        if torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability(0)  # Check first GPU
            if device_cap[0] >= 9:
                print(f"  {Colors.GREEN}✓ H100 detected - FP8 E4M3 supported{Colors.END}")
            elif device_cap[0] >= 8:
                print(f"  {Colors.YELLOW}⚠️  A100 detected - Limited FP8 support{Colors.END}")
            else:
                print(f"  {Colors.YELLOW}⚠️  GPU does not support native FP8{Colors.END}")
    except ImportError:
        print(f"  {Colors.YELLOW}⚠️  Transformer Engine not found (optional for non-H100){Colors.END}")
        print(f"    Install: pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable")
    
    # 5. Check FlashAttention
    print(f"\n{Colors.BLUE}[5/7] Checking FlashAttention...{Colors.END}")
    try:
        import flash_attn
        print(f"  {Colors.GREEN}✓ FlashAttention installed{Colors.END}")
    except ImportError:
        print(f"  {Colors.YELLOW}⚠️  FlashAttention not found (vLLM may still work){Colors.END}")
        print(f"    Install: pip install flash-attn --no-build-isolation")
    
    # 6. Check Megablocks
    print(f"\n{Colors.BLUE}[6/7] Checking Megablocks (MoE kernels)...{Colors.END}")
    try:
        import megablocks
        print(f"  {Colors.GREEN}✓ Megablocks installed{Colors.END}")
    except ImportError:
        print(f"  {Colors.YELLOW}⚠️  Megablocks not found{Colors.END}")
        print(f"    Install: git clone https://github.com/stanford-futuredata/megablocks.git")
        print(f"             cd megablocks && pip install -e .")
    
    # 7. Check NVLink (critical for multi-GPU)
    print(f"\n{Colors.BLUE}[7/7] Checking NVLink (Multi-GPU Communication)...{Colors.END}")
    
    # Check if nvidia-smi is available
    import shutil
    if not shutil.which('nvidia-smi'):
        print(f"  {Colors.YELLOW}⚠️  nvidia-smi not found in PATH{Colors.END}")
        print(f"     Skipping NVLink check")
    else:
        try:
            result = subprocess.run(
                ['nvidia-smi', 'nvlink', '-s'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'Link' in result.stdout:
                link_count = result.stdout.count('Link')
                print(f"  {Colors.GREEN}✓ NVLink detected ({link_count} links){Colors.END}")
                
                # Check topology
                topo_result = subprocess.run(
                    ['nvidia-smi', 'topo', '-m'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if 'NV' in topo_result.stdout:
                    print(f"  {Colors.GREEN}✓ NVLink topology confirmed{Colors.END}")
                else:
                    print(f"  {Colors.YELLOW}⚠️  NVLink may not be fully configured{Colors.END}")
            else:
                print(f"  {Colors.YELLOW}⚠️  NVLink not detected (may impact multi-GPU performance){Colors.END}")
        except subprocess.TimeoutExpired:
            print(f"  {Colors.RED}✗ nvidia-smi timed out{Colors.END}")
        except FileNotFoundError:
            print(f"  {Colors.YELLOW}⚠️  nvidia-smi not found{Colors.END}")
        except Exception as e:
            print(f"  {Colors.YELLOW}⚠️  Could not check NVLink: {e}{Colors.END}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(f"{Colors.GREEN}✓ VALIDATION PASSED{Colors.END}")
        print("\nYou're ready to start! Next steps:")
        print("  1. Review README.md for quick start guide")
        print("  2. Review BENCHMARK_PROTOCOL.md for testing phases")
        print("  3. Start with Stage 1 (FP8 + DBO) - works immediately")
    else:
        print(f"{Colors.RED}✗ VALIDATION FAILED{Colors.END}")
        print("\nPlease install missing components:")
        print("  - See README.md for installation instructions")
        print("  - Run this script again after installation")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
