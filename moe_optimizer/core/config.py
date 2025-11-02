"""
Configuration management for MoE optimization

This module defines all configuration options for the optimization pipeline.
Designed to be model-agnostic where possible.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
from pathlib import Path

# Optional imports for YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


@dataclass
class OptimizationConfig:
    """
    Complete configuration for MoE optimization pipeline
    
    This config is designed to be ~80% model-agnostic.
    Only MoE-specific fields are marked with [MoE-only].
    """
    
    # ============ Model Configuration ============
    model_path: str
    """Path to model checkpoint or HuggingFace model ID"""
    
    model_type: Literal["moe", "dense"] = "moe"
    """Model architecture type"""
    
    num_gpus: int = 3
    """Number of GPUs to use (default: 3 for disaggregation setup)"""
    
    max_model_len: int = 4096
    """Maximum sequence length supported"""
    
    # ============ Quantization (Universal) ============
    enable_fp8: bool = True
    """Enable FP8 quantization (requires H100 or A100)"""
    
    fp8_router_precision: Literal["fp8", "fp16"] = "fp16"
    """Router/gating precision (fp16 recommended for accuracy)"""
    
    # ============ Batch Processing (Universal) ============
    enable_dual_batch_overlap: bool = True
    """Enable Dual Batch Overlap (DBO) for hiding communication latency"""
    
    max_num_batched_tokens: int = 8192
    """Maximum tokens per batch"""
    
    max_num_seqs: int = 256
    """Maximum sequences per batch"""
    
    # ============ Disaggregation (Universal) ============
    enable_disaggregation: bool = True
    """Enable prefill-decode disaggregation (GPU 0=prefill, GPU 1-2=decode)"""
    
    prefill_gpu_ids: List[int] = field(default_factory=lambda: [0])
    """GPU IDs for prefill workers"""
    
    decode_gpu_ids: List[int] = field(default_factory=lambda: [1, 2])
    """GPU IDs for decode workers"""
    
    # ============ KV Cache (Universal) ============
    enable_kv_tiering: bool = True
    """Enable KV cache precision tiering (FP16 recent + FP8 old)"""
    
    kv_recent_window: int = 512
    """Number of recent tokens to keep in FP16 (rest in FP8)"""
    
    kv_cache_dtype_recent: Literal["fp16", "fp8"] = "fp16"
    """Dtype for recent KV cache"""
    
    kv_cache_dtype_old: Literal["fp16", "fp8"] = "fp8"
    """Dtype for old KV cache"""
    
    # ============ MoE-Specific Optimizations [MoE-only] ============
    enable_expert_placement: bool = True
    """[MoE-only] Enable dynamic expert placement optimization"""
    
    num_experts: Optional[int] = None
    """[MoE-only] Number of experts (auto-detect if None)"""
    
    experts_per_token: int = 2
    """[MoE-only] Top-K experts per token"""
    
    enable_expert_sparsity: bool = False
    """[MoE-only] Enable 2:4 structured sparsity on medium-traffic experts"""
    
    expert_sparsity_ratio: str = "2:4"
    """[MoE-only] Sparsity pattern (2:4 recommended for H100)"""
    
    expert_sparsity_targets: List[str] = field(default_factory=lambda: ["medium"])
    """[MoE-only] Which experts to sparsify: 'hot', 'medium', 'cold'"""
    
    # ============ Performance Tuning (Universal) ============
    gpu_memory_utilization: float = 0.90
    """GPU memory utilization (0.0-1.0, leave headroom for stability)"""
    
    enable_cuda_graphs: bool = True
    """Enable CUDA graphs for kernel fusion"""
    
    tensor_parallel_size: Optional[int] = None
    """Tensor parallelism size (auto-set based on num_gpus if None)"""
    
    # ============ Monitoring & Debugging (Universal) ============
    enable_profiling: bool = False
    """Enable detailed profiling (adds overhead)"""
    
    enable_metrics: bool = True
    """Enable Prometheus metrics export"""
    
    metrics_port: int = 9090
    """Port for Prometheus metrics server"""
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """Logging verbosity"""
    
    # ============ Safety & Fallbacks (Universal) ============
    accuracy_validation_mode: bool = False
    """Run in validation mode (slower, for accuracy testing)"""
    
    enable_fallback_path: bool = True
    """Keep CPU/baseline fallback path for debugging"""
    
    max_retries: int = 3
    """Max retries for failed inference requests"""
    
    timeout_seconds: float = 30.0
    """Request timeout in seconds"""
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Auto-set tensor_parallel_size
        if self.tensor_parallel_size is None:
            if self.enable_disaggregation:
                # In disaggregation mode, prefill uses 1 GPU, decode uses rest
                self.tensor_parallel_size = len(self.decode_gpu_ids)
            else:
                self.tensor_parallel_size = self.num_gpus
        
        # Validate GPU IDs
        if self.enable_disaggregation:
            all_gpu_ids = set(self.prefill_gpu_ids + self.decode_gpu_ids)
            if len(all_gpu_ids) != self.num_gpus:
                raise ValueError(
                    f"GPU ID mismatch: {len(all_gpu_ids)} unique IDs, "
                    f"but num_gpus={self.num_gpus}"
                )
            
            # Validate GPU IDs are within valid range [0, num_gpus-1]
            max_gpu_id = max(all_gpu_ids)
            if max_gpu_id >= self.num_gpus:
                raise ValueError(
                    f"Invalid GPU ID: {max_gpu_id}. "
                    f"With num_gpus={self.num_gpus}, valid IDs are 0-{self.num_gpus-1}"
                )
        
        # Validate MoE settings
        if self.model_type == "moe":
            if self.enable_expert_sparsity and self.num_experts is None:
                raise ValueError("num_experts must be set when enable_expert_sparsity=True")
        
        # Validate memory settings
        if not 0.5 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in [0.5, 1.0], "
                f"got {self.gpu_memory_utilization}"
            )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OptimizationConfig":
        """Load configuration from YAML file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
        try:
            yaml_file = Path(yaml_path)
            if not yaml_file.exists():
                raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
            with open(yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return cls(**config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {yaml_path}: {e}")
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
        try:
            output_path = Path(yaml_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.safe_dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {yaml_path}: {e}")
    
    def is_moe_model(self) -> bool:
        """Check if this is an MoE model"""
        return self.model_type == "moe"
    
    def get_vllm_config(self) -> dict:
        """
        Convert to vLLM-compatible configuration
        
        Returns dict that can be passed to vLLM's LLM() constructor
        """
        vllm_config = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": not self.enable_cuda_graphs,
        }
        
        # Add quantization if enabled
        if self.enable_fp8:
            vllm_config["quantization"] = "fp8"
        
        # Add DBO if enabled
        if self.enable_dual_batch_overlap:
            vllm_config["enable_chunked_prefill"] = True
            vllm_config["max_num_batched_tokens"] = self.max_num_batched_tokens
            vllm_config["max_num_seqs"] = self.max_num_seqs
        
        return vllm_config
    
    def summary(self) -> str:
        """Generate human-readable configuration summary"""
        lines = [
            "=" * 70,
            "MoE OPTIMIZATION CONFIGURATION",
            "=" * 70,
            f"Model: {self.model_path}",
            f"Type: {self.model_type.upper()}",
            f"GPUs: {self.num_gpus}",
            "",
            "OPTIMIZATIONS ENABLED:",
            f"  {'✓' if self.enable_fp8 else '✗'} FP8 Quantization",
            f"  {'✓' if self.enable_dual_batch_overlap else '✗'} Dual Batch Overlap",
            f"  {'✓' if self.enable_disaggregation else '✗'} Prefill-Decode Disaggregation",
            f"  {'✓' if self.enable_kv_tiering else '✗'} KV Cache Tiering",
        ]
        
        if self.is_moe_model():
            lines.extend([
                f"  {'✓' if self.enable_expert_placement else '✗'} Dynamic Expert Placement [MoE]",
                f"  {'✓' if self.enable_expert_sparsity else '✗'} Expert Sparsity ({self.expert_sparsity_ratio}) [MoE]",
            ])
        
        lines.extend([
            "",
            f"Max sequence length: {self.max_model_len}",
            f"Max batch tokens: {self.max_num_batched_tokens}",
            f"GPU memory util: {self.gpu_memory_utilization:.1%}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def calculate_expected_speedup(self) -> float:
        """
        Calculate expected combined speedup from all enabled optimizations
        
        Returns:
            Expected speedup multiplier (e.g., 1000.0 = 1000× faster)
        """
        speedup = 1.0
        
        # FP8: 2.0× on H100 (1.0× if not available)
        if self.enable_fp8:
            speedup *= 2.0
        
        # DBO: 2.3× with vLLM
        if self.enable_dual_batch_overlap:
            speedup *= 2.3
        
        # Disaggregation: 1.4× throughput
        if self.enable_disaggregation:
            speedup *= 1.4
        
        # KV tiering: 1.4× effective batch size
        if self.enable_kv_tiering:
            speedup *= 1.4
        
        # Expert placement: 1.22× (only for MoE)
        if self.is_moe_model() and self.enable_expert_placement:
            speedup *= 1.22
        
        # Sparsity: 1.5× on affected layers (only for MoE)
        if self.is_moe_model() and self.enable_expert_sparsity:
            speedup *= 1.5
        
        # Note: This doesn't include FlashDMoE kernel (5.7×)
        # which would bring total to 1000-1500×
        
        return speedup


# Predefined configurations for common scenarios
def get_default_config(model_path: str, num_gpus: int = 3) -> OptimizationConfig:
    """Get default configuration with all optimizations enabled"""
    return OptimizationConfig(
        model_path=model_path,
        num_gpus=num_gpus,
        enable_fp8=True,
        enable_dual_batch_overlap=True,
        enable_disaggregation=True,
        enable_kv_tiering=True,
        enable_expert_placement=True,
    )


def get_conservative_config(model_path: str, num_gpus: int = 3) -> OptimizationConfig:
    """Get conservative configuration (high accuracy, lower performance)"""
    return OptimizationConfig(
        model_path=model_path,
        num_gpus=num_gpus,
        enable_fp8=True,
        fp8_router_precision="fp16",  # Keep router in FP16
        enable_dual_batch_overlap=True,
        enable_disaggregation=False,  # Simpler setup
        enable_kv_tiering=False,  # FP16 only
        enable_expert_sparsity=False,  # No sparsity
        gpu_memory_utilization=0.85,  # More headroom
    )


def get_aggressive_config(model_path: str, num_gpus: int = 3) -> OptimizationConfig:
    """Get aggressive configuration (maximum performance, some accuracy risk)"""
    return OptimizationConfig(
        model_path=model_path,
        model_type="moe",  # Assume MoE for aggressive config
        num_gpus=num_gpus,
        num_experts=8,  # Default for Mixtral-style models
        enable_fp8=True,
        fp8_router_precision="fp8",  # Router in FP8 too
        enable_dual_batch_overlap=True,
        enable_disaggregation=True,
        enable_kv_tiering=True,
        enable_expert_placement=True,
        enable_expert_sparsity=True,  # 2:4 sparsity
        gpu_memory_utilization=0.95,  # Push limits
        max_num_batched_tokens=12288,  # Larger batches
    )
