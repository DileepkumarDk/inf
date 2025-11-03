"""
MoE Optimizer - High-Performance Mixture-of-Experts Inference Optimization

This package implements production-ready optimizations for MoE models:
- FP8 quantization (H100-native)
- Dual Batch Overlap (DBO)
- Prefill-Decode disaggregation
- KV cache tiering
- Dynamic expert placement
"""

__version__ = "1.0.0"
__author__ = "Your Team"

from .core.engine import OptimizedMoEEngine
from .core.config import OptimizationConfig

__all__ = [
    "OptimizedMoEEngine",
    "OptimizationConfig",
]
