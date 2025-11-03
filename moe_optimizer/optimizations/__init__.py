"""
Optimization modules package
"""

# Import optimizations with graceful fallback
try:
    from .fp8_quantization import FP8QuantizationOptimizer
except Exception as e:
    import logging
    logging.warning(f"Failed to import FP8QuantizationOptimizer: {e}")
    FP8QuantizationOptimizer = None

from .dual_batch_overlap import DualBatchOverlapOptimizer
from .disaggregation import PrefillDecodeDisaggregator
from .kv_cache import KVCacheTieringOptimizer
from .expert_placement import ExpertPlacementOptimizer
from .sparsity import StructuredSparsityOptimizer
from .flash_dmoe import FlashDMoEOptimizer

__all__ = [
    "FP8QuantizationOptimizer",
    "DualBatchOverlapOptimizer",
    "PrefillDecodeDisaggregator",
    "KVCacheTieringOptimizer",
    "ExpertPlacementOptimizer",
    "StructuredSparsityOptimizer",
    "FlashDMoEOptimizer",
]
