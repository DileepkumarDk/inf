"""
Unit tests for optimization modules

Run with: pytest tests/
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from moe_optimizer.optimizations import (
    FP8QuantizationOptimizer,
    DualBatchOverlapOptimizer,
    PrefillDecodeDisaggregator,
    KVCacheTieringOptimizer,
    ExpertPlacementOptimizer,
    StructuredSparsityOptimizer,
)


class TestFP8Quantization:
    """Test FP8 quantization optimizer"""
    
    def test_initialization(self):
        """Test basic initialization"""
        opt = FP8QuantizationOptimizer(model_config={})
        assert opt is not None
        assert opt.router_precision == "fp16"
    
    def test_vllm_config(self):
        """Test vLLM config generation"""
        opt = FP8QuantizationOptimizer(model_config={})
        config = opt.get_vllm_config()
        assert isinstance(config, dict)
    
    def test_expected_speedup(self):
        """Test speedup calculation"""
        opt = FP8QuantizationOptimizer(model_config={})
        speedup = opt.get_expected_speedup()
        assert speedup >= 1.0
        assert speedup <= 2.5


class TestDualBatchOverlap:
    """Test DBO optimizer"""
    
    def test_initialization(self):
        """Test basic initialization"""
        opt = DualBatchOverlapOptimizer()
        assert opt is not None
        assert opt.is_available()
    
    def test_vllm_config(self):
        """Test vLLM config generation"""
        opt = DualBatchOverlapOptimizer()
        config = opt.get_vllm_config()
        assert "enable_chunked_prefill" in config
        assert config["enable_chunked_prefill"] == True
    
    def test_expected_speedup(self):
        """Test speedup calculation"""
        opt = DualBatchOverlapOptimizer()
        speedup = opt.get_expected_speedup()
        assert speedup == 2.3


class TestDisaggregation:
    """Test prefill-decode disaggregation"""
    
    def test_initialization(self):
        """Test basic initialization"""
        opt = PrefillDecodeDisaggregator(num_gpus=3)
        assert opt is not None
        assert opt.num_gpus == 3
    
    def test_kv_transfer_estimation(self):
        """Test KV transfer time estimation"""
        opt = PrefillDecodeDisaggregator()
        time_ms = opt.estimate_kv_transfer_time(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            batch_size=512,
            seq_len=2048,
        )
        assert time_ms > 0
        assert time_ms < 1000  # Should be < 1 second


class TestKVCacheTiering:
    """Test KV cache tiering"""
    
    def test_initialization(self):
        """Test basic initialization"""
        opt = KVCacheTieringOptimizer()
        assert opt is not None
        assert opt.hot_tier_size == 2048
    
    def test_memory_savings(self):
        """Test memory savings calculation"""
        opt = KVCacheTieringOptimizer()
        savings = opt.calculate_memory_savings(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            seq_len=4096,
        )
        assert savings["savings_ratio"] > 1.0


class TestExpertPlacement:
    """Test expert placement optimizer"""
    
    def test_initialization(self):
        """Test basic initialization"""
        opt = ExpertPlacementOptimizer(num_gpus=3, num_experts=8)
        assert opt is not None
        assert opt.num_experts == 8
    
    def test_routing_analysis(self):
        """Test routing pattern analysis"""
        opt = ExpertPlacementOptimizer(num_gpus=3, num_experts=8)
        stats = opt.analyze_routing_patterns()
        assert "expert_frequencies" in stats
        assert len(stats["expert_frequencies"]) == 8
    
    def test_optimal_placement(self):
        """Test placement computation"""
        opt = ExpertPlacementOptimizer(num_gpus=3, num_experts=8)
        opt.analyze_routing_patterns()
        placement = opt.compute_optimal_placement()
        assert len(placement) == 8
        # All experts should be assigned to valid GPUs
        assert all(0 <= gpu_id < 3 for gpu_id in placement.values())


class TestStructuredSparsity:
    """Test 2:4 structured sparsity"""
    
    def test_initialization(self):
        """Test basic initialization"""
        opt = StructuredSparsityOptimizer()
        assert opt is not None
        assert opt.sparsity_pattern == "2:4"
    
    def test_medium_traffic_identification(self):
        """Test medium traffic expert identification"""
        opt = StructuredSparsityOptimizer()
        frequencies = [0.05, 0.15, 0.25, 0.35, 0.20, 0.10, 0.30, 0.40]
        medium = opt.identify_medium_traffic_experts(
            frequencies,
            threshold_low=0.1,
            threshold_high=0.3,
        )
        # Experts with freq in [0.1, 0.3] should be identified
        assert 1 in medium  # 0.15
        assert 2 in medium  # 0.25
        assert 4 in medium  # 0.20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
