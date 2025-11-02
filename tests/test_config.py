"""
Unit tests for core configuration system

Run with: pytest tests/
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from moe_optimizer.core.config import (
    OptimizationConfig,
    get_default_config,
    get_conservative_config,
    get_aggressive_config,
)


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass"""
    
    def test_basic_initialization(self):
        """Test basic config creation"""
        config = OptimizationConfig(
            model_path="test-model",
            num_gpus=3,
        )
        assert config.model_path == "test-model"
        assert config.num_gpus == 3
    
    def test_default_values(self):
        """Test default values are set correctly"""
        config = OptimizationConfig(model_path="test")
        assert config.enable_fp8 == True
        assert config.enable_dual_batch_overlap == True
        assert config.max_num_batched_tokens == 8192
    
    def test_vllm_config_generation(self):
        """Test vLLM config generation"""
        config = OptimizationConfig(model_path="test")
        vllm_config = config.get_vllm_config()
        assert "model" in vllm_config
        assert vllm_config["model"] == "test"
    
    def test_is_moe_model(self):
        """Test MoE model detection"""
        moe_config = OptimizationConfig(
            model_path="test",
            model_type="moe"
        )
        dense_config = OptimizationConfig(
            model_path="test",
            model_type="dense"
        )
        assert moe_config.is_moe_model() == True
        assert dense_config.is_moe_model() == False
    
    def test_calculate_expected_speedup(self):
        """Test speedup calculation"""
        config = OptimizationConfig(
            model_path="test",
            enable_fp8=True,
            enable_dual_batch_overlap=True,
            enable_disaggregation=True,
            enable_kv_tiering=True,
        )
        speedup = config.calculate_expected_speedup()
        assert speedup > 1.0
        # With FP8(2×) * DBO(2.3×) * Disagg(1.4×) * KV(1.4×)
        # Should be around 9-11×
        assert 9.0 <= speedup <= 11.0
    
    def test_summary_generation(self):
        """Test summary string generation"""
        config = OptimizationConfig(model_path="test")
        summary = config.summary()
        assert "MoE OPTIMIZATION CONFIGURATION" in summary
        assert "test" in summary


class TestPredefinedConfigs:
    """Test predefined configuration helpers"""
    
    def test_default_config(self):
        """Test default config creation"""
        config = get_default_config("test-model", num_gpus=3)
        assert config.model_path == "test-model"
        assert config.num_gpus == 3
        assert config.enable_fp8 == True
    
    def test_conservative_config(self):
        """Test conservative config"""
        config = get_conservative_config("test-model")
        assert config.enable_fp8 == True
        assert config.fp8_router_precision == "fp16"
        assert config.enable_disaggregation == False
        assert config.gpu_memory_utilization == 0.85
    
    def test_aggressive_config(self):
        """Test aggressive config"""
        config = get_aggressive_config("test-model")
        assert config.enable_fp8 == True
        assert config.fp8_router_precision == "fp8"
        assert config.enable_disaggregation == True
        assert config.enable_expert_sparsity == True
        assert config.gpu_memory_utilization == 0.95


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_gpu_id_validation(self):
        """Test GPU ID validation in disaggregation"""
        with pytest.raises(ValueError):
            config = OptimizationConfig(
                model_path="test",
                num_gpus=2,
                enable_disaggregation=True,
                prefill_gpu_ids=[0],
                decode_gpu_ids=[1, 2],  # GPU 2 doesn't exist
            )
    
    def test_memory_utilization_validation(self):
        """Test GPU memory utilization bounds"""
        with pytest.raises(ValueError):
            config = OptimizationConfig(
                model_path="test",
                gpu_memory_utilization=1.5,  # Invalid
            )
    
    def test_moe_sparsity_validation(self):
        """Test MoE sparsity requires num_experts"""
        with pytest.raises(ValueError):
            config = OptimizationConfig(
                model_path="test",
                model_type="moe",
                enable_expert_sparsity=True,
                num_experts=None,  # Missing
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
