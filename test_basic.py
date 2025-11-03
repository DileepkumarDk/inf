"""
Quick test script to verify the MoE optimizer works without GPU

This script can run on a laptop without GPU and validates the code structure.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from moe_optimizer import OptimizedMoEEngine, OptimizationConfig


def test_config_creation():
    """Test 1: Configuration creation"""
    print("\n" + "="*70)
    print("TEST 1: Configuration Creation")
    print("="*70)
    
    config = OptimizationConfig(
        model_path="test/model/path",  # Generic path for testing
        model_type="moe",
        num_gpus=3,
        enable_fp8=True,
        enable_dual_batch_overlap=True,
        enable_disaggregation=True,
    )
    
    print(config.summary())
    print("\n✓ Config created successfully")
    
    return config


def test_engine_creation(config):
    """Test 2: Engine creation (without GPU)"""
    print("\n" + "="*70)
    print("TEST 2: Engine Creation (CPU-only mode)")
    print("="*70)
    
    engine = OptimizedMoEEngine(config)
    print("\n✓ Engine created successfully")
    
    return engine


def test_health_check(engine):
    """Test 3: Health check"""
    print("\n" + "="*70)
    print("TEST 3: Health Check")
    print("="*70)
    
    health = engine.health_check()
    
    print("\nHealth Status:")
    print(f"  Overall: {health['status']}")
    print("\nChecks:")
    for check, result in health['checks'].items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}: {result}")
    
    return health


def test_mock_generation(engine):
    """Test 4: Mock generation (works without GPU)"""
    print("\n" + "="*70)
    print("TEST 4: Mock Generation (CPU-only)")
    print("="*70)
    
    # NOTE: This won't actually load a model, just tests the API
    prompts = [
        "Explain quantum computing",
        "Write Python code for sorting",
    ]
    
    try:
        # Try to initialize (will fail gracefully without GPU)
        engine.initialize()
    except Exception as e:
        print(f"\nExpected error (no GPU): {type(e).__name__}")
        print("This is normal on CPU-only machines.")
    
    # Test mock generation
    print("\nTesting mock generation API...")
    results = engine._generate_mock(prompts, max_tokens=50)
    
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Prompt: {result['prompt'][:50]}...")
        print(f"    Output: {result['text']}")
        print(f"    Mode: {result['mode']}")
    
    print("\n✓ Mock generation works")


def test_stats(engine):
    """Test 5: Get stats"""
    print("\n" + "="*70)
    print("TEST 5: Engine Statistics")
    print("="*70)
    
    stats = engine.get_stats()
    
    print("\nEngine Stats:")
    print(f"  Initialized: {stats['initialized']}")
    print(f"  Model type: {stats['config']['model_type']}")
    print(f"  Num GPUs: {stats['config']['num_gpus']}")
    print(f"  FP8 enabled: {stats['config']['fp8_enabled']}")
    print(f"  DBO enabled: {stats['config']['dbo_enabled']}")
    print(f"  Disaggregation: {stats['config']['disaggregation_enabled']}")
    
    if 'gpus' in stats:
        print("\nGPU Info:")
        for gpu_name, gpu_info in stats['gpus'].items():
            print(f"  {gpu_name}: {gpu_info['name']}")
    else:
        print("\n  No GPUs detected (normal on laptop)")


def test_predefined_configs():
    """Test 6: Predefined configurations"""
    print("\n" + "="*70)
    print("TEST 6: Predefined Configurations")
    print("="*70)
    
    from moe_optimizer.core.config import (
        get_default_config,
        get_conservative_config,
        get_aggressive_config
    )
    
    model_path = "test/model"
    
    configs = {
        "Default": get_default_config(model_path),
        "Conservative": get_conservative_config(model_path),
        "Aggressive": get_aggressive_config(model_path),
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  FP8: {config.enable_fp8}")
        print(f"  DBO: {config.enable_dual_batch_overlap}")
        print(f"  Disaggregation: {config.enable_disaggregation}")
        print(f"  Expert Sparsity: {config.enable_expert_sparsity}")
        print(f"  GPU Memory: {config.gpu_memory_utilization:.0%}")
    
    print("\n✓ All predefined configs work")


def main():
    """Run all tests"""
    print("="*70)
    print("MoE OPTIMIZER - CPU-ONLY TESTING")
    print("="*70)
    print("\nThis test validates the code structure without requiring GPU.")
    print("Full functionality requires GPU with vLLM installed.\n")
    
    try:
        # Test 1: Config
        config = test_config_creation()
        
        # Test 2: Engine
        engine = test_engine_creation(config)
        
        # Test 3: Health
        health = test_health_check(engine)
        
        # Test 4: Mock generation
        test_mock_generation(engine)
        
        # Test 5: Stats
        test_stats(engine)
        
        # Test 6: Predefined configs
        test_predefined_configs()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Install PyTorch: pip install torch")
        print("  2. Install vLLM: pip install vllm")
        print("  3. Rent a GPU instance (3× H100 recommended)")
        print("  4. Run full tests with GPU")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
