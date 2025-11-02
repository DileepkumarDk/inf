"""
Model Inspector - Auto-detect model properties

This module introspects MoE models to determine:
- Number of experts
- Model size (total parameters)
- Recommended GPU count
- Whether it's actually a MoE model
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModelInspector:
    """
    Inspect MoE models to determine configuration
    
    This helps auto-configure the optimizer for any model,
    not just Mixtral 8×7B.
    """
    
    # Known model families and their properties
    KNOWN_MODELS = {
        "mixtral": {
            "8x7b": {"num_experts": 8, "experts_per_token": 2, "size_gb": 90},
            "8x22b": {"num_experts": 8, "experts_per_token": 2, "size_gb": 281},
        },
        "deepseek": {
            "6.7b": {"num_experts": 8, "experts_per_token": 2, "size_gb": 26},
            "16b": {"num_experts": 64, "experts_per_token": 8, "size_gb": 65},
        },
        "phi": {
            "3.5-moe": {"num_experts": 16, "experts_per_token": 2, "size_gb": 14},
        },
    }
    
    @staticmethod
    def inspect_model(model_path: str) -> Dict[str, Any]:
        """
        Inspect a model and return its properties
        
        Args:
            model_path: Path to model or HuggingFace model ID
        
        Returns:
            Dict with model properties:
            {
                "is_moe": bool,
                "num_experts": int or None,
                "experts_per_token": int or None,
                "estimated_size_gb": float,
                "recommended_gpus": int,
                "architecture": str,
            }
        """
        logger.info(f"Inspecting model: {model_path}")
        
        # Try HuggingFace config first
        if TRANSFORMERS_AVAILABLE:
            try:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                return ModelInspector._inspect_from_config(config)
            except Exception as e:
                logger.warning(f"Could not load HuggingFace config: {e}")
        
        # Fallback: Try to infer from model name
        result = ModelInspector._inspect_from_name(model_path)
        
        if result["is_moe"] is None:
            logger.warning(
                "Could not determine if model is MoE. "
                "Assuming standard Transformer. "
                "Please specify num_experts manually if this is a MoE model."
            )
            result.update({
                "is_moe": False,
                "num_experts": None,
                "experts_per_token": None,
            })
        
        return result
    
    @staticmethod
    def _inspect_from_config(config) -> Dict[str, Any]:
        """Extract info from HuggingFace config"""
        result = {
            "architecture": config.architectures[0] if hasattr(config, "architectures") else "unknown",
        }
        
        # Check for MoE indicators
        is_moe = False
        num_experts = None
        experts_per_token = None
        
        # Mixtral-style
        if hasattr(config, "num_local_experts"):
            is_moe = True
            num_experts = config.num_local_experts
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
        
        # DeepSeek-style
        elif hasattr(config, "n_routed_experts"):
            is_moe = True
            num_experts = config.n_routed_experts
            experts_per_token = getattr(config, "num_experts_per_tok", 2)
        
        # Generic MoE config
        elif hasattr(config, "moe_num_experts"):
            is_moe = True
            num_experts = config.moe_num_experts
            experts_per_token = getattr(config, "moe_top_k", 2)
        
        result.update({
            "is_moe": is_moe,
            "num_experts": num_experts,
            "experts_per_token": experts_per_token,
        })
        
        # Estimate size
        if hasattr(config, "num_parameters"):
            params = config.num_parameters
        elif hasattr(config, "vocab_size") and hasattr(config, "hidden_size"):
            # Rough estimate for Transformer models
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            num_layers = getattr(config, "num_hidden_layers", 32)
            
            # params ≈ vocab_size * hidden_size * 2 + num_layers * (12 * hidden_size^2)
            params = vocab_size * hidden_size * 2 + num_layers * (12 * hidden_size * hidden_size)
            
            if is_moe and num_experts:
                # MoE: multiply FFN params by num_experts
                params = params + (num_layers * num_experts * 8 * hidden_size * hidden_size)
        else:
            params = None
        
        if params:
            # Assume FP16 = 2 bytes per param
            size_gb = (params * 2) / (1024**3)
            result["estimated_size_gb"] = size_gb
        else:
            result["estimated_size_gb"] = None
        
        # Recommend GPU count
        if result["estimated_size_gb"]:
            if result["estimated_size_gb"] < 40:
                result["recommended_gpus"] = 1  # Fits on 1×H100
            elif result["estimated_size_gb"] < 120:
                result["recommended_gpus"] = 2  # Needs 2×H100
            elif result["estimated_size_gb"] < 200:
                result["recommended_gpus"] = 3  # Needs 3×H100
            else:
                result["recommended_gpus"] = 4  # Needs 4+×H100
        else:
            result["recommended_gpus"] = 1  # Default safe
        
        logger.info(f"Model inspection complete: {result}")
        return result
    
    @staticmethod
    def _inspect_from_name(model_path: str) -> Dict[str, Any]:
        """
        Try to infer model properties from name
        
        This is a fallback when config isn't available
        """
        model_lower = model_path.lower()
        
        result = {
            "is_moe": None,
            "num_experts": None,
            "experts_per_token": None,
            "estimated_size_gb": None,
            "recommended_gpus": 1,
            "architecture": "unknown",
        }
        
        # Check known patterns
        for family, variants in ModelInspector.KNOWN_MODELS.items():
            if family in model_lower:
                for variant, props in variants.items():
                    if variant.replace(".", "") in model_lower.replace(".", ""):
                        result.update({
                            "is_moe": True,
                            "num_experts": props["num_experts"],
                            "experts_per_token": props["experts_per_token"],
                            "estimated_size_gb": props["size_gb"],
                            "architecture": f"{family}-{variant}",
                        })
                        
                        # Recommend GPUs based on size
                        if props["size_gb"] < 40:
                            result["recommended_gpus"] = 1
                        elif props["size_gb"] < 120:
                            result["recommended_gpus"] = 2
                        elif props["size_gb"] < 200:
                            result["recommended_gpus"] = 3
                        else:
                            result["recommended_gpus"] = 4
                        
                        logger.info(f"Matched known model: {family}-{variant}")
                        return result
        
        # Check for generic MoE indicators
        if any(x in model_lower for x in ["moe", "mixture", "mixtral", "expert"]):
            result["is_moe"] = True
            logger.warning("Detected MoE from name, but couldn't determine expert count")
        
        # Try to extract size from name (e.g., "7b", "13b")
        import re
        size_match = re.search(r'(\d+\.?\d*)b', model_lower)
        if size_match:
            size_b = float(size_match.group(1))
            # Rough estimate: 7B model ≈ 14GB in FP16
            result["estimated_size_gb"] = size_b * 2
            
            if result["estimated_size_gb"] < 40:
                result["recommended_gpus"] = 1
            elif result["estimated_size_gb"] < 120:
                result["recommended_gpus"] = 2
            else:
                result["recommended_gpus"] = 3
        
        return result
    
    @staticmethod
    def validate_gpu_count(required_gpus: int, model_size_gb: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate if we have enough GPUs
        
        Args:
            required_gpus: Number of GPUs requested
            model_size_gb: Estimated model size
        
        Returns:
            (is_valid, message)
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False, "No CUDA GPUs available"
        
        available_gpus = torch.cuda.device_count()
        
        if available_gpus < required_gpus:
            return False, f"Requested {required_gpus} GPUs but only {available_gpus} available"
        
        # Check GPU memory
        if model_size_gb and available_gpus > 0:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            required_memory_per_gpu = model_size_gb / required_gpus
            
            # Add 20% overhead for activations, KV cache, etc.
            required_memory_per_gpu *= 1.2
            
            if required_memory_per_gpu > gpu_memory_gb:
                return False, (
                    f"Model requires ~{required_memory_per_gpu:.1f}GB per GPU "
                    f"but GPUs only have {gpu_memory_gb:.1f}GB"
                )
        
        return True, f"✓ {available_gpus} GPU(s) available"


def auto_configure_for_model(model_path: str, override_gpus: Optional[int] = None) -> Dict[str, Any]:
    """
    Auto-configure optimization settings for a model
    
    Args:
        model_path: Path to model or HuggingFace ID
        override_gpus: Override GPU count (None = auto-detect)
    
    Returns:
        Configuration dict suitable for OptimizationConfig
    """
    inspector = ModelInspector()
    info = inspector.inspect_model(model_path)
    
    config = {
        "model_path": model_path,
        "model_type": "moe" if info["is_moe"] else "dense",  # Use "dense" not "transformer"
    }
    
    # GPU count
    if override_gpus:
        config["num_gpus"] = override_gpus
    else:
        config["num_gpus"] = info["recommended_gpus"]
    
    # MoE-specific
    if info["is_moe"]:
        config["num_experts"] = info["num_experts"]
        config["experts_per_token"] = info["experts_per_token"]
        config["enable_expert_placement"] = True
    else:
        config["enable_expert_placement"] = False
    
    # Disaggregation: only enable if 2+ GPUs
    config["enable_disaggregation"] = (config["num_gpus"] >= 2)
    
    # Memory utilization based on GPU count
    if config["num_gpus"] == 1:
        config["gpu_memory_utilization"] = 0.90  # Single GPU - use more
    else:
        config["gpu_memory_utilization"] = 0.85  # Multi-GPU - more conservative
    
    # Validate
    is_valid, msg = inspector.validate_gpu_count(
        config["num_gpus"],
        info["estimated_size_gb"]
    )
    
    if not is_valid:
        logger.error(f"GPU validation failed: {msg}")
        raise RuntimeError(f"Insufficient GPU resources: {msg}")
    
    logger.info(f"Auto-configured for {model_path}:")
    logger.info(f"  Architecture: {info['architecture']}")
    logger.info(f"  MoE: {info['is_moe']}")
    logger.info(f"  Experts: {info['num_experts']}")
    if info['estimated_size_gb'] is not None:
        logger.info(f"  Est. Size: {info['estimated_size_gb']:.1f}GB")
    else:
        logger.info(f"  Est. Size: Unknown")
    logger.info(f"  GPUs: {config['num_gpus']}")
    logger.info(f"  {msg}")
    
    return config
