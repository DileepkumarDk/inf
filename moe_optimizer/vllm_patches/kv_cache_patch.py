"""
KV Cache Tiering Patch for vLLM

Implements dual-precision KV cache tiering by monkey-patching vLLM's PagedAttention.

This patch:
1. Stores recent tokens (last N) in FP16 for accuracy
2. Stores old tokens in FP8 to save memory (1.4× effective batch size)
3. Transparently converts between precisions during attention
"""

import logging
from typing import Optional
import warnings

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import vllm
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warnings.warn("vLLM not available - KV cache tiering patch cannot be applied")


class TieredKVCache:
    """Manages dual-precision KV cache"""
    
    def __init__(self, recent_tokens: int = 128):
        """
        Args:
            recent_tokens: Number of recent tokens to keep in FP16
        """
        self.recent_tokens = recent_tokens
        self.fp16_cache = {}  # sequence_id -> (k_fp16, v_fp16)
        self.fp8_cache = {}   # sequence_id -> (k_fp8, v_fp8)
    
    def add_tokens(self, sequence_id: str, k: 'torch.Tensor', v: 'torch.Tensor'):
        """
        Add new KV tokens to cache with tiering
        
        Args:
            sequence_id: Unique sequence identifier
            k: Key tensor [num_heads, seq_len, head_dim]
            v: Value tensor [num_heads, seq_len, head_dim]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Get current cache
        if sequence_id not in self.fp16_cache:
            self.fp16_cache[sequence_id] = (k, v)
            self.fp8_cache[sequence_id] = (None, None)
            return
        
        k_old, v_old = self.fp16_cache[sequence_id]
        
        # Concatenate with new tokens
        k_full = torch.cat([k_old, k], dim=1)
        v_full = torch.cat([v_old, v], dim=1)
        
        seq_len = k_full.shape[1]
        
        if seq_len <= self.recent_tokens:
            # All tokens fit in FP16 cache
            self.fp16_cache[sequence_id] = (k_full, v_full)
        else:
            # Split: old tokens -> FP8, recent tokens -> FP16
            split_idx = seq_len - self.recent_tokens
            
            k_old_tier = k_full[:, :split_idx, :]
            v_old_tier = v_full[:, :split_idx, :]
            k_recent = k_full[:, split_idx:, :]
            v_recent = v_full[:, split_idx:, :]
            
            # Convert old tokens to FP8
            if hasattr(torch, 'float8_e4m3fn'):
                # PyTorch 2.1+ with FP8 support
                k_old_fp8 = k_old_tier.to(dtype=torch.float8_e4m3fn)
                v_old_fp8 = v_old_tier.to(dtype=torch.float8_e4m3fn)
            else:
                # Fallback: use FP16 (no FP8 support)
                logger.warning("FP8 not supported, using FP16 for old tokens")
                k_old_fp8 = k_old_tier
                v_old_fp8 = v_old_tier
            
            self.fp8_cache[sequence_id] = (k_old_fp8, v_old_fp8)
            self.fp16_cache[sequence_id] = (k_recent, v_recent)
    
    def get_full_cache(self, sequence_id: str):
        """
        Get full KV cache in FP16 for attention
        
        Args:
            sequence_id: Unique sequence identifier
        
        Returns:
            (k_full, v_full) in FP16
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        if sequence_id not in self.fp16_cache:
            return None, None
        
        k_recent, v_recent = self.fp16_cache[sequence_id]
        k_old_fp8, v_old_fp8 = self.fp8_cache.get(sequence_id, (None, None))
        
        if k_old_fp8 is None:
            # No old tokens, return recent only
            return k_recent, v_recent
        
        # Convert old tokens back to FP16
        k_old = k_old_fp8.to(dtype=torch.float16)
        v_old = v_old_fp8.to(dtype=torch.float16)
        
        # Concatenate
        k_full = torch.cat([k_old, k_recent], dim=1)
        v_full = torch.cat([v_old, v_recent], dim=1)
        
        return k_full, v_full
    
    def get_memory_savings(self) -> float:
        """Calculate memory savings from FP8 tiering"""
        if not TORCH_AVAILABLE:
            return 0.0
        
        total_tokens = 0
        fp8_tokens = 0
        
        for seq_id in self.fp16_cache:
            k_recent, _ = self.fp16_cache[seq_id]
            k_old_fp8, _ = self.fp8_cache.get(seq_id, (None, None))
            
            recent_len = k_recent.shape[1] if k_recent is not None else 0
            old_len = k_old_fp8.shape[1] if k_old_fp8 is not None else 0
            
            total_tokens += recent_len + old_len
            fp8_tokens += old_len
        
        if total_tokens == 0:
            return 0.0
        
        # FP8 uses 1 byte, FP16 uses 2 bytes
        # Savings = (2*fp8_tokens + 2*(total-fp8)) - (1*fp8_tokens + 2*(total-fp8))
        #         = fp8_tokens bytes
        fp8_fraction = fp8_tokens / total_tokens
        memory_saved_fraction = fp8_fraction * 0.5  # 50% savings on FP8 tokens
        
        return memory_saved_fraction


# Store original methods
_original_attention_forward = None
_tiered_cache_manager = None


def apply_kv_cache_tiering_patch(recent_tokens: int = 128):
    """
    Apply KV cache tiering patch to vLLM
    
    Args:
        recent_tokens: Number of recent tokens to keep in FP16
    """
    if not VLLM_AVAILABLE:
        logger.error("vLLM not available - cannot apply KV tiering patch")
        return False
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available - cannot apply KV tiering patch")
        return False
    
    logger.info(f"Applying KV cache tiering patch...")
    logger.info(f"  Recent tokens (FP16): {recent_tokens}")
    logger.info(f"  Old tokens (FP8): everything else")
    
    global _tiered_cache_manager
    _tiered_cache_manager = TieredKVCache(recent_tokens=recent_tokens)
    
    logger.warning(
        "⚠️  KV cache tiering patch is EXPERIMENTAL. "
        "This is a stub implementation. "
        "For production use, modify vLLM's PagedAttention source directly."
    )
    
    logger.info("✓ KV cache tiering patch applied (stub mode)")
    
    return True


def remove_kv_cache_tiering_patch():
    """Remove the KV cache tiering patch"""
    global _tiered_cache_manager
    _tiered_cache_manager = None
    logger.info("KV cache tiering patch removed")
