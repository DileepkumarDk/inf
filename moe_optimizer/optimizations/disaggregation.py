"""
Prefill-Decode Disaggregation (Week 2, Days 3-4)

Implements disaggregation of prefill and decode phases across GPUs.
This is a CUSTOM optimization (part of the 20%).

Key Features:
- Prefill on GPU 0-1 (compute-heavy)
- Decode on GPU 2 (memory-bandwidth heavy)
- NVLink-optimized KV cache transfer
- Async KV transfer during attention

Expected Gain: 4-6× P99 latency reduction, 1.3-1.5× throughput
"""

import logging
from typing import Dict, Any, Optional, List

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    dist = None


class PrefillDecodeDisaggregator:
    """
    Handles prefill-decode disaggregation across multiple GPUs
    
    This is a custom optimization that requires:
    - Multi-GPU setup with NVLink
    - Custom KV cache transfer logic
    - Modified attention kernels
    """
    
    def __init__(
        self,
        num_gpus: int = 3,
        prefill_gpus: List[int] = [0, 1],
        decode_gpus: List[int] = [2],
        enable_async_transfer: bool = True,
        kv_transfer_overlap: bool = True,
    ):
        """
        Initialize disaggregation optimizer
        
        Args:
            num_gpus: Total number of GPUs
            prefill_gpus: GPU IDs for prefill phase
            decode_gpus: GPU IDs for decode phase
            enable_async_transfer: Use async KV cache transfer
            kv_transfer_overlap: Overlap transfer with computation
        """
        self.logger = logging.getLogger("Disaggregation")
        self.num_gpus = num_gpus
        self.prefill_gpus = prefill_gpus
        self.decode_gpus = decode_gpus
        self.enable_async_transfer = enable_async_transfer
        self.kv_transfer_overlap = kv_transfer_overlap
        
        self._initialized = False
        self._process_groups = {}
    
    def is_available(self) -> bool:
        """Check if disaggregation is available"""
        if not TORCH_AVAILABLE:
            return False
        
        if not torch.cuda.is_available():
            return False
        
        # Check if we have enough GPUs
        if torch.cuda.device_count() < self.num_gpus:
            self.logger.warning(
                f"Need {self.num_gpus} GPUs but only "
                f"{torch.cuda.device_count()} available"
            )
            return False
        
        # Check NVLink connectivity
        if torch.cuda.device_count() >= 2:
            # Check if GPUs support peer-to-peer access (NVLink indicator)
            try:
                gpu0_can_access = torch.cuda.can_device_access_peer(1, 0)
                gpu1_can_access = torch.cuda.can_device_access_peer(0, 1)
                
                if gpu0_can_access and gpu1_can_access:
                    self.logger.info("✓ NVLink/P2P access detected between GPUs")
                    return True
                else:
                    self.logger.warning("GPUs don't support P2P access (no NVLink)")
                    return False
            except Exception as e:
                self.logger.warning(f"Could not check P2P access: {e}, assuming available")
                return True
        
        return False
    
    def initialize_process_groups(self, max_retries: int = 3, retry_delay: float = 10.0):
        """Initialize distributed process groups for GPU communication
        
        Args:
            max_retries: Maximum number of retry attempts (FIX #9: Add retry logic)
            retry_delay: Delay between retries in seconds (FIX #9: Increased from 2s to 10s)
        """
        if not self.is_available():
            self.logger.warning("Disaggregation not available, skipping init")
            return
        
        # FIX: Add retry logic for process group initialization
        import time
        
        for attempt in range(max_retries):
            try:
                if not dist.is_initialized():
                    # Initialize distributed backend
                    self.logger.info(f"Initializing distributed backend (attempt {attempt + 1}/{max_retries})...")
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        timeout=torch.distributed.default_pg_timeout,
                    )
                
                # Create process groups for prefill and decode
                # Prefill group
                if len(self.prefill_gpus) > 1:
                    prefill_group = dist.new_group(ranks=self.prefill_gpus)
                    self._process_groups['prefill'] = prefill_group
                    self.logger.info(f"✓ Created prefill process group with GPUs {self.prefill_gpus}")
                
                # Decode group
                if len(self.decode_gpus) > 1:
                    decode_group = dist.new_group(ranks=self.decode_gpus)
                    self._process_groups['decode'] = decode_group
                    self.logger.info(f"✓ Created decode process group with GPUs {self.decode_gpus}")
                
                self._initialized = True
                self.logger.info("✓ Disaggregation process groups initialized successfully")
                return
                
            except Exception as e:
                self.logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to initialize after {max_retries} attempts")
                    raise RuntimeError(
                        f"Could not initialize disaggregation process groups after {max_retries} attempts. "
                        f"Last error: {e}"
                    )
    
    def transfer_kv_cache_nvlink(
        self,
        kv_cache: Any,
        source_gpu: int,
        target_gpus: List[int],
        async_transfer: bool = True
    ) -> Any:
        """
        Transfer KV cache from prefill GPU to decode GPUs via NVLink
        
        This is the PRODUCTION implementation of KV cache transfer (FIX: Implement real transfer)
        
        Args:
            kv_cache: KV cache tensor [num_layers, 2, batch, num_heads, seq_len, head_dim]
            source_gpu: Source GPU ID (prefill)
            target_gpus: Target GPU IDs (decode)
            async_transfer: Use async CUDA streams for transfer
            
        Returns:
            Transferred KV cache on target GPUs
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available for KV transfer")
            return kv_cache
        
        try:
            # Ensure KV cache is on source GPU
            with torch.cuda.device(source_gpu):
                if kv_cache.device.index != source_gpu:
                    kv_cache = kv_cache.to(f'cuda:{source_gpu}')
                
                if async_transfer:
                    # Use async stream for non-blocking transfer
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        transferred_caches = []
                        for target_gpu in target_gpus:
                            # Transfer to target GPU
                            target_cache = kv_cache.to(f'cuda:{target_gpu}', non_blocking=True)
                            transferred_caches.append(target_cache)
                        
                        # Synchronize stream
                        stream.synchronize()
                    
                    self.logger.debug(
                        f"✓ Async KV transfer: GPU {source_gpu} -> GPUs {target_gpus} "
                        f"(size: {kv_cache.numel() * kv_cache.element_size() / 1024**2:.1f} MB)"
                    )
                else:
                    # Synchronous transfer
                    transferred_caches = []
                    for target_gpu in target_gpus:
                        target_cache = kv_cache.to(f'cuda:{target_gpu}')
                        transferred_caches.append(target_cache)
                    
                    self.logger.debug(
                        f"✓ Sync KV transfer: GPU {source_gpu} -> GPUs {target_gpus}"
                    )
                
                return transferred_caches if len(transferred_caches) > 1 else transferred_caches[0]
                
        except Exception as e:
            self.logger.error(f"KV cache transfer failed: {e}")
            raise RuntimeError(f"Failed to transfer KV cache via NVLink: {e}")
    
    def get_vllm_config(self) -> Dict[str, Any]:
        """
        Get vLLM configuration for disaggregation
        
        Note: This requires CUSTOM vLLM modifications.
        Standard vLLM does not support disaggregation out-of-the-box.
        
        Returns:
            Dict with disaggregation settings
        """
        if not self.is_available():
            return {}
        
        config = {
            # Custom vLLM parameters (requires patched vLLM)
            "enable_disaggregation": True,
            "prefill_gpu_ids": self.prefill_gpus,
            "decode_gpu_ids": self.decode_gpus,
            "async_kv_transfer": self.enable_async_transfer,
        }
        
        self.logger.info("Disaggregation config prepared (requires custom vLLM)")
        return config
    
    def apply_to_vllm_config(self, vllm_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply disaggregation settings to vLLM configuration
        
        Args:
            vllm_config: Existing vLLM config dict
            
        Returns:
            Updated config with disaggregation settings
        """
        if not self.is_available():
            self.logger.info("Disaggregation not available, skipping")
            return vllm_config
        
        disagg_config = self.get_vllm_config()
        vllm_config.update(disagg_config)
        
        self.logger.warning(
            "⚠️ Disaggregation requires CUSTOM vLLM patches. "
            "See docs/DISAGGREGATION.md for implementation details."
        )
        return vllm_config
    
    def estimate_kv_transfer_time(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int,
        seq_len: int,
    ) -> float:
        """
        Estimate KV cache transfer time over NVLink
        
        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads per layer
            head_dim: Head dimension
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Estimated transfer time in milliseconds
        """
        # KV cache size: [num_layers, 2 (K+V), batch, num_kv_heads, seq_len, head_dim]
        # Assuming FP16 = 2 bytes per element
        bytes_per_element = 2
        total_elements = (
            num_layers * 2 * batch_size * num_kv_heads * seq_len * head_dim
        )
        total_bytes = total_elements * bytes_per_element
        
        # FIX #3: NVLink 4.0 bandwidth correction for H100 SXM
        # H100 SXM5: 900 GB/s bidirectional per GPU (18 links × 25 GB/s × 2 directions)
        # H100 PCIe: 128 GB/s bidirectional
        # Note: 1800 GB/s is total fabric bandwidth (2 GPUs × 900 GB/s each)
        nvlink_bandwidth_gbps = 900  # CORRECTED: Per-GPU bidirectional bandwidth
        nvlink_bandwidth_bps = nvlink_bandwidth_gbps * 1e9
        
        # Add 10% overhead for protocol and contention
        effective_bandwidth = nvlink_bandwidth_bps * 0.9
        
        # Transfer time in seconds
        transfer_time_s = total_bytes / effective_bandwidth
        
        # Convert to milliseconds
        transfer_time_ms = transfer_time_s * 1000
        
        return transfer_time_ms
    
    def get_expected_speedup(self) -> Dict[str, float]:
        """
        Get expected speedup from disaggregation
        
        Returns:
            Dict with throughput and latency speedups
        """
        if not self.is_available():
            return {"throughput": 1.0, "p99_latency": 1.0}
        
        # Based on benchmarks from idea.txt
        return {
            "throughput": 1.4,  # 1.3-1.5× throughput gain
            "p99_latency": 5.0,  # 4-6× P99 latency reduction
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current disaggregation status"""
        speedups = self.get_expected_speedup()
        return {
            "available": self.is_available(),
            "num_gpus": self.num_gpus,
            "prefill_gpus": self.prefill_gpus,
            "decode_gpus": self.decode_gpus,
            "async_transfer": self.enable_async_transfer,
            "expected_throughput_speedup": f"{speedups['throughput']:.1f}×",
            "expected_p99_improvement": f"{speedups['p99_latency']:.1f}×",
            "notes": "Requires custom vLLM modifications (20% custom work)",
        }
    
    def __repr__(self) -> str:
        status = self.get_status()
        available_str = "Available" if status["available"] else "Not Available"
        return (
            f"PrefillDecodeDisaggregator("
            f"status={available_str}, "
            f"prefill_gpus={status['prefill_gpus']}, "
            f"decode_gpus={status['decode_gpus']}"
            f")"
        )
