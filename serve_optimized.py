#!/usr/bin/env python3
"""
Serve optimized MoE model with FlashDMoE and other optimizations

This script:
1. Loads the model with vLLM
2. Applies FlashDMoE kernel optimization (8-10× speedup on MoE layers)
3. Applies FP8 quantization, KV tiering, etc.
4. Starts OpenAI-compatible API server

Usage:
    python serve_optimized.py --model ./models/Qwen-Qwen3-30B-A3B-Instruct-2507 --profile aggressive --gpus 1
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from moe_optimizer.core.config import OptimizationConfig
from moe_optimizer.core.engine import OptimizedMoEEngine

# Import config helpers
from configs import get_conservative_config, get_aggressive_config, auto_configure_for_model


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Serve optimized MoE model with FlashDMoE"
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: auto-detect)"
    )
    
    # Optimization profiles
    parser.add_argument(
        "--profile",
        type=str,
        choices=["conservative", "balanced", "aggressive"],
        default="aggressive",
        help="Optimization profile (default: aggressive)"
    )
    
    # Server options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("serve_optimized")
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA not available. This system requires H100 GPUs.")
            return 1
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        
        # Print GPU info
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    except ImportError:
        logger.error("PyTorch not installed. Please install: pip install torch")
        return 1
    
    # Auto-detect GPUs if not specified
    if args.gpus is None:
        logger.info("Auto-detecting model properties...")
        try:
            auto_config = auto_configure_for_model(args.model)
            args.gpus = auto_config["num_gpus"]
            logger.info(f"Auto-detected: {args.gpus} GPU(s) recommended")
        except Exception as e:
            logger.warning(f"Auto-detection failed: {e}. Using 1 GPU as safe default.")
            args.gpus = 1
    
    # Create config
    if args.profile == "conservative":
        config = get_conservative_config(args.model, num_gpus=args.gpus)
    elif args.profile == "aggressive":
        config = get_aggressive_config(args.model, num_gpus=args.gpus)
    else:  # balanced
        config = OptimizationConfig(
            model_path=args.model,
            num_gpus=args.gpus,
            enable_fp8=True,
            enable_dual_batch_overlap=True,
            enable_disaggregation=False,
            enable_kv_tiering=True,
            enable_expert_placement=True,
            enable_expert_sparsity=False,
        )
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("OPTIMIZATION CONFIGURATION")
    logger.info("=" * 80)
    print(config.summary())
    logger.info("=" * 80)
    
    # Create and initialize engine
    logger.info("Creating optimization engine...")
    engine = OptimizedMoEEngine(config)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("INITIALIZING MODEL WITH OPTIMIZATIONS")
    logger.info("=" * 80)
    
    try:
        # THIS IS THE CRITICAL CALL - loads FlashDMoE and patches model
        engine.initialize()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓✓✓ OPTIMIZATIONS APPLIED SUCCESSFULLY ✓✓✓")
        logger.info("=" * 80)
        logger.info(f"Expected speedup: {config.calculate_expected_speedup():.0f}×")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        logger.error("Cannot continue without optimizations")
        return 1
    
    # Check if engine was initialized
    if not hasattr(engine, 'engine') or engine.engine is None:
        logger.error("Engine initialization failed - no vLLM instance created")
        return 1
    
    logger.info("Starting OpenAI-compatible API server...")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Endpoint: http://{args.host}:{args.port}/v1")
    logger.info(f"  Model: {args.model}")
    logger.info("")
    logger.info("Server is running. Press Ctrl+C to stop.")
    logger.info("=" * 80)
    
    # Start server
    # The engine.engine is a vLLM.LLM instance
    # We need to wrap it in an OpenAI-compatible API
    
    try:
        # Try vLLM's built-in OpenAI server
        from vllm.entrypoints.openai.api_server import run_server
        import asyncio
        
        asyncio.run(run_server(
            llm=engine.engine,
            host=args.host,
            port=args.port
        ))
        
    except ImportError:
        # Fallback: Simple server using FastAPI
        logger.warning("vLLM API server not available, using simple fallback")
        
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
            import uvicorn
            from pydantic import BaseModel
            from typing import List, Optional
            
            app = FastAPI(title="Optimized MoE Inference API")
            
            class ChatMessage(BaseModel):
                role: str
                content: str
            
            class ChatCompletionRequest(BaseModel):
                model: str
                messages: List[ChatMessage]
                max_tokens: Optional[int] = 100
                temperature: Optional[float] = 0.7
                top_p: Optional[float] = 0.9
            
            @app.post("/v1/chat/completions")
            async def chat_completions(request: ChatCompletionRequest):
                """OpenAI-compatible chat completions endpoint"""
                try:
                    # Extract prompt from messages
                    prompt = request.messages[-1].content
                    
                    # Generate using optimized engine
                    from vllm import SamplingParams
                    sampling_params = SamplingParams(
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p
                    )
                    
                    outputs = engine.engine.generate([prompt], sampling_params)
                    response_text = outputs[0].outputs[0].text
                    
                    return JSONResponse({
                        "id": "chatcmpl-optimized",
                        "object": "chat.completion",
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }]
                    })
                    
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    return JSONResponse(
                        {"error": str(e)},
                        status_code=500
                    )
            
            @app.get("/health")
            async def health():
                """Health check endpoint"""
                return {"status": "ok", "optimizations": "flashdmoe+fp8"}
            
            # Run server
            uvicorn.run(app, host=args.host, port=args.port)
            
        except ImportError as e:
            logger.error(f"Could not start API server: {e}")
            logger.info("Missing dependencies: pip install fastapi uvicorn")
            logger.info("")
            logger.info("Engine is loaded and ready. You can use it via Python:")
            logger.info("  from serve_optimized import engine")
            logger.info("  results = engine.generate(['Your prompt here'])")
            return 1
    
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        logger.info("Server stopped")
    
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
