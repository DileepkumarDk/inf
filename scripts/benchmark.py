#!/usr/bin/env python
"""
Batch-Aware Benchmark Script

CRITICAL: The 1000× speedup ONLY happens at high batch sizes!
This script tests at multiple concurrency levels to show the true scaling.

Usage:
    # Test single request (low latency)
    python benchmark.py --url http://localhost:8000 --batch 1
    
    # Test high throughput (high batch)
    python benchmark.py --url http://localhost:8000 --batch 512
    
    # Test all batch sizes (recommended)
    python benchmark.py --url http://localhost:8000 --test-all-batches
"""

import argparse
import asyncio
import json
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import sys

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    print("❌ aiohttp not installed. Install with: pip install aiohttp")
    AIOHTTP_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    batch_size: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_time_seconds: float
    throughput_tps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    requests_per_second: float


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """Send a single completion request"""
    payload = {
        "model": "current-model",  # Will use whatever model is loaded
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    start_time = time.time()
    try:
        async with session.post(f"{url}/v1/completions", json=payload) as response:
            result = await response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token count
            tokens = result.get("usage", {}).get("completion_tokens", max_tokens)
            
            return {
                "success": True,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "latency_ms": (time.time() - start_time) * 1000,
            "tokens": 0,
            "error": str(e)
        }


async def run_benchmark_batch(
    url: str,
    batch_size: int,
    num_requests: int = 100,
    max_tokens: int = 100
) -> BenchmarkResult:
    """
    Run benchmark with specified batch size (concurrent requests)
    
    Args:
        url: API endpoint URL
        batch_size: Number of concurrent requests
        num_requests: Total requests to send
        max_tokens: Tokens per request
    """
    print(f"\n{'='*70}")
    print(f"🔥 BATCH SIZE: {batch_size} concurrent requests")
    print(f"{'='*70}")
    
    # Sample prompts (diverse to avoid caching)
    prompts = [
        "def fibonacci(n):",
        "Write a Python function to sort a list:",
        "Explain quantum computing in simple terms:",
        "def binary_search(arr, target):",
        "What are the benefits of async programming?",
        "class Node:",
        "Implement a hash table in Python:",
        "Explain the difference between TCP and UDP:",
    ]
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        all_results = []
        start_time = time.time()
        
        # Send requests in batches
        for batch_start in range(0, num_requests, batch_size):
            batch_end = min(batch_start + batch_size, num_requests)
            batch_tasks = []
            
            for i in range(batch_start, batch_end):
                prompt = prompts[i % len(prompts)]
                task = send_request(session, url, prompt, max_tokens)
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            
            # Progress indicator
            completed = len(all_results)
            print(f"  Progress: {completed}/{num_requests} requests ({completed*100//num_requests}%)", end="\r")
        
        total_time = time.time() - start_time
        print(f"\n  ✓ Completed {num_requests} requests in {total_time:.2f}s")
    
    # Calculate statistics
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    
    latencies = [r["latency_ms"] for r in successful]
    total_tokens = sum(r["tokens"] for r in successful)
    
    if not latencies:
        print("  ❌ All requests failed!")
        return BenchmarkResult(
            batch_size=batch_size,
            total_requests=num_requests,
            successful_requests=0,
            failed_requests=len(failed),
            total_tokens=0,
            total_time_seconds=total_time,
            throughput_tps=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            latency_mean_ms=0,
            requests_per_second=0,
        )
    
    latencies.sort()
    p50 = latencies[len(latencies) * 50 // 100]
    p95 = latencies[len(latencies) * 95 // 100]
    p99 = latencies[len(latencies) * 99 // 100]
    mean = statistics.mean(latencies)
    
    throughput = total_tokens / total_time
    rps = len(successful) / total_time
    
    result = BenchmarkResult(
        batch_size=batch_size,
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_tokens=total_tokens,
        total_time_seconds=total_time,
        throughput_tps=throughput,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        latency_mean_ms=mean,
        requests_per_second=rps,
    )
    
    # Print results
    print(f"\n  📊 RESULTS:")
    print(f"     Throughput:        {throughput:,.0f} tokens/sec")
    print(f"     Requests/sec:      {rps:.1f}")
    print(f"     Latency (P50):     {p50:.1f} ms")
    print(f"     Latency (P95):     {p95:.1f} ms")
    print(f"     Latency (P99):     {p99:.1f} ms")
    print(f"     Success rate:      {len(successful)}/{num_requests} ({len(successful)*100//num_requests}%)")
    if failed:
        print(f"     ⚠️  Failures:        {len(failed)}")
        error_msg = str(failed[0]['error']) if failed[0]['error'] else "Unknown error"
        print(f"        First error:    {error_msg[:60]}")
    
    return result


def print_comparison_table(results: List[BenchmarkResult]):
    """Print comparison table across batch sizes"""
    print(f"\n\n{'='*90}")
    print("📊 BATCH SIZE COMPARISON - THIS SHOWS WHY BATCH MATTERS!")
    print(f"{'='*90}")
    print(f"{'Batch Size':<12} {'Throughput':<18} {'Latency P50':<15} {'Latency P99':<15} {'Speedup':<10}")
    print(f"{'-'*90}")
    
    baseline_throughput = results[0].throughput_tps if results else 1
    
    for r in results:
        speedup = r.throughput_tps / baseline_throughput
        print(f"{r.batch_size:<12} {r.throughput_tps:>10,.0f} tok/s   {r.latency_p50_ms:>8.1f} ms    {r.latency_p99_ms:>8.1f} ms    {speedup:>5.1f}×")
    
    print(f"{'='*90}")
    print("\n💡 KEY INSIGHT:")
    print(f"   Batch 1:   Low latency, LOW throughput (good for chat)")
    print(f"   Batch 512: High latency, HIGH throughput (good for batch jobs)")
    print(f"   The 1000× speedup claim uses batch 512+ with all optimizations!")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Batch-aware benchmark for MoE optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API endpoint URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size (concurrent requests). If not set, tests multiple batches."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests (default: 100)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Tokens per request (default: 100)"
    )
    parser.add_argument(
        "--test-all-batches",
        action="store_true",
        help="Test multiple batch sizes: 1, 4, 16, 64, 256, 512"
    )
    parser.add_argument(
        "--save",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    if not AIOHTTP_AVAILABLE:
        print("Install aiohttp first: pip install aiohttp")
        return 1
    
    # Determine batch sizes to test
    if args.test_all_batches:
        batch_sizes = [1, 4, 16, 64, 256, 512]
    elif args.batch:
        batch_sizes = [args.batch]
    else:
        # Default: test key batch sizes
        batch_sizes = [1, 64, 256]
    
    print("\n" + "="*90)
    print("🚀 MoE OPTIMIZATION BENCHMARK")
    print("="*90)
    print(f"   URL:          {args.url}")
    print(f"   Batch sizes:  {batch_sizes}")
    print(f"   Requests:     {args.requests} per batch size")
    print(f"   Tokens/req:   {args.max_tokens}")
    print()
    print("⚠️  IMPORTANT: Make sure your API server is running!")
    print("   Example: python -m vllm.entrypoints.openai.api_server --model <model>")
    print()
    
    # Test server connectivity
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    models = await response.json()
                    print(f"✓ Server is running")
                    print(f"  Models: {models.get('data', [])}")
                else:
                    print(f"⚠️  Server returned status {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure server is running at {args.url}")
        return 1
    
    # Run benchmarks
    results = []
    for batch_size in batch_sizes:
        result = await run_benchmark_batch(
            args.url,
            batch_size,
            args.requests,
            args.max_tokens
        )
        results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Print comparison
    if len(results) > 1:
        print_comparison_table(results)
    
    # Save results
    if args.save:
        results_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "url": args.url,
            "requests_per_batch": args.requests,
            "max_tokens": args.max_tokens,
            "results": [
                {
                    "batch_size": r.batch_size,
                    "throughput_tps": r.throughput_tps,
                    "latency_p50_ms": r.latency_p50_ms,
                    "latency_p99_ms": r.latency_p99_ms,
                    "success_rate": r.successful_requests / r.total_requests,
                }
                for r in results
            ]
        }
        
        with open(args.save, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\n✓ Results saved to {args.save}")
    
    return 0


if __name__ == "__main__":
    if not AIOHTTP_AVAILABLE:
        sys.exit(1)
    
    sys.exit(asyncio.run(main()))
