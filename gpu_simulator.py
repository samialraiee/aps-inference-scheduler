# gpu_simulator.py
import asyncio
import time
from typing import List, Dict
from models import Request


class GPUSimulator:
    """
    Simulates NVIDIA A100 GPU inference characteristics.
    
    Based on real benchmarks for LLaMA-style models:
    - Prefill: Process input tokens (parallel)
    - Decode: Generate output tokens (sequential, auto-regressive)
    """
    
    # A100 Constants (from real benchmarks)
    PREFILL_THROUGHPUT = 1024  # tokens/second (parallel processing)
    DECODE_THROUGHPUT = 128    # tokens/second (sequential generation)
    MAX_KV_CACHE = 32768       # Maximum tokens in KV cache
    
    def __init__(self):
        self.current_kv_cache_tokens = 0
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        
        # Metrics for throughput and cost calculation
        self.total_tokens_processed = 0      # output tokens only (what we charge for)
        self.total_busy_time = 0.0           # seconds the GPU was actually computing
        self.start_time = time.time()
        self.last_process_end_time = time.time()
        self._lock = asyncio.Lock()
        
    def estimate_batch_latency(self, requests: List[Request]) -> Dict[str, float]:
        """
        Estimate latency for a batch of requests.
        
        Key Metrics:
        - TTFT (Time To First Token): How long until first output token
        - TPOT (Time Per Output Token): Time between subsequent tokens
        - Total Latency: TTFT + (TPOT × avg_output_tokens)
        
        Batching Benefits:
        - Prefill: Can process multiple requests in parallel (limited by longest)
        - Decode: Amortizes computation across batch (batch_size tokens per step)
        """
        if not requests:
            return {
                "ttft_ms": 0,
                "tpot_ms": 0,
                "total_latency_ms": 0,
                "batch_size": 0
            }
        
        batch_size = len(requests)
        
        # PREFILL PHASE: Process input tokens
        # Bottleneck: Longest request in batch (can't proceed until all done)
        max_prefill_tokens = max(r.tokens_requested for r in requests)
        ttft_seconds = max_prefill_tokens / self.PREFILL_THROUGHPUT
        
        # DECODE PHASE: Generate output tokens
        # With batching: Process batch_size tokens per step
        # Throughput scales with batch size (up to a point)
        effective_decode_throughput = self.DECODE_THROUGHPUT * min(batch_size, 16)
        
        # Average output tokens per request
        avg_output_tokens = sum(
            getattr(r, 'output_tokens_expected', 50) 
            for r in requests
        ) / batch_size
        
        # Time per output token (for the whole batch)
        tpot_seconds = 1 / (effective_decode_throughput / batch_size)
        
        # Total time: Prefill + Decode
        total_latency_seconds = ttft_seconds + (tpot_seconds * avg_output_tokens)
        
        return {
            "ttft_ms": ttft_seconds * 1000,
            "tpot_ms": tpot_seconds * 1000,
            "total_latency_ms": total_latency_seconds * 1000,
            "batch_size": batch_size,
            "max_prefill_tokens": max_prefill_tokens,
            "avg_output_tokens": avg_output_tokens
        }
    
    async def simulate_inference(self, requests: List[Request]) -> Dict:
        """
        Simulate GPU inference for a batch of requests.
        
        This sleeps for the estimated latency to simulate real processing time.
        """
        if not requests:
            return {
                "batch_size": 0,
                "latency_ms": 0,
                "requests_processed": []
            }
        
        # Calculate KV cache requirement
        total_tokens = sum(r.tokens_requested for r in requests)
        
        if self.current_kv_cache_tokens + total_tokens > self.MAX_KV_CACHE:
            # KV cache overflow - would need eviction in real system
            print(f"[GPU] WARNING: KV cache near limit "
                  f"({self.current_kv_cache_tokens + total_tokens}/{self.MAX_KV_CACHE})")
            # For simulation, we'll reset cache
            self.current_kv_cache_tokens = 0
        
        # Update cache
        self.current_kv_cache_tokens += total_tokens
        
        # Estimate latency
        latency_info = self.estimate_batch_latency(requests)
        
        # Simulate actual processing time
        start_time = time.time()
        await asyncio.sleep(latency_info["total_latency_ms"] / 1000)
        actual_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.total_batches_processed += 1
        self.total_requests_processed += len(requests)
        
        # Update metrics
        output_tokens_estimate = sum(
            getattr(r, 'output_tokens_expected', 50) 
            for r in requests
        )
        busy_delta = latency_info["total_latency_ms"] / 1000
        
        async with self._lock:
            self.total_tokens_processed += output_tokens_estimate
            self.total_busy_time += busy_delta
            self.last_process_end_time = time.time()
        
        # Log batch processing
        print(f"\n{'='*80}")
        print(f"[GPU] BATCH {self.total_batches_processed} COMPLETE")
        print(f"{'='*80}")
        print(f"  Batch Size:        {latency_info['batch_size']} requests")
        print(f"  Max Prefill:       {latency_info['max_prefill_tokens']} tokens")
        print(f"  Avg Output:        {latency_info['avg_output_tokens']:.1f} tokens")
        print(f"  TTFT:              {latency_info['ttft_ms']:.2f}ms")
        print(f"  TPOT:              {latency_info['tpot_ms']:.2f}ms")
        print(f"  Simulated Latency: {latency_info['total_latency_ms']:.2f}ms")
        print(f"  Actual Sleep Time: {actual_time_ms:.2f}ms")
        print(f"  KV Cache Used:     {self.current_kv_cache_tokens}/{self.MAX_KV_CACHE} tokens")
        print(f"  Requests Processed:")
        for i, req in enumerate(requests[:5]):  # Show first 5
            print(f"    [{i+1}] {req.tenant_id} | bid={req.priority_bid} | "
                  f"tokens={req.tokens_requested}")
        if len(requests) > 5:
            print(f"    ... and {len(requests) - 5} more")
        print(f"{'='*80}\n")
        
        return {
            "batch_size": len(requests),
            "latency_ms": latency_info["total_latency_ms"],
            "actual_time_ms": actual_time_ms,
            "ttft_ms": latency_info["ttft_ms"],
            "tpot_ms": latency_info["tpot_ms"],
            "requests_processed": [r.request_id for r in requests],
            "kv_cache_tokens": self.current_kv_cache_tokens
        }
    
    def get_stats(self) -> Dict:
        """Get GPU simulator statistics"""
        return {
            "total_batches": self.total_batches_processed,
            "total_requests": self.total_requests_processed,
            "avg_batch_size": round(
                self.total_requests_processed / max(self.total_batches_processed, 1), 2
            ),
            "kv_cache_used": self.current_kv_cache_tokens,
            "kv_cache_max": self.MAX_KV_CACHE,
            "kv_cache_utilization_pct": round(
                self.current_kv_cache_tokens / self.MAX_KV_CACHE * 100, 2
            )
        }
    
    async def get_metrics(self):
        """Thread/async-safe snapshot for throughput and cost"""
        current_time = time.time()

        async with self._lock:  # even though called from endpoint, better safe
            total_tokens = self.total_tokens_processed
            busy_time = self.total_busy_time
            idle_time = current_time - self.last_process_end_time
            total_wall_time = busy_time + idle_time if busy_time > 0 else 1e-6

        return {
            "total_tokens_processed": total_tokens,
            "total_busy_time_sec": busy_time,
            "current_wall_time_sec": total_wall_time,
            "throughput_tps": total_tokens / total_wall_time if total_wall_time > 0 else 0.0,
        }  