from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
from asyncio import PriorityQueue
from typing import Optional, Dict, Any
import time
import random
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import heapq

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Request, TenantConfig, InferenceResponse, HeapEntry, make_heap_entry
from tenant_manager import TenantManager
from gpu_simulator import GPUSimulator


# Global state
tenant_manager = TenantManager()
gpu_simulator = GPUSimulator()
request_queue: list[HeapEntry] = []
queue_lock = asyncio.Lock()

# Per-tenant throughput tracking for Jain's fairness index
tenant_stats: Dict[str, int] = {}   # tenant_id -> total output tokens

# Constants for batching
MAX_BATCH_SIZE = 16
BATCH_WAIT_MS = 5

# Metrics
total_requests = 0
accepted_requests = 0
rejected_requests = 0


async def worker(priority_queue: list[HeapEntry], queue_lock: asyncio.Lock, gpu_simulator):
    MAX_BATCH = 16
    MAX_KV = 32768
    BATCH_WINDOW = 0.010  # 10ms

    while True:
        # Get first item (highest priority) — this is the only time we pay for re-evaluation
        async with queue_lock:
            if not priority_queue:
                await asyncio.sleep(0.001)
                continue

            # Pop the best one (may be stale — we accept it)
            entry = heapq.heappop(priority_queue)

        first_req = entry.request

        # Start micro-batching window
        batch = [first_req]
        await asyncio.sleep(BATCH_WINDOW)

        # Now grab as many as possible — again, only pay cost when popping
        async with queue_lock:
            while len(batch) < MAX_BATCH and priority_queue:
                entry = heapq.heappop(priority_queue)
                batch.append(entry.request)

        # ────────────────────────────────────────────────
        # Efficiency calculation
        used_kv = sum(r.tokens_requested for r in batch)
        efficiency = used_kv / MAX_KV
        print(f"Batch size={len(batch):2d} | efficiency={efficiency:5.1%} | "
              f"used={used_kv:5,d}/{MAX_KV:,}")

        # Process!
        result = await gpu_simulator.simulate_inference(batch)
        
        # Update per-tenant stats for Jain's fairness index
        for req in batch:
            output_tokens = getattr(req, 'output_tokens_expected', 50)
            tenant_stats[req.tenant_id] = tenant_stats.get(req.tenant_id, 0) + output_tokens


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP CODE
    default_tenants = [
        TenantConfig(tenant_id="tenant_a", rate_limit=500.0, burst_cap=5000),
        TenantConfig(tenant_id="tenant_b", rate_limit=300.0, burst_cap=3000),
        TenantConfig(tenant_id="tenant_c", rate_limit=1000.0, burst_cap=10000),
    ]
    
    for tenant in default_tenants:
        tenant_manager.register_tenant(tenant)
    
    # Start background worker
    worker_task = asyncio.create_task(worker(request_queue, queue_lock, gpu_simulator))
    
    print("[Server] Multi-Tenant AI Inference Scheduler started")
    print(f"[Server] GPU Simulator: A100 (Prefill={GPUSimulator.PREFILL_THROUGHPUT} t/s, "
          f"Decode={GPUSimulator.DECODE_THROUGHPUT} t/s)")
    print(f"[Server] Batching: Max {MAX_BATCH_SIZE} requests, {BATCH_WAIT_MS}ms wait")
    print(f"[Server] Registered {len(default_tenants)} default tenants")
    
    yield  # Server runs here
    
    # SHUTDOWN CODE
    worker_task.cancel()
    print("[Server] Shutting down...")

# Update FastAPI initialization
app = FastAPI(
    title="Multi-Tenant AI Inference Scheduler",
    lifespan=lifespan  # ← ADD THIS PARAMETER
)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: Request) -> InferenceResponse:
    """
    Main inference endpoint - The Gatekeeper
    
    Flow:
    1. Validate tenant
    2. Check rate limit (token bucket)
    3. If allowed, enqueue request
    4. If denied, reject with 429
    """
    global total_requests, accepted_requests, rejected_requests
    
    total_requests += 1
    
    try:
        # Step 1: Attempt to consume tokens from tenant's bucket
        allowed = await tenant_manager.consume(
            tenant_id=request.tenant_id,
            amount=request.tokens_requested
        )
        
        if not allowed:
            # Rate limit exceeded
            rejected_requests += 1
            return InferenceResponse(
                request_id=request.request_id,
                status="rejected",
                message=f"Rate limit exceeded for tenant {request.tenant_id}. Try again later."
            )
        
        # Step 2: Enqueue request with priority
        heap_entry = make_heap_entry(request)
        
        async with queue_lock:
            heapq.heappush(request_queue, heap_entry)
        
        accepted_requests += 1
        
        # Estimate wait time (rough approximation)
        queue_size = len(request_queue)
        estimated_wait_ms = queue_size * 50  # Assume 50ms per request
        
        print(f"[Gatekeeper] ACCEPTED request {request.request_id} from {request.tenant_id} "
              f"(bid={request.priority_bid}, queue_pos={queue_size})")
        
        return InferenceResponse(
            request_id=request.request_id,
            status="queued",
            message="Request accepted and queued for processing",
            queue_position=queue_size,
            estimated_wait_ms=estimated_wait_ms
        )
    
    except ValueError as e:
        # Tenant not registered
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "queue_size": len(request_queue),
        "queue_capacity": 1000,
        "total_requests": total_requests,
        "accepted": accepted_requests,
        "rejected": rejected_requests,
        "rejection_rate": round(rejected_requests / max(total_requests, 1) * 100, 2),
        "gpu_stats": gpu_simulator.get_stats()
    }


@app.get("/tenants/{tenant_id}/status")
async def tenant_status(tenant_id: str):
    """Get current status of a tenant's rate limit bucket"""
    try:
        return tenant_manager.get_tenant_status(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/tenants/register")
async def register_tenant(config: TenantConfig):
    """Register a new tenant with rate limiting config"""
    try:
        tenant_manager.register_tenant(config)
        return {"status": "success", "message": f"Tenant {config.tenant_id} registered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    sim_metrics = await gpu_simulator.get_metrics()

    # Cost calculation
    A100_COST_PER_HOUR = 3.00
    seconds_per_hour = 3600
    cost_per_token = (
        A100_COST_PER_HOUR / seconds_per_hour / sim_metrics["throughput_tps"]
        if sim_metrics["throughput_tps"] > 0 else 0.0
    )
    cost_per_million_tokens = cost_per_token * 1_000_000

    # Jain's Fairness Index (needs per-tenant throughput)
    # Get list of normalized throughputs (tokens processed per tenant)
    if tenant_stats:
        throughputs = list(tenant_stats.values())
        n = len(throughputs)
        if n > 0:
            sum_x = sum(throughputs)
            sum_x2 = sum(x * x for x in throughputs)
            jains_index = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 0.0
        else:
            jains_index = 0.0
    else:
        jains_index = 1.0  # perfect if no tenants yet

    return {
        "throughput_tokens_per_second": round(sim_metrics["throughput_tps"], 2),
        "total_tokens_processed": sim_metrics["total_tokens_processed"],
        "gpu_utilization_percent": round(
            (sim_metrics["total_busy_time_sec"] / sim_metrics["current_wall_time_sec"]) * 100,
            1
        ) if sim_metrics["current_wall_time_sec"] > 0 else 0.0,
        "cost_per_1M_tokens_usd": round(cost_per_million_tokens, 4),
        "jains_fairness_index": round(jains_index, 4),
        "active_tenants_tracked": len(tenant_stats),
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")