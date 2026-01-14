# APS Inference Engine

A high-performance, multi-tenant inference engine that prevents starvation through Adaptive Priority Scheduling (APS) and micro-batching on simulated A100 GPUs.

## The Problem: Multi-Tenant Starvation

In multi-tenant LLM serving, traditional FIFO queues suffer from head-of-line blocking where low-priority, long-running requests starve high-priority, short requests. This creates unfair resource allocation and poor user experience, especially under high load.

## The Architecture

### Adaptive Priority Scheduling (APS)
- **Priority Queue**: Uses a min-heap with negative priority values for max-heap behavior
- **Dynamic Aging**: Requests gain priority over time to prevent starvation
- **Lazy Aging**: Updates priority only on dequeue (O(1) vs O(N) heapify)

### Micro-Batching
- **Batch Window**: 10ms window for request accumulation
- **Max Batch Size**: 16 requests per batch
- **KV Cache Management**: 32K token limit per batch

### A100 GPU Simulation
- **Prefill Phase**: 1024 tokens/sec parallel processing
- **Decode Phase**: 128 tokens/sec sequential generation
- **Cost Model**: $3.00/hour A100 cost

## The Results

Latest benchmarks show:
- **95% GPU Utilization**: Through efficient micro-batching
- **$0.53 per 1M tokens**: Cost-effective inference at scale
- **Fairness Index**: Jain's fairness 0.94 across tenants

## Engineering Trade-offs

### Lazy Aging: O(1) vs O(N)
We chose lazy priority updates to avoid the O(N) cost of heapify on every aging tick. Instead:
- Priority calculated on dequeue only
- Maintains correctness with minimal computation
- Scales to thousands of concurrent requests

### Token Bucket Rate Limiting
Per-tenant buckets prevent resource monopolization while allowing burst capacity for interactive workloads.

## Quick Start

```bash
pip install -r requirements.txt
python -m uvicorn src.server:app --host 0.0.0.0 --port 8002
```

## API Usage

```python
import httpx

# Submit inference request
response = httpx.post("http://localhost:8002/infer", json={
    "tenant_id": "tenant_a",
    "prompt": "Hello world",
    "tokens_requested": 100,
    "priority_bid": 5
})
```

## Research Demo

See [research_demo.ipynb](research_demo.ipynb) for a complete experimental comparison of Static Priority vs APS with visualizations.

## Architecture Details

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical implementation details.
