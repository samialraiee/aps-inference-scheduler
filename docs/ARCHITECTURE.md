# Architecture: Request Flow

This document outlines the technical flow of an inference request through the APS Inference Engine.

## Request Lifecycle

### 1. API Reception
FastAPI endpoint receives the inference request, validates the JSON payload using Pydantic models, and checks the tenant's token bucket for rate limiting. If approved, the request is converted to a priority queue entry.

### 2. Queue Processing
The request enters a priority heap with lazy aging. A background worker accumulates requests into micro-batches within a 10ms window, selecting the highest priority items for processing.

### 3. GPU Simulation
Batched requests are sent to the GPU simulator, which models A100 throughput: parallel prefill phase (1024 tokens/sec) followed by sequential decode phase (128 tokens/sec), respecting KV-cache limits.

## Key Components

- **TenantManager**: Token bucket rate limiting
- **PriorityQueue**: Heap-based scheduling with aging
- **GPUSimulator**: A100 performance modeling
- **Worker**: Async batch processing loop

## Performance Characteristics

- **Throughput**: Up to 95% GPU utilization via micro-batching
- **Latency**: Sub-100ms TTFT for high-priority requests
- **Fairness**: Prevents starvation through adaptive priorities
- **Scalability**: O(log N) queue operations, O(1) aging
