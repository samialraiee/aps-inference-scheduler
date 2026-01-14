import asyncio
import httpx
import time
import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Request


async def stress_test():
    """Stress test: Tenant A (VIP, bid=10) sends 50 requests, Tenant B (Free, bid=1) sends 50 requests"""
    print("\n" + "="*80)
    print("🚀 FINAL STRESS TEST: Multi-Tenant Priority Scheduling")
    print("="*80)
    print("Scenario: Tenant A (VIP, bid=10) vs Tenant B (Free, bid=1)")
    print("Each tenant sends 50 requests concurrently")
    print("Expected: Tenant A requests should be prioritized over Tenant B")
    print()

    async with httpx.AsyncClient(base_url="http://localhost:8001") as client:
        # Create requests for Tenant A (high priority)
        tenant_a_requests = []
        for i in range(50):
            req = Request(
                tenant_id="tenant_a",
                prompt=f"VIP request {i} from Tenant A",
                tokens_requested=100,
                output_tokens_expected=50,
                priority_bid=10  # High bid
            )
            tenant_a_requests.append(req)

        # Create requests for Tenant B (low priority)
        tenant_b_requests = []
        for i in range(50):
            req = Request(
                tenant_id="tenant_b",
                prompt=f"Free request {i} from Tenant B",
                tokens_requested=100,
                output_tokens_expected=50,
                priority_bid=1  # Low bid
            )
            tenant_b_requests.append(req)

        # Send all requests concurrently
        print("📤 Sending 100 requests concurrently (50 from each tenant)...")
        start_time = time.time()

        tasks = []
        for req in tenant_a_requests + tenant_b_requests:
            task = client.post("/infer", json=req.model_dump(mode='json'))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        send_time = time.time() - start_time

        # Analyze responses
        accepted_a = 0
        accepted_b = 0
        rejected_a = 0
        rejected_b = 0

        for req, resp in zip(tenant_a_requests + tenant_b_requests, responses):
            status = resp.json()['status']
            if req.tenant_id == "tenant_a":
                if status == "queued":
                    accepted_a += 1
                else:
                    rejected_a += 1
            else:
                if status == "queued":
                    accepted_b += 1
                else:
                    rejected_b += 1

        print("📊 Initial Response Summary:")
        print(f"  Tenant A (VIP):    {accepted_a} accepted, {rejected_a} rejected")
        print(f"  Tenant B (Free):   {accepted_b} accepted, {rejected_b} rejected")
        print(f"  Total sent in {send_time:.2f}s")

        # Wait for processing and monitor metrics
        print("\n⏳ Processing requests... Monitoring metrics every 2 seconds")
        print("Expected: High Jain's fairness index due to priority scheduling")

        for i in range(15):  # Monitor for 30 seconds
            await asyncio.sleep(2)

            try:
                metrics = await client.get("/metrics")
                data = metrics.json()

                print(f"\n[{i*2:2d}s] Metrics:")
                print(f"  Throughput: {data['throughput_tokens_per_second']} t/s")
                print(f"  GPU Util:   {data['gpu_utilization_percent']}%")
                print(f"  Cost/1M:    ${data['cost_per_1M_tokens_usd']}")
                print(f"  Jain's:     {data['jains_fairness_index']}")
                print(f"  Tenants:    {data['active_tenants_tracked']}")

            except Exception as e:
                print(f"  Error fetching metrics: {e}")

        # Final health check
        print("\n🏁 Final Health Check:")
        health = await client.get("/health")
        health_data = health.json()
        print(f"  Queue size: {health_data['queue_size']}")
        print(f"  Total requests: {health_data['total_requests']}")
        print(f"  Accepted: {health_data['accepted']}")
        print(f"  Rejected: {health_data['rejected']}")
        print(f"  Rejection rate: {health_data['rejection_rate']}%")

        print("\n✅ Stress test completed!")
        print("Check server logs for priority queue behavior.")


if __name__ == "__main__":
    asyncio.run(stress_test())