import asyncio
import time
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TenantConfig
from collections import defaultdict


class TenantManager:
    """
    Manages token bucket rate limiting per tenant.
    
    Economics Layer: Each tenant has a budget (tokens) that refills over time.
    This prevents any single tenant from monopolizing resources.
    """
    
    def __init__(self):
        self._configs: Dict[str, TenantConfig] = {}
        # Store: {tenant_id: (current_tokens, last_update_time)}
        self._buckets: Dict[str, Tuple[float, float]] = {}
        # One lock per tenant to prevent race conditions
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    def register_tenant(self, config: TenantConfig) -> None:
        """Register a new tenant with rate limiting config"""
        self._configs[config.tenant_id] = config
        # Initialize bucket to full capacity
        self._buckets[config.tenant_id] = (float(config.burst_cap), time.time())
        print(f"[TenantManager] Registered tenant {config.tenant_id}: "
              f"{config.rate_limit} tokens/sec, burst={config.burst_cap}")
    
    def _refill_bucket(self, tenant_id: str) -> float:
        """
        Calculate current token count using token bucket algorithm.
        
        Formula: CurrentTokens = min(BurstCap, LastTokens + (Now - LastUpdate) × RateLimit)
        """
        if tenant_id not in self._configs:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        config = self._configs[tenant_id]
        last_tokens, last_update = self._buckets[tenant_id]
        
        now = time.time()
        elapsed = now - last_update
        
        # Refill tokens based on elapsed time and rate limit
        new_tokens = last_tokens + (elapsed * config.rate_limit)
        # Cap at burst capacity
        current_tokens = min(config.burst_cap, new_tokens)
        
        # Update bucket
        self._buckets[tenant_id] = (current_tokens, now)
        
        return current_tokens
    
    async def consume(self, tenant_id: str, amount: int) -> bool:
        """
        Attempt to consume tokens from tenant's bucket.
        
        Returns True if successful, False if insufficient tokens.
        Thread-safe via per-tenant locks.
        """
        if tenant_id not in self._configs:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        # Acquire tenant-specific lock to prevent race conditions
        async with self._locks[tenant_id]:
            # Refill bucket based on elapsed time
            current_tokens = self._refill_bucket(tenant_id)
            
            if current_tokens >= amount:
                # Sufficient tokens: consume them
                new_tokens = current_tokens - amount
                self._buckets[tenant_id] = (new_tokens, time.time())
                
                print(f"[TenantManager] Tenant {tenant_id} consumed {amount} tokens "
                      f"({new_tokens:.1f}/{self._configs[tenant_id].burst_cap} remaining)")
                return True
            else:
                # Insufficient tokens: reject
                print(f"[TenantManager] Tenant {tenant_id} REJECTED: "
                      f"needs {amount}, has {current_tokens:.1f}")
                return False
    
    def get_tenant_status(self, tenant_id: str) -> Dict:
        """Get current token bucket status for a tenant"""
        if tenant_id not in self._configs:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        current_tokens = self._refill_bucket(tenant_id)
        config = self._configs[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "current_tokens": round(current_tokens, 2),
            "burst_cap": config.burst_cap,
            "rate_limit": config.rate_limit,
            "utilization_pct": round((1 - current_tokens / config.burst_cap) * 100, 1)
        }