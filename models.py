from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
import uuid
from dataclasses import dataclass


class Request(BaseModel):
    """Inference request from a tenant"""
    
    model_config = ConfigDict(frozen=False)
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Unique tenant identifier")
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    tokens_requested: int = Field(..., gt=0, description="Estimated token count")
    output_tokens_expected: int = Field(default=50, gt=0, description="Expected output tokens")
    priority_bid: int = Field(default=1, ge=0, description="Priority bid (higher = more priority)")
    arrival_time: datetime = Field(default_factory=datetime.utcnow)
    
    def effective_priority(self) -> float:
        """Calculate effective priority (can be dynamic based on wait time, etc.)"""
        # For now, just return the bid, but can be extended for dynamic priority
        return self.priority_bid
    
    def __lt__(self, other: 'Request') -> bool:
        """For PriorityQueue comparison: higher bid first, then older"""
        # Note: PriorityQueue is min-heap, so negate bid for max behavior
        return (-self.priority_bid, self.arrival_time) < (-other.priority_bid, other.arrival_time)


@dataclass
class HeapEntry:
    """Entry for priority queue with dynamic priority support"""
    # We use NEGATIVE priority for max-heap behavior (heapq is min-heap)
    neg_effective_priority: float
    arrival_time: float           # for stable sort
    request: Request              # the actual payload

    def __lt__(self, other):
        if self.neg_effective_priority != other.neg_effective_priority:
            return self.neg_effective_priority < other.neg_effective_priority  # smaller neg = higher priority
        return self.arrival_time < other.arrival_time


# Helper to create entry with current priority snapshot
def make_heap_entry(req: Request) -> HeapEntry:
    return HeapEntry(
        neg_effective_priority= -req.effective_priority(),
        arrival_time=req.arrival_time.timestamp(),
        request=req
    )


class TenantConfig(BaseModel):
    """Tenant rate limiting configuration"""
    
    model_config = ConfigDict(frozen=True)
    
    tenant_id: str = Field(..., description="Unique tenant identifier")
    rate_limit: float = Field(..., gt=0, description="Tokens per second allowed")
    burst_cap: int = Field(..., gt=0, description="Maximum token bucket capacity")


class InferenceResponse(BaseModel):
    """Response from inference endpoint"""
    
    request_id: str
    status: str  # "queued" | "rejected" | "processing"
    message: str
    queue_position: Optional[int] = None
    estimated_wait_ms: Optional[float] = None