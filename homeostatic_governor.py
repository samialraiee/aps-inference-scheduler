"""
Homeostatic Governor: An entropic feedback loop for adaptive batch scheduling.

This module implements a self-regulating system that measures Shannon entropy
of request inter-arrival times to dynamically adjust micro-batching windows.
The governor prevents phase transitions into congestion through negative feedback,
maintaining system stability without explicit control policies.

References:
    - Shannon, C. E. (1948). "A Mathematical Theory of Communication"
    - Wiener, N. (1948). "Cybernetics: Or Control and Communication..."
"""

import math
import time
from collections import deque
from typing import Deque, Dict, Literal


class HomeostaticGovernor:
    """
    Entropic feedback loop for self-regulating batch window adaptation.
    
    Measures Shannon entropy H of inter-arrival intervals to distinguish
    chaotic (high-H) from patterned (low-H) request streams. Applies
    exponential scaling to base batch window, creating a negative feedback
    mechanism: chaos → narrower window → faster queue drain.
    
    Attributes:
        arrival_times: Bounded deque (size=50) of request arrival timestamps.
        base_batch_window: Base window in seconds (default 0.01s = 10ms).
        current_entropy: Last computed entropy value.
    """

    ENTROPY_CRITICAL_THRESHOLD: float = 1.5
    """Threshold below which system signals 'CRITICAL BURST' state."""

    def __init__(
        self,
        window_size: int = 50,
        base_batch_window: float = 0.01,
    ) -> None:
        """
        Initialize the homeostatic governor.
        
        Args:
            window_size: Size of arrival_times deque. Larger window = slower
                         entropy response but more stable baseline (default: 50).
            base_batch_window: Base batch window in seconds (default: 0.01s).
        """
        self.arrival_times: Deque[float] = deque(maxlen=window_size)
        self.base_batch_window: float = base_batch_window
        self.current_entropy: float = 0.0

    def record_arrival(self) -> None:
        """
        Record a request arrival timestamp using high-precision wall clock.
        
        Called on every inbound request. Updates the deque of recent arrivals.
        """
        self.arrival_times.append(time.perf_counter())

    def calculate_entropy(self) -> float:
        r"""
        Compute Shannon entropy H of inter-arrival intervals.
        
        Given intervals I = {t[i] - t[i-1] for i in 1..n}, we bin intervals
        at 1ms precision and compute:
        
            H = -Σ p_i log₂(p_i)
        
        where p_i = (count in bin i) / (total intervals).
        
        Returns:
            Entropy in bits. Range: [0, log₂(num_bins)].
            0 = perfect regularity, ~3 = full chaos (50-interval window).
        """
        if len(self.arrival_times) < 2:
            self.current_entropy = 0.0
            return 0.0

        # Compute inter-arrival intervals with monotonicity check
        intervals: list[float] = [
            self.arrival_times[i] - self.arrival_times[i - 1]
            for i in range(1, len(self.arrival_times))
            if self.arrival_times[i] >= self.arrival_times[i - 1]
        ]

        if not intervals:
            self.current_entropy = 0.0
            return 0.0

        # Bin intervals at 1ms granularity
        bins: Dict[float, int] = {}
        for interval in intervals:
            bucket = round(interval, 3)  # 1ms precision
            bins[bucket] = bins.get(bucket, 0) + 1

        # Compute Shannon entropy
        total = len(intervals)
        entropy = 0.0
        for count in bins.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        self.current_entropy = entropy
        return entropy

    def get_adaptive_batch_window(self) -> float:
        r"""
        Compute dynamically scaled batch window via entropic feedback.
        
        Formula:
            window = base_window × exp(-H / 5.0)
        
        Where H is current entropy. Intuition:
        - H ≈ 0 (pattern): exp(0) = 1.0 → window = base (exploit pattern)
        - H ≈ 2.5 (chaos): exp(-0.5) ≈ 0.6 → window = 0.6× base (shrink to drain)
        - H → ∞: exp(-∞) → 0 → window → 0 (extreme chaos, minimal batching)
        
        Returns:
            Adaptive batch window in seconds.
        """
        entropy = self.calculate_entropy()
        adjustment = math.exp(-entropy / 5.0)
        return self.base_batch_window * adjustment

    def get_system_status(self) -> Literal["CRITICAL BURST", "STABLE"]:
        """
        Classify current system state based on entropy threshold.
        
        Returns:
            "CRITICAL BURST" if entropy < 1.5 (indicates bursty, synchronized arrivals).
            "STABLE" if entropy >= 1.5 (indicates diverse, patterned arrivals).
        """
        return "CRITICAL BURST" if self.current_entropy < self.ENTROPY_CRITICAL_THRESHOLD else "STABLE"
