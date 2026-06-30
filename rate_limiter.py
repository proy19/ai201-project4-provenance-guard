"""
Token-bucket rate limiter — per-IP, in-memory.

Each IP gets a bucket with a fixed capacity. Tokens refill at a steady
rate. A request costs one token. If the bucket is empty the request is
rejected with 429.

Usage:
    limiter = RateLimiter(capacity=10, refill_rate=1.0)   # 10 req burst, 1 req/s steady
    allowed, retry_after = limiter.check("192.168.1.1")
    if not allowed:
        return jsonify({"error": "rate limit exceeded"}), 429
"""

import time
import threading
from dataclasses import dataclass, field


@dataclass
class Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    capacity    — maximum tokens per bucket (burst ceiling)
    refill_rate — tokens added per second (steady-state throughput)
    """

    def __init__(self, capacity: float = 10, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    def _refill(self, bucket: Bucket) -> None:
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
        bucket.last_refill = now

    def check(self, key: str) -> tuple[bool, float]:
        """
        Returns (allowed, retry_after_seconds).
        retry_after is 0.0 when allowed.
        """
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = Bucket(tokens=self.capacity)

            bucket = self._buckets[key]
            self._refill(bucket)

            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return True, 0.0
            else:
                # How long until one token refills
                retry_after = (1 - bucket.tokens) / self.refill_rate
                return False, round(retry_after, 2)

    def purge_stale(self, max_age_seconds: float = 3600) -> int:
        """Remove buckets that haven't been touched in max_age_seconds. Returns count removed."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, b in self._buckets.items()
                     if (now - b.last_refill) > max_age_seconds]
            for k in stale:
                del self._buckets[k]
        return len(stale)
