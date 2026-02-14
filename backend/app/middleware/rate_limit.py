"""
In-memory rate limiting middleware using cachetools TTLCache.

Tracks requests per client IP with configurable limits per endpoint pattern.
"""

from cachetools import TTLCache
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Per-endpoint rate limits: (max_requests, window_seconds)
ENDPOINT_LIMITS: dict[str, tuple[int, int]] = {
    "/api/v1/predict": (10, 60),
    "/api/v1/chat": (20, 60),
    "/api/v1/analyze-query": (30, 60),
}
DEFAULT_LIMIT = (60, 60)

# One TTLCache per endpoint pattern, keyed by client IP.
# Max size per cache is generous to avoid silent eviction.
_caches: dict[str, TTLCache] = {}


def _get_cache(path: str) -> tuple[TTLCache, int]:
    """Return (cache, max_requests) for the given path."""
    for pattern, (max_req, window) in ENDPOINT_LIMITS.items():
        if path.startswith(pattern):
            if pattern not in _caches:
                _caches[pattern] = TTLCache(maxsize=10_000, ttl=window)
            return _caches[pattern], max_req

    if "__default__" not in _caches:
        max_req, window = DEFAULT_LIMIT
        _caches["__default__"] = TTLCache(maxsize=10_000, ttl=window)
    return _caches["__default__"], DEFAULT_LIMIT[0]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces per-IP rate limits."""

    async def dispatch(self, request: Request, call_next):
        # Use X-Forwarded-For when behind API Gateway / load balancer
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        cache, max_requests = _get_cache(path)

        # Get current request count for this IP
        count = cache.get(client_ip, 0)

        if count >= max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": "60"},
            )

        # Increment count (TTLCache auto-expires after window)
        cache[client_ip] = count + 1

        return await call_next(request)
