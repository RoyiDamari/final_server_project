import time
import functools
from fastapi import Request
from redis.asyncio.client import Redis
from app.utils.redis import get_redis
from app.utils.cache_keys import CacheKeys
from app.exceptions.rate_limit import RateLimitException


async def check_rate_limit(
        key: str,
        redis: Redis,
        max_requests: int,
        window: int,
):
    """
    Enforce a sliding-window rate limit using Redis list timestamps.

    Strategy:
      - Keep up to max_requests timestamps in a Redis list.
      - If list has capacity -> push now, trim, set expire(window).
      - If full and oldest within window -> raise RateLimitException(retry_after).
      - If window expired -> reset by pushing now.

    Args:
        key: Redis key for this limiter bucket.
        redis: Redis client.
        max_requests: Maximum allowed requests within the window.
        window: Window size in seconds.

    Returns:
        None

    Raises:
        RateLimitException: If limit is exceeded and caller should retry later.
        redis.exceptions.RedisError: If Redis operations fail.
        ValueError: If stored timestamps are not parseable as ints.
    """

    now = int(time.time())

    timestamps = await redis.lrange(key, 0, -1)
    timestamps = [int(ts) for ts in timestamps]

    # allowed — push timestamp & trim
    if len(timestamps) < max_requests:
        pipe = redis.pipeline()
        pipe.lpush(key, now)
        pipe.ltrim(key, 0, max_requests - 1)
        pipe.expire(key, window)
        await pipe.execute()
        return

    # full — compute retry AFTER without pushing
    oldest = timestamps[-1]
    elapsed = now - oldest
    if elapsed < window:
        retry_after = window - elapsed
        raise RateLimitException(retry_after=retry_after)

    # window expired → reset by pushing timestamp
    pipe = redis.pipeline()
    pipe.lpush(key, now)
    pipe.ltrim(key, 0, max_requests - 1)
    pipe.expire(key, window)
    await pipe.execute()


def _build_identifier(
        scope: str,
        request: Request | None,
        kwargs: dict,
) -> str:
    """
    Build a stable identifier string used for rate limiting buckets.

    Priority:
      1) Authenticated user.id (if present in kwargs["user"])
      2) Client IP (if request.client is available)
      3) Fallback "unknown"

    Args:
        scope: Logical scope name (e.g., "login", "train").
        request: Optional FastAPI Request.
        kwargs: Endpoint kwargs (used to detect authenticated user).

    Returns:
        str: Fully-qualified identifier string (not yet passed through CacheKeys).
    """

    user = kwargs.get("user")
    if user:
        return f"ratelimit:{scope}:user:{user.id}"

    if request and request.client:
        return f"ratelimit:{scope}:ip:{request.client.host}"

    return f"ratelimit:{scope}:unknown"


def rate_limited(scope: str, max_requests: int, window: int):
    """
    Decorator that rate-limits an async FastAPI endpoint using Redis.

    Args:
        scope: Scope name for the limiter (used in the Redis key namespace).
        max_requests: Allowed requests per window.
        window: Window length in seconds.

    Returns:
        Callable: Decorator that wraps the endpoint.

    Raises:
        RateLimitException: Propagated from check_rate_limit when limit is exceeded.
        redis.exceptions.RedisError: If Redis is unavailable and get_redis() / Redis ops fail.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request | None = (
                    kwargs.get("request")
                    or kwargs.get("_request")
            )

            identifier = _build_identifier(scope, request, kwargs)
            redis = await get_redis()

            key = CacheKeys.rate_limit(identifier)
            await check_rate_limit(key, redis, max_requests, window)

            return await func(*args, **kwargs)

        return wrapper

    return decorator
