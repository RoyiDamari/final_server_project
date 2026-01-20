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
    Enforces rate limiting using Redis.

    Args:
        key (str): Fully-qualified rate-limit key (includes scope + identifier)
        redis (Redis): Redis client
        max_requests (int): Allowed requests per window
        window (int): Window size in seconds

    Raises:
        RateLimitException: if limit exceeded
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
    Build a stable identifier for rate limiting.

    Priority:
    1. Authenticated user.id
    2. Username / email from request payload (login/register)
    3. Client IP
    """
    user = kwargs.get("user")
    if user:
        return f"ratelimit:{scope}:user:{user.id}"

    if request and request.client:
        return f"ratelimit:{scope}:ip:{request.client.host}"

    return f"ratelimit:{scope}:unknown"


def rate_limited(scope: str, max_requests: int, window: int):
    """
    Rate limit decorator.

    Example:
        @rate_limited("login", max_requests=10, window=600)
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
