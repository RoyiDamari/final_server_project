from redis.asyncio.client import Redis
from app.config import config
from typing import Optional, cast

redis_client: Optional[Redis] = None


async def init_redis() -> None:
    """Initialize a global Redis async client"""
    global redis_client
    redis_client = Redis.from_url(
        config.REDIS_URL,
        max_connections=10,
        decode_responses=True,
    )
    print("✅ Redis client initialized")


async def get_redis() -> Redis:
    """Return the initialized Redis client"""
    if redis_client is None:
        raise RuntimeError("Redis has not been initialized")
    return redis_client


async def close_redis() -> None:
    """Close Redis connection gracefully"""
    global redis_client
    client = redis_client

    if isinstance(client, Redis):
        await cast(Redis, client).aclose()

    redis_client = None
    print("🛑 Redis connection closed")
