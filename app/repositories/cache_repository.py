import json
from redis.asyncio.client import Redis
from typing import Optional, Any


class CacheRepository:
    @staticmethod
    async def set_cache_entity(redis: Redis, key: str, value: str, ttl: int) -> None:
        """
        Create or overwrite a cache entry in Redis.
        Stores the value as JSON with an optional TTL (in seconds).
        """
        await redis.setex(key, ttl, value)

    @staticmethod
    async def get_cache_entity(redis: Redis, key: str) -> bool:
        """
        Returns True if key exists, False otherwise.
        Value content is irrelevant for marker semantics.
        """
        val = await redis.get(key)
        return val is not None


    @staticmethod
    async def set_version(redis: Redis, key: str, value: str) -> None:
        await redis.set(key, value)


    @staticmethod
    async def get_version(redis: Redis, key: str) -> Optional[str]:
        val = await redis.get(key)
        if val is None:
            return None
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)


    @staticmethod
    async def set_list(redis: Redis, key: str, value: list) -> None:
        await redis.set(key, json.dumps(value))


    @staticmethod
    async def get_list(redis: Redis, key: str) -> Optional[list]:
        raw = await redis.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @staticmethod
    async def set_json(redis: Redis, key: str, value: Any) -> None:
        await redis.set(key, json.dumps(value))


    @staticmethod
    async def get_json(redis: Redis, key: str) -> Any | None:
        raw = await redis.get(key)
        return json.loads(raw) if raw else None

    @staticmethod
    async def delete(redis, key: str):
        return await redis.delete(key)

