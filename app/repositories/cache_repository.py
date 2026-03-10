import json
import asyncio
from redis.asyncio.client import Redis
from redis.exceptions import RedisError
from typing import Optional, Any
from contextlib import suppress


class CacheRepository:
    @staticmethod
    async def set_cache_entity(redis: Redis, key: str, value: str, ttl: int) -> None:
        """
        Store a string value under a Redis key with a TTL.

        Args:
            redis: Redis async client.
            key: Redis key.
            value: String payload to store (caller controls encoding/format).
            ttl: Time-to-live in seconds.

        Returns:
            None
        """

        await redis.setex(key, ttl, value)

    @staticmethod
    async def get_cache_entity(redis: Redis, key: str) -> bool:
        """
        Check whether a Redis key exists (marker semantics).

        Args:
            redis: Redis async client.
            key: Redis key.

        Returns:
            bool: True if key exists, False otherwise.
        """

        val = await redis.get(key)
        return val is not None

    @staticmethod
    async def set_version(redis: Redis, key: str, value: str) -> None:
        """
        Store a version marker string in Redis.

        Args:
            redis: Redis async client.
            key: Redis key.
            value: Version string (e.g., ISO timestamp).

        Returns:
            None
        """

        await redis.set(key, value)

    @staticmethod
    async def get_version(redis: Redis, key: str) -> Optional[str]:
        """
        Fetch a version marker string from Redis.

        Args:
            redis: Redis async client.
            key: Redis key.

        Returns:
            Optional[str]: Stored version string, or None if missing.
        """

        val = await redis.get(key)
        if val is None:
            return None
        return val

    @staticmethod
    async def set_list(redis: Redis, key: str, value: list) -> None:
        """
        JSON-encode and store a list payload in Redis.

        Args:
            redis: Redis async client.
            key: Redis key.
            value: List payload (must be JSON serializable).

        Returns:
            None
        """

        await redis.set(key, json.dumps(value))

    @staticmethod
    async def get_list(redis: Redis, key: str) -> Optional[list]:
        """
        Fetch a JSON-encoded list payload from Redis.

        Args:
            redis: Redis async client.
            key: Redis key.

        Returns:
            Optional[list]:
                - Decoded list if present and valid JSON list
                - None if key missing or invalid JSON or not a list
        """

        raw = await redis.get(key)
        if raw is None:
            return None

        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            return None

        return val if isinstance(val, list) else None

    @staticmethod
    async def set_json(redis: Redis, key: str, value: Any) -> None:
        """
        JSON-encode and store an arbitrary JSON-serializable payload in Redis.

        Args:
            redis: Redis async client.
            key: Redis key.
            value: JSON-serializable payload.

        Returns:
            None
        """

        await redis.set(key, json.dumps(value))

    @staticmethod
    async def get_json(redis: Redis, key: str) -> Any | None:
        """
        Fetch a JSON-encoded payload from Redis.

        Args:
            redis: Redis async client.
            key: Redis key.

        Returns:
            Any | None:
                - Decoded JSON payload if present and valid
                - None if key missing
        """

        raw = await redis.get(key)
        return json.loads(raw) if raw else None

    @staticmethod
    async def delete(redis, key: str):
        """
        Fetch a JSON-encoded payload from Redis.

        Args:
            redis: Redis async client.
            key: Redis key.

        Returns:
            Any | None:
                - Decoded JSON payload if present and valid
                - None if key missing
        """

        return await redis.delete(key)


    @staticmethod
    async def try_set_lock(redis: Redis, key: str, value: str, ttl_s: int) -> bool:
        """
        Try to acquire a lock key using Redis SET with NX + EX.

        Args:
            redis: Redis async client.
            key: Lock key name.
            value: Unique value stored in the lock (used to safely release).
            ttl_s: Lock TTL in seconds.

        Returns:
            bool: True if lock acquired, False if already locked or Redis error.
        """
        try:
            ok = await redis.set(key, value, nx=True, ex=ttl_s)
            return bool(ok)
        except (RedisError, asyncio.TimeoutError):
            return False

    @staticmethod
    async def release_lock(redis: Redis, key: str, value: str) -> None:
        """
        Best-effort safe lock release (only delete if value matches).

        This prevents one request from deleting another request's lock.

        Args:
            redis: Redis async client.
            key: Lock key name.
            value: The expected lock value.

        Returns:
            None.
        """
        lua = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        with suppress(RedisError, asyncio.TimeoutError, Exception):
            await redis.eval(lua, numkeys=1, keys=[key], args=[value])