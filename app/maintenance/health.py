import asyncio
import os
from pathlib import Path
from redis.asyncio.client import Redis
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text


def ensure_disk_ok(base_dir: str, min_free_mb: int = 50) -> bool:
    """
    Quick guard: verify we can write to base_dir and that free space >= min_free_mb.
    Return True if safe, False if we should refuse heavy work.
    """
    p = Path(base_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)

        stat = os.statvfs(str(p))
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        if free_mb < min_free_mb:
            return False

        probe = p / ".health"
        probe.write_text("ok")

        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return True
    except OSError:
        return False


async def db_ping_once(engine: AsyncEngine, timeout_s: float = 2.0) -> bool:
    """
    Try a tiny round trip to the DB. Return True if OK, False otherwise.
    """

    async def _ping():
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(_ping(), timeout=timeout_s)
        return True
    except (SQLAlchemyError, asyncio.TimeoutError):
        return False


async def redis_ping_once(redis: Redis, timeout_s: float = 1.0) -> bool:
    async def _ping():
        await redis.ping()

    try:
        await asyncio.wait_for(_ping(), timeout=timeout_s)
        return True
    except (asyncio.TimeoutError, Exception):
        return False



