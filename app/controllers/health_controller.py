from fastapi import APIRouter, Depends, Response, status
from redis.asyncio.client import Redis
from app.maintenance.health import ensure_disk_ok, db_ping_once, redis_ping_once
from app.utils.redis import get_redis
from app.database import engine
from app.core.logging_config import errors

router = APIRouter(prefix="", tags=["health"])

_last_ready_ok = True


@router.get("/ready")
async def ready(redis: Redis = Depends(get_redis)) -> Response:
    """
    Readiness probe.

    Meaning:
      - App can serve real traffic correctly.

    Checks:
      - Disk writable + enough free space
      - Postgres reachable (SELECT 1)
      - Redis reachable (PING)

    Args:
        redis: Redis client dependency.

    Returns:
      - 200 if ready
      - 503 if not ready
    """
    global _last_ready_ok

    disk_ok = (
            ensure_disk_ok("/app/saved_models", min_free_mb=50)
            and ensure_disk_ok("/app/logs", min_free_mb=50)
    )
    db_ok = await db_ping_once(engine, timeout_s=2.0)
    redis_ok = await redis_ping_once(redis, timeout_s=1.0)

    ok = bool(disk_ok and db_ok and redis_ok)

    if ok and not _last_ready_ok:
        errors.info("Readiness transition: NOT READY → READY")
    elif not ok and _last_ready_ok:
        errors.warning(
            "Readiness transition: READY → NOT READY (disk_ok=%s db_ok=%s redis_ok=%s)",
            disk_ok, db_ok, redis_ok
        )

    _last_ready_ok = ok
    return Response(status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE)
