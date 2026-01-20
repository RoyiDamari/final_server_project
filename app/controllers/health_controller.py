from fastapi import APIRouter, Response, status
from app.maintenance.health import ensure_disk_ok, db_ping_once
from app.database import engine
from app.core.logging_config import errors


router = APIRouter(prefix="", tags=["health"])

_last_ok = True

@router.get("/health")
async def health() -> Response:
    """
    Return 200 when the app is healthy:
      - disk writable with enough free space
      - DB reachable
    Otherwise 503.
    """
    global _last_ok
    disk_ok = (
            ensure_disk_ok("/app/saved_models", min_free_mb=50)
            and ensure_disk_ok("/app/logs", min_free_mb=50)
    )
    db_ok = await db_ping_once(engine, timeout_s=2.0)
    ok = bool(disk_ok and db_ok)

    if ok and not _last_ok:
        errors.info("Health transition: UNHEALTHY → HEALTHY")
    elif not ok and _last_ok:
        errors.warning("Health transition: HEALTHY → UNHEALTHY (disk_ok=%s db_ok=%s)", disk_ok, db_ok)

    _last_ok = ok
    return Response(status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE)
