from redis.asyncio.client import Redis
from app.repositories.cache_repository import CacheRepository as CRepo
from app.core.logging_config import errors


async def invalidate_global_models_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global list of ALL users' trained models
    and set new global version.
    """
    try:
        list_key = "models:all:list"
        ver_key = "models:all:version"

        await CRepo.set_version(redis, ver_key, ts)
        await CRepo.delete(redis, list_key)

    except Exception as e:
        errors.warning(f"cache bump failed: {e!r}")


async def invalidate_global_predictions_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate global predictions viewer cache
    when ANY user posts a new prediction.
    """
    try:

        list_key = "preds:all:list"
        ver_key = "preds:all:version"

        await CRepo.set_version(redis, ver_key, ts)
        await CRepo.delete(redis, list_key)

    except Exception as e:
        errors.warning(f"cache bump failed: {e!r}")


