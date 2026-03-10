from redis.asyncio.client import Redis
from app.repositories.cache_repository import CacheRepository as CRepo
from app.core.logging_config import errors


async def _invalidate_pairs(redis: Redis, ts: str, pairs: list[tuple[str, str]], label: str) -> None:
    """
    Invalidate Redis cache entries for a set of (payload_key, version_key) pairs.

    Invalidation strategy:
      1) Set version_key = ts (forces version mismatch in callers).
      2) Delete payload_key   (removes cached payload).

    Args:
        redis: Redis client.
        ts: Version string (typically ISO timestamp) used as the new cache version.
        pairs: List of (payload_key, version_key) pairs to invalidate.
        label: Label used for logging context on failure.

    Returns:
        None
    """

    try:
        for list_key, ver_key in pairs:
            await CRepo.set_version(redis, ver_key, ts)
            await CRepo.delete(redis, list_key)
    except Exception as e:
        errors.warning(f"{label} cache invalidation failed: {e!r}")


async def invalidate_global_models_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global "all users trained models" cache in Redis.

    Args:
        redis: Redis client.
        ts: Version string (typically applied.created_at.isoformat()).

    Returns:
        None
    """

    await _invalidate_pairs(
        redis,
        ts,
        pairs=[("models:all:list", "models:all:version")],
        label="trained models",
    )


async def invalidate_global_predictions_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global "all users predictions" cache in Redis.

    Args:
        redis: Redis client.
        ts: Version string (typically applied.created_at.isoformat()).

    Returns:
        None
    """

    await _invalidate_pairs(
        redis,
        ts,
        pairs=[("preds:all:list", "preds:all:version")],
        label="predictions",
    )


async def invalidate_global_token_credits_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global "all active users token credits history" cache in Redis.

    Args:
        redis: Redis client.
        ts: Version string (typically applied.created_at.isoformat()).

    Returns:
        None
    """

    await _invalidate_pairs(
        redis,
        ts,
        pairs=[("token_credits:all:list", "token_credits:all:version")],
        label="token credits",
    )


async def invalidate_global_user_usage_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate cached global aggregations used by the usage dashboard.

    These aggregations depend on global trained-model state, so when users are deleted
    or new models are added, the cached distributions may become stale.

    Args:
        redis: Redis client.
        ts: Version string (typically a timestamp) used to bump cache version keys.

    Returns:
        None
    """

    await _invalidate_pairs(
        redis,
        ts,
        pairs=[
            ("usage:model_type:list", "usage:model_type:version"),
            ("usage:type_split:list", "usage:type_split:version"),
            ("usage:label_distribution:json", "usage:label_distribution:version"),
            ("usage:metric_distribution:json", "usage:metric_distribution:version"),
        ],
        label="user usage",
    )


async def _delete_keys(redis: Redis, keys: list[str], label: str) -> None:
    """
    Delete a list of Redis keys best-effort.

    Args:
        redis: Redis client.
        keys: List of Redis keys to delete.
        label: Label used for logging context on failure.

    Returns:
        None
    """

    try:
        for k in keys:
            await CRepo.delete(redis, k)
    except Exception as e:
        errors.warning(f"{label} key cleanup failed: {e!r}")


