from redis.asyncio.client import Redis
from app.repositories.cache_repository import CacheRepository as CRepo
from app.core.logging_config import errors


async def _invalidate_pairs(redis: Redis, ts: str, pairs: list[tuple[str, str]], label: str) -> None:
    """
    Internal helper to invalidate a set of (list_key, version_key) cache pairs.

    Invalidation strategy:
      1) set version_key = ts   (forces version mismatch / refresh)
      2) delete list_key        (removes cached payload)

    Notes:
      - Version bump is optional in your architecture because deleting list_key
        alone is enough to force a rebuild (your service code handles cache-miss).
      - Keeping the bump is fine as a "force refresh" signal.
    """
    try:
        for list_key, ver_key in pairs:
            await CRepo.set_version(redis, ver_key, ts)
            await CRepo.delete(redis, list_key)
    except Exception as e:
        errors.warning(f"{label} cache invalidation failed: {e!r}")


async def invalidate_global_models_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global 'all users models' cache.

    Keys:
      - models:all:list
      - models:all:version
    """
    await _invalidate_pairs(
        redis,
        ts,
        pairs=[("models:all:list", "models:all:version")],
        label="trained models",
    )


async def invalidate_global_predictions_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global 'all users predictions' cache.

    Keys:
      - preds:all:list
      - preds:all:version
    """
    await _invalidate_pairs(
        redis,
        ts,
        pairs=[("preds:all:list", "preds:all:version")],
        label="predictions",
    )


async def invalidate_global_token_credits_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate the global 'all users token credits history' cache.

    Keys:
      - token_credits:all:list
      - token_credits:all:version
    """
    await _invalidate_pairs(
        redis,
        ts,
        pairs=[("token_credits:all:list", "token_credits:all:version")],
        label="token credits",
    )


async def invalidate_global_user_usage_cache(redis: Redis, ts: str) -> None:
    """
    Invalidate cached aggregations for the usage dashboard.

    These endpoints typically depend on global trained-model state, so when users
    are deleted or new models appear, cached distributions can become stale.

    Keys:
      - usage:model_type:list / usage:model_type:version
      - usage:type_split:list / usage:type_split:version
      - usage:label_distribution:list / usage:label_distribution:version
      - usage:metric_distribution:list / usage:metric_distribution:version
    """
    await _invalidate_pairs(
        redis,
        ts,
        pairs=[
            ("usage:model_type:list", "usage:model_type:version"),
            ("usage:type_split:list", "usage:type_split:version"),
            ("usage:label_distribution:list", "usage:label_distribution:version"),
            ("usage:metric_distribution:list", "usage:metric_distribution:version"),
        ],
        label="user usage",
    )


async def _delete_keys(redis: Redis, keys: list[str], label: str) -> None:
    """Internal helper to delete a list of Redis keys safely."""
    try:
        for k in keys:
            await CRepo.delete(redis, k)
    except Exception as e:
        errors.warning(f"{label} key cleanup failed: {e!r}")


async def cleanup_user_seen_keys(redis: Redis, user_id: int) -> None:
    """
    Delete all per-user Version-D billing markers ('last_seen' keys) for dashboards.

    Optional cleanup:
      - Not required for correctness after a user is deactivated
      - Prevents orphan per-user Redis keys from accumulating
    """
    keys = [
        # usage dashboards
        f"usage:model_type:last_seen:{user_id}",
        f"usage:type_split:last_seen:{user_id}",
        f"usage:label_distribution:last_seen:{user_id}",
        f"usage:metric_distribution:last_seen:{user_id}",

        # global dashboards
        f"models:all:last_seen:{user_id}",
        f"preds:all:last_seen:{user_id}",
        f"tokens:all:last_seen:{user_id}",
    ]
    await _delete_keys(redis, keys, label="user seen")
