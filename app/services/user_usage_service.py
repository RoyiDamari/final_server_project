import asyncio
from typing import Dict, Any
from contextlib import suppress
from redis.asyncio.client import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.cache_repository import CacheRepository as CRepo
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.user_usage_repository import UserUsageRepository as UURepo
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.repositories.seen_version_repository import SeenVersionRepository as SVRepo
from app.models.orm_models import User
from app.models.enums import ActionType
from app.core.logs import log_action, errors


class UserUsageService:
    @staticmethod
    async def get_model_type_distribution(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return model-type distribution with "charge once per dataset version".

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated user.
            action: ActionType containing token cost.

        Returns:
            dict[str, Any]: {"data": list, "charged": bool, "balance": int}
        """

        resource = "usage:model_type"
        list_key = "usage:model_type:list"
        ver_key = "usage:model_type:version"

        async with db.begin():
            db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
            if db_ver_dt is None:
                return {"data": [], "charged": False, "balance": user.tokens}

            db_ver = db_ver_dt.isoformat()

            first_time = await SVRepo.insert_seen(db, user_id=user.id, resource=resource, version=db_ver)
            if first_time:
                balance = await URepo.update_tokens(db, user.id, action.cost)
                charged = True
            else:
                balance = await URepo.get_tokens_by_id(db, user.id)
                charged = False

        data: list[dict] | None = None
        try:
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                data = await CRepo.get_list(redis, list_key)
        except (RedisError, asyncio.TimeoutError):
            data = None

        if data is None:
            rows = await UURepo.get_model_type_distribution(db)
            data = rows or []

            with suppress(RedisError, asyncio.TimeoutError):
                await CRepo.set_list(redis, list_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="usage_model_type_distribution",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=(action.cost if charged else 0),
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed in get_model_type_distribution: %r", e)

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_regression_vs_classification_split(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return regression-vs-classification split with "charge once per dataset version".

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated user.
            action: ActionType containing token cost.

        Returns:
            dict[str, Any]: {"data": list, "charged": bool, "balance": int}
        """

        resource = "usage:type_split"
        list_key = "usage:type_split:list"
        ver_key = "usage:type_split:version"

        async with db.begin():
            db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
            if db_ver_dt is None:
                return {"data": [], "charged": False, "balance": user.tokens}

            db_ver = db_ver_dt.isoformat()

            first_time = await SVRepo.insert_seen(db, user_id=user.id, resource=resource, version=db_ver)
            if first_time:
                balance = await URepo.update_tokens(db, user.id, action.cost)
                charged = True
            else:
                balance = await URepo.get_tokens_by_id(db, user.id)
                charged = False

        data: list[dict] | None = None
        try:
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                data = await CRepo.get_list(redis, list_key)
        except (RedisError, asyncio.TimeoutError):
            data = None

        if data is None:
            rows = await UURepo.get_regression_vs_classification_split(db)
            data = rows or []

            with suppress(RedisError, asyncio.TimeoutError):
                await CRepo.set_list(redis, list_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="problem_type_split",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=(action.cost if charged else 0),
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed in get_regression_vs_classification_split: %r", e)

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_label_distribution(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return label distribution with "charge once per dataset version".

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated user.
            action: ActionType containing token cost.

        Returns:
            dict[str, Any]: {"data": dict, "charged": bool, "balance": int}
        """

        resource = "usage:label_distribution"
        json_key = "usage:label_distribution:json"
        ver_key = "usage:label_distribution:version"

        async with db.begin():
            db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
            if db_ver_dt is None:
                return {"data": {"classification": [], "regression": []}, "charged": False, "balance": user.tokens}

            db_ver = db_ver_dt.isoformat()

            first_time = await SVRepo.insert_seen(db, user_id=user.id, resource=resource, version=db_ver)
            if first_time:
                balance = await URepo.update_tokens(db, user.id, action.cost)
                charged = True
            else:
                balance = await URepo.get_tokens_by_id(db, user.id)
                charged = False

        data: dict | None = None
        try:
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                data = await CRepo.get_json(redis, json_key)
        except (RedisError, asyncio.TimeoutError):
            data = None

        if data is None:
            data = await UURepo.get_label_distribution(db) or {"classification": [], "regression": []}
            with suppress(RedisError, asyncio.TimeoutError):
                await CRepo.set_json(redis, json_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="user_viewed_label_distribution",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=(action.cost if charged else 0),
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed in get_label_distribution: %r", e)

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_metric_distribution(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return metric distributions with "charge once per dataset version".

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated user.
            action: ActionType containing token cost.

        Returns:
            dict[str, Any]: {"data": dict, "charged": bool, "balance": int}
        """

        resource = "usage:metric_distribution"
        json_key = "usage:metric_distribution:json"
        ver_key = "usage:metric_distribution:version"

        async with db.begin():
            db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
            if db_ver_dt is None:
                return {"data": {"classification": [], "regression": []}, "charged": False, "balance": user.tokens}

            db_ver = db_ver_dt.isoformat()

            first_time = await SVRepo.insert_seen(db, user_id=user.id, resource=resource, version=db_ver)
            if first_time:
                balance = await URepo.update_tokens(db, user.id, action.cost)
                charged = True
            else:
                balance = await URepo.get_tokens_by_id(db, user.id)
                charged = False

        data: dict | None = None
        try:
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                data = await CRepo.get_json(redis, json_key)
        except (RedisError, asyncio.TimeoutError):
            data = None

        if data is None:
            data = await UURepo.get_metric_distribution(db) or {"classification": [], "regression": []}
            with suppress(RedisError, asyncio.TimeoutError):
                await CRepo.set_json(redis, json_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="user_viewed_metric_distribution",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=(action.cost if charged else 0),
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed in get_metric_distribution: %r", e)

        return {"data": data, "charged": charged, "balance": balance}
