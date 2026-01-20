from typing import Dict, Any
from redis.asyncio.client import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.user_usage_repository import UserUsageRepository as UURepo
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.models.orm_models import User
from app.models.enums import ActionType
from app.core.logs import log_action


class UserUsageService:
    @staticmethod
    async def get_model_type_distribution(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:

        list_key = "usage:model_type:list"
        ver_key = "usage:model_type:version"
        seen_key = f"usage:model_type:last_seen:{user.id}"

        db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
        if db_ver_dt is None:
            return {"data": [], "charged": False, "balance": user.tokens}

        db_ver = db_ver_dt.isoformat()
        redis_ver = await CRepo.get_version(redis, ver_key)

        if redis_ver == db_ver:
            cached = await CRepo.get_list(redis, list_key)
            if cached is not None:
                data = cached
            else:
                rows = await UURepo.get_model_type_distribution(db)
                data = rows or []
                await CRepo.set_list(redis, list_key, data)
        else:
            rows = await UURepo.get_model_type_distribution(db)
            data = rows or []
            await CRepo.set_list(redis, list_key, data)
            await CRepo.set_version(redis, ver_key, db_ver)

        seen_ver = await CRepo.get_version(redis, seen_key)
        if seen_ver == db_ver:
            charged = False
            balance = user.tokens
        else:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            charged = True
            await CRepo.set_version(redis, seen_key, db_ver)

        log_action(
            event="usage_model_type_distribution",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=charged,
            balance_after=balance,
        )

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_regression_vs_classification_split(
        db: AsyncSession,
        redis: Redis,
        user,
        action: ActionType,
    ) -> Dict[str, Any]:

        list_key = "usage:type_split:list"
        ver_key  = "usage:type_split:version"
        user_seen_key = f"usage:type_split:last_seen:{user.id}"

        db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
        if db_ver_dt is None:
            return {"data": [], "charged": False, "balance": user.tokens}

        db_ver = db_ver_dt.isoformat()
        redis_ver = await CRepo.get_version(redis, ver_key)

        if redis_ver == db_ver:
            cached = await CRepo.get_list(redis, list_key)
            if cached is not None:
                data = cached
            else:
                raw = await UURepo.get_regression_vs_classification_split(db)
                data = raw or []
                await CRepo.set_list(redis, list_key, data)
        else:
            raw = await UURepo.get_regression_vs_classification_split(db)
            data = raw or []
            await CRepo.set_list(redis, list_key, data)
            await CRepo.set_version(redis, ver_key, db_ver)

        last_seen = await CRepo.get_version(redis, user_seen_key)
        if last_seen == db_ver:
            charged = False
            balance = user.tokens
        else:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            charged = True
            await CRepo.set_version(redis, user_seen_key, db_ver)

        log_action(
            event="problem_type_split",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=charged,
            balance_after=balance,
        )

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_label_distribution(
        db: AsyncSession,
        redis: Redis,
        user,
        action: ActionType,
    ) -> Dict[str, Any]:

        list_key = f"usage:label_distribution:list"
        ver_key  = f"usage:label_distribution:version"
        user_seen_key = f"usage:label_distribution:last_seen:{user.id}"

        db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
        if db_ver_dt is None:
            return {"data": [], "charged": False, "balance": user.tokens}

        db_ver = db_ver_dt.isoformat()
        redis_ver = await CRepo.get_version(redis, ver_key)

        if redis_ver == db_ver:
            cached = await CRepo.get_list(redis, list_key)
            if cached is not None:
                data = cached
            else:
                raw = await UURepo.get_label_distribution(db)
                data = raw or []
                await CRepo.set_list(redis, list_key, data)
        else:
            raw = await UURepo.get_label_distribution(db)
            data = raw or []
            await CRepo.set_list(redis, list_key, data)
            await CRepo.set_version(redis, ver_key, db_ver)

        last_seen = await CRepo.get_version(redis, user_seen_key)
        if last_seen == db_ver:
            charged = False
            balance = user.tokens
        else:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            charged = True
            await CRepo.set_version(redis, user_seen_key, db_ver)

        log_action(
            event="user_viewed_label_distribution",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=charged,
            balance_after=balance,
        )

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_metric_distribution(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Version-D billing:
        - if no accuracy metrics exist → no charge
        - each user charged ONCE per dataset version (max(created_at))
        - new model with accuracy metric triggers version bump & re-charge
        """

        list_key = "usage:metric_distribution:list"
        ver_key = "usage:metric_distribution:version"
        seen_key = f"usage:metric_distribution:last_seen:{user.id}"

        db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
        if db_ver_dt is None:
            return {"data": [], "charged": False, "balance": user.tokens}

        db_ver = db_ver_dt.isoformat()
        redis_ver = await CRepo.get_version(redis, ver_key)

        if redis_ver == db_ver:
            cached = await CRepo.get_json(redis, list_key)
            if cached is not None:
                data = cached
            else:
                raw = await UURepo.get_metric_distribution(db)
                data = raw
                await CRepo.set_json(redis, list_key, data)
        else:
            raw = await UURepo.get_metric_distribution(db)
            data = raw
            await CRepo.set_json(redis, list_key, data)
            await CRepo.set_version(redis, ver_key, db_ver)

        seen_ver = await CRepo.get_version(redis, seen_key)
        if seen_ver == db_ver:
            charged = False
            balance = user.tokens
        else:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            charged = True
            await CRepo.set_version(redis, seen_key, db_ver)

        log_action(
            event="user_viewed_metric_distribution",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=charged,
            balance_after=balance,
        )

        return {"data": data, "charged": charged, "balance": balance}


