import asyncio
from typing import Dict, Any, Optional
from redis.asyncio.client import Redis
from contextlib import suppress
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from app.exceptions.train_model import  TrainModelInProgressException, TrainingFailedException
from app.models.orm_models.users import User
from app.models.orm_models.trained_models import TrainedModel
from app.models.pydantic_models.train_model import TrainedModelResponse
from app.models.enums import ActionType, RowStatus
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.utils.validators import (
    ensure_csv_valid, ensure_label_valid, ensure_features_valid,
    ensure_model_type_valid, normalize_params, ensure_params_valid,
    normalize_meta_for_fingerprint, validate_param_values
)
from app.models.ml_models.model_strategy_factory import get_model_strategy
from app.utils.fingerprint_hashing import compute_training_fingerprint
from app.utils.files import unique_model_path, temp_path_for, move_temp_to_final, ArtifactWriteException, safe_unlink
from app.utils.cache_invalidation import invalidate_global_models_cache
from app.workers.procs import build_train_worker_cmd, run_training_subprocess
from app.core.logs import log_action


class TrainModelService:
    @staticmethod
    async def train_model(
        db: AsyncSession,
        redis: Redis,
        user: User,
        file: str,
        model_type: str,
        features: list[str],
        label: str,
        model_params: dict,
        action: ActionType,
    ) -> dict:

        df = ensure_csv_valid(file)
        label_c = ensure_label_valid(df, label)
        feats_c = ensure_features_valid(df, features, label_c)
        mt_c = ensure_model_type_valid(model_type)
        params_n = normalize_params(model_params)
        strat = get_model_strategy(mt_c, feats_c, label_c, dict(params_n))
        y = df[label_c]
        strat.validate_target_type(y)
        params_c = ensure_params_valid(strat, params_n, df)
        validate_param_values(mt_c, params_c)

        fp = compute_training_fingerprint(
            csv_file_path=file,
            sorted_features_clean=sorted(feats_c),
            label_clean=label_c,
            model_type_clean=mt_c,
            params_norm=normalize_meta_for_fingerprint(mt_c, params_c, params_n),
        )
        final_path = unique_model_path(user, fp)
        tmp_path = temp_path_for(final_path)

        feature_schema = {}
        for col in feats_c:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_schema[col] = "numeric"
            else:
                feature_schema[col] = "categorical"

        row_id = await TMRepo.try_insert_pending(
            db,
            user_id=user.id,
            model_type=mt_c,
            features=feats_c,
            model_params=params_n,
            label=label_c,
            feature_schema=feature_schema,
            fingerprint=fp,
            model_path=final_path
        )

        if row_id is None:
            existing = await TMRepo.get_by_user_fingerprint(db, user.id, fp)
            if not existing or existing.status == RowStatus.pending:
                raise TrainModelInProgressException()
            if existing.status == RowStatus.applied:
                fresh_balance = await URepo.get_tokens_by_id(db, user.id)
                return {"data": existing, "charged": False, "balance": fresh_balance}
            row_id = await TMRepo.restart_existing_row(db, user.id, fp)
            if row_id is None:
                raise TrainModelInProgressException()

        cmd = build_train_worker_cmd(
            csv_path=file,
            features=feats_c,
            label=label_c,
            model_type=mt_c,
            params=params_n,
            tmp_out=tmp_path,
        )

        try:
            rc, out, err = await run_training_subprocess(cmd)
        except asyncio.CancelledError:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path)
            raise
        if rc != 0:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path)
            raise TrainingFailedException(log_detail=(err.strip() or "no stderr"))

        try:
            metrics = TrainModelService._parse_metrics_or_raise(out)
        except ValueError as e:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path)
            raise TrainingFailedException(log_detail=f"metrics-parse failed: {e!r}") from e

        try:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            applied = await TMRepo.mark_applied(db, trained_model_id=row_id, metrics=metrics)
            if not applied:
                raise TrainingFailedException(
                    log_detail=f"apply state mismatch: id={row_id}, fp={fp} (expected pending)"
                )

        except asyncio.CancelledError:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path)
            raise
        except Exception:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path)
            raise

        try:
            move_temp_to_final(tmp_path, final_path)
        except asyncio.CancelledError:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path, final_path)
            raise
        except ArtifactWriteException:
            await TrainModelService._fail_and_cleanup(db, row_id, tmp_path, final_path)
            raise

        ts = applied.created_at.isoformat()
        await invalidate_global_models_cache(redis, ts)

        log_action(
            "train_model_has_been_made",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=action.cost,
            balance_after=balance
        )

        return {"data": applied, "charged": True, "balance": balance}

    @staticmethod
    async def get_user_models(db: AsyncSession, user: User) -> list[TrainedModel]:
        models = await TMRepo.get_user_models(db, user.id)

        log_action(
            event="user_viewed_his_training_models",
            user_id=user.id,
            username=user.username,
            charged=False,
        )
        return models

    @staticmethod
    async def get_all_users_models(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return all users' trained models, with per-user version-based billing.

        Billing rule (Model B / per-user):
        - If no models exist globally → no charge.
        - Each user is charged at most once per dataset version.
        - When models data changes (new model, delete user, etc.),
          the global version changes, and each user will be charged
          once again when they first view that new version.
        """

        list_key = "models:all:list"
        ver_key = "models:all:version"
        user_seen_key = f"models:all:last_seen:{user.id}"

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
                rows = await TMRepo.get_all_users_models(db)
                data = [
                    TrainedModelResponse.model_validate(m).model_dump(mode="json")
                    for m in rows
                ]
                await CRepo.set_list(redis, list_key, data)
        else:
            rows = await TMRepo.get_all_users_models(db)
            data = [
                TrainedModelResponse.model_validate(m).model_dump(mode="json")
                for m in rows
            ]

            await CRepo.set_list(redis, list_key, data)
            await CRepo.set_version(redis, ver_key, db_ver)

        user_seen_ver = await CRepo.get_version(redis, user_seen_key)

        if user_seen_ver == db_ver:
            charged = False
            balance = user.tokens
        else:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            charged = True
            await CRepo.set_version(redis, user_seen_key, db_ver)

        log_action(
            event="user_viewed_all_users_training_models",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=action.cost,
            balance_after=balance,
        )

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_user_models_internal(db: AsyncSession, user: User) -> list[TrainedModel]:
        """
        Internal, non-billed metadata fetch.
        Used only for UX composition (prediction form, etc.)
        """
        return await TMRepo.get_user_models(db, user.id)

    @staticmethod
    async def _fail_and_cleanup(
            db,
            row_id: int,
            tmp_path: Optional[str] = None,
            final_path: Optional[str] = None) -> None:

        with suppress(SQLAlchemyError):
            await TMRepo.mark_failed(db, trained_model_id=row_id)

        safe_unlink(tmp_path)
        safe_unlink(final_path)

    @staticmethod
    def _parse_metrics_or_raise(txt: str) -> dict:
        import json
        try:
            return json.loads(txt or "{}")
        except (TypeError, json.JSONDecodeError) as e:
            raise ValueError("Malformed metrics JSON") from e

    @staticmethod
    def _charged_marker_key(user_id: int, fp: str) -> str:
        return f"train:charged:{user_id}:{fp}"