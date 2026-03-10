import asyncio
import json
import pandas as pd
from fastapi import Request
from contextlib import suppress
from typing import Dict, Any, Optional
from redis.asyncio.client import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from app.exceptions.base import BaseAppException
from app.exceptions.train_model import TrainModelInProgressException, TrainingFailedException
from app.exceptions.artifact import ArtifactWriteException
from app.models.orm_models.users import User
from app.models.orm_models.trained_models import TrainedModel
from app.models.pydantic_models.train_model import TrainedModelResponse
from app.models.enums import ActionType, RowStatus
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.repositories.seen_version_repository import SeenVersionRepository as SVRepo
from app.utils.validators import (
    ensure_csv_valid, ensure_label_valid, ensure_features_valid,
    ensure_model_type_valid, normalize_params, ensure_params_valid,
    normalize_meta_for_fingerprint, validate_param_values
)
from app.models.ml_models.model_strategy_factory import get_model_strategy
from app.utils.fingerprint_hashing import compute_training_fingerprint
from app.utils.files import unique_model_path, temp_path_for, move_temp_to_final, safe_unlink
from app.utils.cache_invalidation import invalidate_global_models_cache
from app.workers.procs import build_train_worker_cmd, run_training_subprocess
from app.core.logs import log_action, errors


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
            model_params: dict[str, Any],
            action: ActionType,
            request: Request | None = None,
    ) -> dict[str, Any]:
        """
    Train an ML model end-to-end using short DB transactions and an external worker process.

    Design:
        - Avoid holding a DB transaction open during CPU-heavy training by the subprocess.
        - Ensure idempotency using UNIQUE(user_id, fingerprint) with a pending/applied/failed lifecycle.
        - Publish the artifact (tmp -> final) before marking the DB row as applied.
        - Optionally detect client disconnect and terminate the worker best-effort.

    Args:
        db: Async SQLAlchemy session (per-request).
        redis: Redis client (used for best-effort cache invalidation).
        user: Authenticated user ORM instance.
        file: Filesystem path to the uploaded CSV (already persisted to disk).
        model_type: Model family identifier (e.g., "linear", "logistic", "random_forest").
        features: List of selected feature column names.
        label: Target/label column name.
        model_params: Strategy hyperparameters (already JSON-parsed).
        action: ActionType defining token cost and audit metadata.
        request: Optional FastAPI Request used for disconnect detection.

    Returns:
        dict[str, Any]:
            {
              "data": TrainedModel,   # ORM row (applied or existing applied)
              "charged": bool,        # True if tokens were deducted for this call
              "balance": int          # resulting balance (or fresh balance for replay)
            }

    Raises:
        TrainModelInProgressException: If an identical training request is already pending (or race detected).
        TrainingFailedException: If training fails, publishing fails, or DB apply fails.
        BaseAppException: Propagated for known validation/billing/domain errors.
        asyncio.CancelledError: If the request task is cancelled by the server runtime.
    """

        try:
            label_c, feats_c, mt_c, params_n, fp, feature_schema = await asyncio.to_thread(
                TrainModelService._prepare_training_inputs,
                file,
                model_type,
                features,
                label,
                model_params,
            )
        except BaseAppException:
            raise
        except Exception as e:
            raise TrainingFailedException(log_detail=f"prep failed: {e!r}") from e

        final_path = unique_model_path(user, fp)
        tmp_path = temp_path_for(final_path)

        async with db.begin():
            row_id = await TMRepo.try_insert_pending(
                db,
                user_id=user.id,
                model_type=mt_c,
                features=feats_c,
                model_params=params_n,
                label=label_c,
                feature_schema=feature_schema,
                fingerprint=fp,
                model_path=final_path,
            )

            if row_id is None:
                existing = await TMRepo.get_by_user_fingerprint(db, user.id, fp)

                if not existing or existing.status == RowStatus.pending:
                    raise TrainModelInProgressException()

                if existing.status == RowStatus.applied:
                    fresh_balance = await URepo.get_tokens_by_id(db, user.id)
                    return {"data": existing, "charged": False, "balance": fresh_balance}

                if existing.status == RowStatus.failed:
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
            rc, out, err = await run_training_subprocess(cmd, request=request)
        except asyncio.CancelledError:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path)
            raise
        except BaseAppException:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path)
            raise
        except Exception as e:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path)
            raise TrainingFailedException(log_detail=f"worker crashed: {e!r}") from e

        if rc != 0:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path)
            raise TrainingFailedException(log_detail=f"worker failed rc={rc}: {(err.strip() or 'no stderr')}")

        try:
            metrics = TrainModelService._parse_metrics_or_raise(out)
        except Exception as e:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path)
            raise TrainingFailedException(log_detail=f"metrics parse failed: {e!r}") from e

        try:
            move_temp_to_final(tmp_path, final_path)
        except ArtifactWriteException as e:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path, final_path=final_path)
            raise TrainingFailedException(log_detail=f"publish failed: {e.log_detail}") from e
        except Exception as e:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, tmp_path=tmp_path, final_path=final_path)
            raise TrainingFailedException(log_detail=f"publish unexpected error: {e!r}") from e

        try:
            async with db.begin():
                balance = await URepo.update_tokens(db, user.id, action.cost)
                applied = await TMRepo.mark_applied(db, trained_model_id=row_id, metrics=metrics)
                if not applied:
                    raise TrainingFailedException(log_detail=f"apply mismatch: id={row_id} expected pending")
        except BaseAppException:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, final_path=final_path)
            raise
        except Exception as e:
            await TrainModelService._mark_failed_and_cleanup_files(db, row_id, final_path=final_path)
            raise TrainingFailedException(log_detail=f"db apply failed: {e!r}") from e

        with suppress(RedisError, asyncio.TimeoutError):
            await invalidate_global_models_cache(redis, applied.created_at.isoformat())

        try:
            log_action(
                "train_model_has_been_made",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=action.cost,
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed in train_model: %r", e)

        return {"data": applied, "charged": True, "balance": balance}

    @staticmethod
    async def get_user_models(db: AsyncSession, user: User) -> list[TrainedModel]:
        """
        Fetch all trained models for the authenticated user.

        Notes:
            - No token charge.
            - No side effects (read-only).
            - Returns an empty list if the user has no models.

        Args:
            db: Async SQLAlchemy session.
            user: Authenticated user ORM instance.

        Returns:
            list[TrainedModel]: ORM rows ordered by created_at (repository-defined ordering).

        Raises:
            None explicitly. Any unexpected DB errors will propagate.
        """

        models = await TMRepo.get_user_models(db, user.id)

        try:
            log_action(
                event="user_viewed_his_training_models",
                user_id=user.id,
                username=user.username,
                charged=False,
            )
        except Exception as e:
            errors.exception("log_action failed in get_user_models: %r", e)

        return models

    @staticmethod
    async def get_all_users_models(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return all users' trained models (ACTIVE users only) with per-user-per-version billing.

        Billing correctness:
            - Version source of truth is Postgres: db_ver := max(trained_models.created_at) across active users.
            - A user is charged at most once per version, tracked durably in Postgres via user_seen_versions.

        Performance:
            - The heavy payload (the list of models) is cached in Redis keyed by version.
            - Redis is best-effort only. If Redis is down, fall back to DB and still keep billing correct.

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated viewer user.
            action: ActionType that defines token cost (metadata charge).

        Returns:
            dict[str, Any]:
                {
                  "data": list[dict],   # list of TrainedModelResponse dicts
                  "charged": bool,
                  "balance": int
                }
        """

        resource = "models:all"
        list_key = "models:all:list"
        ver_key = "models:all:version"

        async with db.begin():
            db_ver_dt = await TMRepo.get_latest_created_at_all_users(db)
            if db_ver_dt is None:
                return {"data": [], "charged": False, "balance": user.tokens}

            db_ver = db_ver_dt.isoformat()

            first_time = await SVRepo.insert_seen(
                db,
                user_id=user.id,
                resource=resource,
                version=db_ver,
            )

            if first_time:
                balance = await URepo.update_tokens(db, user.id, action.cost)
                charged = True
            else:
                balance = await URepo.get_tokens_by_id(db, user.id)
                charged = False

        data: list[dict] | None = None

        with suppress(RedisError, asyncio.TimeoutError, Exception):
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                cached = await CRepo.get_list(redis, list_key)
                if cached is not None:
                    data = cached

        if data is None:
            rows = await TMRepo.get_all_users_models(db)
            data = [TrainedModelResponse.model_validate(m).model_dump(mode="json") for m in rows]

            with suppress(RedisError, asyncio.TimeoutError, Exception):
                await CRepo.set_list(redis, list_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="user_viewed_all_users_training_models",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=(action.cost if charged else 0),
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed: %r", e)

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def get_user_models_internal(db: AsyncSession, user: User) -> list[TrainedModel]:
        """
        Internal, non-billed metadata fetch.
        Used only for UX composition (prediction form, etc.)
        """

        return await TMRepo.get_user_models(db, user.id)

    @staticmethod
    def _prepare_training_inputs(
            file: str,
            model_type: str,
            features: list[str],
            label: str,
            model_params: dict[str, Any],
    ) -> tuple[str, list[str], str, dict[str, Any], str, dict[str, str]]:
        """
        Perform blocking/CPU work required before launching the training subprocess.

        This function is intended to run in a worker thread via asyncio.to_thread().

        Responsibilities:
            - Read CSV from disk (pandas) and validate format.
            - Normalize and validate user inputs (label/features/model_type).
            - Validate strategy compatibility (classification vs regression).
            - Validate hyperparameters against estimator and semantic rules.
            - Compute a deterministic fingerprint for idempotency and caching.
            - Build feature_schema metadata (numeric vs categorical).

        Args:
            file: Filesystem path to the uploaded CSV.
            model_type: Requested model family identifier.
            features: List of raw feature names.
            label: Raw label column name.
            model_params: Raw model hyperparameters (dict).

        Returns:
            tuple[
                str,               # label_c (validated/normalized label)
                list[str],         # feats_c (validated/normalized features)
                str,               # mt_c (validated/normalized model_type)
                dict[str, Any],    # params_n (normalized params)
                str,               # fp (training fingerprint)
                dict[str, str],    # feature_schema (col -> "numeric"/"categorical")
            ]

        Raises:
            BaseAppException: If validation fails (CSV/label/features/params/model type).
            Exception: Any unexpected parsing/compute errors from libraries may propagate.
        """

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

        feature_schema: dict[str, str] = {}
        for col in feats_c:
            feature_schema[col] = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

        return label_c, feats_c, mt_c, params_n, fp, feature_schema

    @staticmethod
    async def _mark_failed_and_cleanup_files(
            db: AsyncSession,
            row_id: int,
            tmp_path: Optional[str] = None,
            final_path: Optional[str] = None,
    ) -> None:
        """
        Best-effort failure handler for training jobs.

        Behavior:
            - Attempts to mark the DB row as FAILED inside a short transaction.
              Any DB error is swallowed to avoid masking the original failure.
            - Deletes tmp and/or final artifact files best-effort.

        Args:
            db: Async SQLAlchemy session.
            row_id: Primary key of the trained_models row to mark failed.
            tmp_path: Optional path to the tmp artifact file to delete.
            final_path: Optional path to the final artifact file to delete.

        Returns:
            None

        Raises:
            None explicitly. Exceptions are suppressed for DB updates and file deletion.
        """

        with suppress(SQLAlchemyError):
            async with db.begin():
                await TMRepo.mark_failed(db, trained_model_id=row_id)

        safe_unlink(tmp_path)
        safe_unlink(final_path)

    @staticmethod
    def _parse_metrics_or_raise(txt: str) -> dict:
        """
        Parse worker stdout into a JSON metrics dictionary.

        Args:
            txt: Raw stdout text produced by the training worker.

        Returns:
            dict: Parsed metrics (empty dict if txt is empty/whitespace).

        Raises:
            ValueError: If stdout is not valid JSON.
        """

        try:
            return json.loads(txt or "{}")
        except (TypeError, json.JSONDecodeError) as e:
            raise ValueError("Malformed metrics JSON") from e
