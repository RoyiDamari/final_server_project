import asyncio
import pandas as pd
from contextlib import suppress
from redis.exceptions import RedisError
from redis.asyncio.client import Redis
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, Tuple
from app.models.enums import ActionType, RowStatus
from app.exceptions.prediction import (
    PredictionInProgressException,
    ModelNotFoundException,
    FeatureMismatchException,
    PredictionFailedException,
)
from app.exceptions.base import BaseAppException
from app.exceptions.artifact import ArtifactMissingException
from app.models.orm_models.users import User
from app.models.orm_models.trained_models import TrainedModel
from app.models.orm_models.predictions import Prediction
from app.models.pydantic_models.prediction import PredictionRequest, PredictionResponse
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.repositories.prediction_repository import PredictionRepository as PRepo
from app.repositories.seen_version_repository import SeenVersionRepository as SVRepo
from app.utils.cache_invalidation import invalidate_global_predictions_cache
from app.core.logs import log_action, errors
from app.utils.fingerprint_hashing import compute_prediction_fingerprint
from app.utils.files import load_joblib_model


class PredictionService:
    @staticmethod
    async def predict(
            db: AsyncSession,
            redis: Redis,
            user: User,
            request: PredictionRequest,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Make a prediction with idempotency and short DB transactions.

        Design goals:
        - Keep DB transactions short (do NOT hold a transaction during prediction compute).
        - Preserve idempotency via (user_id, fingerprint) unique constraint.
        - If compute/apply fails, mark the pending row as failed best-effort.

        Args:
            db: Async SQLAlchemy session (per-request).
            redis: Redis client (for cache invalidation).
            user: Authenticated user ORM.
            request: PredictionRequest (model_id + feature_values).
            action: ActionType enum (token cost).

        Returns:
            dict[str, Any]: {"data": <Prediction ORM>, "charged": bool, "balance": int}

        Raises:
            ModelNotFoundException: if model not found / not owned / not applied.
            PredictionInProgressException: if identical request is currently pending.
            PredictionFailedException: If prediction fails, publishing fails, or DB apply fails.
        """

        tm_row, loaded_model = await PredictionService._load_model_row_for_user(
            db=db,
            user_id=user.id,
            model_id=request.model_id,
        )

        fp = compute_prediction_fingerprint(
            model_id=tm_row.id,
            feature_values=request.feature_values,
        )

        async with db.begin():
            pred_id = await PRepo.try_insert_pending(
                db=db,
                user_id=user.id,
                model_id=tm_row.id,
                model_type=tm_row.model_type,
                feature_values=request.feature_values,
                fingerprint=fp,
            )

            if pred_id is None:
                existing = await PRepo.get_by_user_fingerprint(db, user.id, fp)

                if not existing or existing.status == RowStatus.pending:
                    raise PredictionInProgressException()

                if existing.status == RowStatus.applied:
                    fresh_balance = await URepo.get_tokens_by_id(db, user.id)
                    return {"data": existing, "charged": False, "balance": fresh_balance}

                if existing.status == RowStatus.failed:
                    restarted_id = await PRepo.restart_existing_row(db, user.id, fp)
                    if restarted_id is None:
                        raise PredictionInProgressException()

        try:
            result_str = await PredictionService._run_prediction(
                model=loaded_model,
                feature_order=tm_row.features,
                provided=request.feature_values,
                timeout_s=30.0,
            )
        except BaseAppException:
            await PredictionService._mark_failed(db, pred_id)
            raise
        except Exception as e:
            await PredictionService._mark_failed(db, pred_id)
            raise PredictionFailedException(log_detail=f"predict compute failed: {e!r}") from e

        try:
            async with db.begin():
                balance = await URepo.update_tokens(db, user.id, action.cost)
                applied = await PRepo.mark_applied(db, pred_id, result_str)
                if not applied:
                    raise PredictionFailedException(
                        log_detail=f"apply state mismatch: id={pred_id} (expected pending)"
                    )
        except BaseAppException:
            await PredictionService._mark_failed(db, pred_id)
            raise
        except Exception as e:
            await PredictionService._mark_failed(db, pred_id)
            raise PredictionFailedException(log_detail=f"apply step failed: {e!r}") from e

        with suppress(RedisError, asyncio.TimeoutError):  # noqa
            await invalidate_global_predictions_cache(redis, applied.created_at.isoformat())

        try:
            log_action(
                "prediction_has_been_made",
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
    async def get_user_predictions(db: AsyncSession, user: User) -> list[Prediction]:
        """
        Fetch all predictions for the authenticated user.

        Args:
            db: Async SQLAlchemy session.
            user: Authenticated user ORM instance.

        Returns:
            list[Prediction]: Prediction ORM rows for the user (ordering is defined by repository query).
        """

        predictions = await PRepo.get_user_predictions(db, user.id)

        try:
            log_action(
                event="user_viewed_his_predictions",
                user_id=user.id,
                username=user.username,
                charged=False,
            )
        except Exception as e:
            errors.exception("log_action failed in get_user_predictions: %r", e)

        return predictions

    @staticmethod
    async def get_all_users_predictions(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return all users' predictions (ACTIVE users only) with per-user-per-version billing.

        Billing correctness:
            - Version source of truth is Postgres: db_ver := max(predictions.created_at) across active users.
            - A user is charged at most once per version, tracked durably in Postgres via user_seen_versions.

        Performance:
            - The heavy payload (the list of predictions) is cached in Redis keyed by version.
            - Redis is best-effort only. If Redis is down, fall back to DB and still keep billing correct.

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated viewer user.
            action: ActionType that defines token cost (metadata charge).

        Returns:
            dict[str, Any]:
                {
                  "data": list[dict],   # list of PredictionResponse dicts
                  "charged": bool,
                  "balance": int
                }
        """

        resource = "preds:all"
        list_key = "preds:all:list"
        ver_key = "preds:all:version"

        async with db.begin():
            db_ver_dt = await PRepo.get_latest_created_at_all_users(db)
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

        with suppress(RedisError, asyncio.TimeoutError, Exception):
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                cached = await CRepo.get_list(redis, list_key)
                if cached is not None:
                    data = cached

        if data is None:
            rows = await PRepo.get_all_users_predictions(db)
            data = [PredictionResponse.model_validate(m).model_dump(mode="json") for m in rows]

            with suppress(RedisError, asyncio.TimeoutError, Exception):
                await CRepo.set_list(redis, list_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="user_viewed_all_users_predictions",
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
    def _ensure_feature_keys_match(expected: list[str], provided: Dict[str, Any]) -> list[str]:
        """
        Validate that the provided feature keys match the model's expected feature set.

        This enforces an exact set match (order-independent), then returns the expected order
        so downstream code can build a DataFrame with consistent column order.

        Args:
            expected: Feature names stored with the trained model (authoritative list/order).
            provided: Feature-value mapping submitted by the user.

        Returns:
            list[str]: The feature names in the correct order to feed into the model.

        Raises:
            FeatureMismatchException: If the provided keys do not match the expected keys.
        """

        provided_keys = list(provided.keys())
        if set(provided_keys) != set(expected):
            raise FeatureMismatchException(
                f"Provided features do not match model features (expected={expected}, got={provided_keys})"
            )
        return list(expected)

    @staticmethod
    async def _load_model_row_for_user(
            db: AsyncSession,
            user_id: int,
            model_id: int
    ) -> Tuple[TrainedModel, tuple]:
        """
        Load a user's trained model row (must be APPLIED) and its on-disk artifact.

        Steps:
            1) DB lookup: ensure the trained model exists, belongs to the user, and status=applied.
            2) Filesystem load: deserialize the joblib artifact from model_path.

        Args:
            db: Async SQLAlchemy session.
            user_id: Authenticated user's id.
            model_id: Target trained model id.

        Returns:
            tuple[TrainedModel, Any]:
                (trained_model_row, loaded_model_object)

        Raises:
            ModelNotFoundException: If the model does not exist / not owned / not applied.
            ArtifactMissingException: If the artifact file is missing or cannot be accessed.
            PredictionFailedException: If artifact loading fails unexpectedly.
        """

        async with db.begin():
            row = await PRepo.get_model_for_user_applied(db, user_id, model_id)
            if row is None:
                raise ModelNotFoundException()

        path = row.model_path or ""

        try:
            model = load_joblib_model(path)
            return row, model

        except FileNotFoundError as e:
            raise ArtifactMissingException(
                log_detail=f"artifact not found path={path!r} err={e!r}"
            ) from e
        except PermissionError as e:
            raise ArtifactMissingException(
                log_detail=f"artifact permission error path={path!r} err={e!r}"
            ) from e
        except OSError as e:
            raise ArtifactMissingException(
                log_detail=f"artifact os error path={path!r} errno={getattr(e, 'errno', None)} err={e!r}"
            ) from e
        except Exception as e:
            raise PredictionFailedException(
                log_detail=f"artifact load unexpected path={path!r} err={e!r}"
            ) from e

    @staticmethod
    def _predict(model: Any, ordered_keys, provided) -> str:
        """
        Run a single prediction synchronously using an already-loaded model.

        Intended to be executed via asyncio.to_thread() to avoid blocking the event loop.

        Args:
            model: Loaded model object (e.g., sklearn Pipeline).
            ordered_keys: Ordered feature names matching model training order.
            provided: Feature-value mapping for one inference request.

        Returns:
            str: The prediction result converted to string.
        """

        df = pd.DataFrame([provided], columns=ordered_keys)
        y = model.predict(df)
        return str(y[0])

    @staticmethod
    async def _run_prediction(
            model: Any,
            feature_order: list[str],
            provided: Dict[str, Any],
            timeout_s: float = 10.0
    ) -> str:
        """
        Validate inputs and execute prediction in a worker thread with a timeout.

        Flow:
            1) Validate provided features match the trained model feature set.
            2) Run the synchronous _predict() in a thread via asyncio.to_thread().
            3) Apply an asyncio.wait_for timeout guard.

        Args:
            model: Loaded model object (e.g., sklearn Pipeline).
            feature_order: Feature names stored with the trained model.
            provided: Feature-value mapping submitted by the user.
            timeout_s: Maximum time to wait for prediction before failing.

        Returns:
            str: Prediction output as string.

        Raises:
            PredictionFailedException: If prediction times out or an unexpected error occurs.
            FeatureMismatchException: If provided features do not match expected features.
        """

        ordered_keys = PredictionService._ensure_feature_keys_match(feature_order, provided)

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    PredictionService._predict,
                    model,
                    ordered_keys,
                    provided
                ),
                timeout=timeout_s
            )
        except asyncio.TimeoutError as e:
            raise PredictionFailedException(log_detail=f"predict timeout after {timeout_s}s") from e
        except Exception as e:
            raise PredictionFailedException(log_detail=f"predict error: {e!r}") from e

    @staticmethod
    async def _mark_failed(db: AsyncSession, pred_id: int) -> None:
        """
        Best-effort: mark the prediction row as failed in a short transaction.
        Never raises.
        """

        with suppress(SQLAlchemyError):
            async with db.begin():
                await PRepo.mark_failed(db, pred_id)
