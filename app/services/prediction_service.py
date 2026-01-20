import asyncio
import pandas as pd
from redis.asyncio.client import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from contextlib import suppress
from typing import Any, Dict, Tuple
from app.models.enums import ActionType, RowStatus
from app.exceptions.prediction import (
    PredictionInProgressException,
    ModelNotFoundException,
    ArtifactMissingException,
    FeatureMismatchException,
    PredictionFailedException,
)
from app.models.orm_models.users import User
from app.models.orm_models.trained_models import TrainedModel
from app.models.orm_models.predictions import Prediction
from app.models.pydantic_models.prediction import PredictionRequest, PredictionResponse
from app.repositories.prediction_repository import PredictionRepository as PRepo
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.utils.cache_invalidation import invalidate_global_predictions_cache
from app.core.logs import log_action
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
    ) -> dict:
        """
        Flow (idempotent & cancel-safe):

        1) Load model row (must be owned by user and status=applied) + artifact from disk.
        2) Idempotent gate on (user_id, model_id, idempotency_key):
           - insert_pending_returning → pred_id
           - else duplicate: if applied → return it; if pending → 409; if failed → restart_failed_returning.
        3) Run prediction in a thread with timeout (convert hangs/errors to PredictionFailedException).
        4) In a tx: charge tokens and mark_applied_returning(result). Guard state mismatch.
        5) Log activity and return the applied row.
        """
        tm_row, loaded_model = await PredictionService._load_model_row_for_user(
            db, user_id=user.id, model_id=request.model_id
        )

        fp = compute_prediction_fingerprint(
            model_id=tm_row.id,
            feature_values=request.feature_values,
        )

        row_id = await PRepo.try_insert_pending(
            db=db,
            user_id=user.id,
            model_id=tm_row.id,
            model_type=tm_row.model_type,
            input_data=request.feature_values,
            fingerprint=fp,
        )

        if row_id is None:
            existing = await PRepo.get_by_user_fingerprint(db, user.id, fp)
            if not existing or existing.status == RowStatus.pending:
                raise PredictionInProgressException()
            if existing.status == RowStatus.applied:
                fresh_balance = await URepo.get_tokens_by_id(db, user.id)
                return {"data": existing, "charged": False, "balance": fresh_balance }
            row_id = await PRepo.restart_existing_row(db, user.id, fp)
            if row_id is None:
                raise PredictionInProgressException()

        try:
            result_str = await PredictionService._run_prediction(
                model=loaded_model,
                feature_order=tm_row.features,
                provided=request.feature_values,
                timeout_s=10.0,
            )
        except asyncio.CancelledError:
            await PredictionService._fail_prediction_safely(db, row_id)
            raise
        except Exception as e:
            await PredictionService._fail_prediction_safely(db, row_id)
            raise PredictionFailedException(log_detail=f"predict error: {e!r}") from e

        try:
            balance = await URepo.update_tokens(db, user.id, action.cost)
            applied = await PRepo.mark_applied(db, row_id, result_str)
            if not applied:
                raise PredictionFailedException(
                    log_detail=f"apply state mismatch: id={row_id} (expected pending)"
                )
        except asyncio.CancelledError:
            await PredictionService._fail_prediction_safely(db, row_id)
            raise
        except PredictionFailedException:
            await PredictionService._fail_prediction_safely(db, row_id)
            raise

        ts = applied.created_at.isoformat()
        await invalidate_global_predictions_cache(redis, ts)

        log_action(
            "prediction_has_been_made",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=action.cost,
            balance_after=balance,
        )

        return {"data": applied, "charged": True, "balance": balance}

    @staticmethod
    async def get_user_predictions(db: AsyncSession, user: User) -> list[Prediction]:
        """
        Return the authenticated user's predictions.

        - No token charge
        - No side effects
        - Empty list if none exist
        """

        predictions = await PRepo.get_user_predictions(db, user.id)

        log_action(
            event="user_viewed_his_predictions",
            user_id=user.id,
            username=user.username,
            charged=False,
        )

        return predictions

    @staticmethod
    async def get_all_users_predictions(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return all users' predictions (for metadata dashboard).

        Billing rule (Version-D, per-user-per-version):
        - If NO predictions globally → no charge for anyone.
        - Version is defined by max(created_at) across all active users.
        - Each user is charged AT MOST once per dataset version.
        - When new prediction is created (any user) or cache bumped, version changes.
        """

        list_key = "preds:all:list"
        ver_key = "preds:all:version"
        seen_key = f"preds:all:last_seen:{user.id}"

        db_ver_dt = await PRepo.get_latest_created_at_all_users(db)
        if db_ver_dt is None:
            return {"data": [], "charged": False, "balance": user.tokens}

        db_ver = db_ver_dt.isoformat()
        redis_ver = await CRepo.get_version(redis, ver_key)

        if redis_ver == db_ver:
            cached = await CRepo.get_list(redis, list_key)
            if cached is not None:
                data = cached
            else:
                rows = await PRepo.get_all_users_predictions(db)
                data = [
                    PredictionResponse.model_validate(m).model_dump(mode="json")
                    for m in rows
                ]
                await CRepo.set_list(redis, list_key, data)
        else:
            rows = await PRepo.get_all_users_predictions(db)
            data = [
                PredictionResponse.model_validate(m).model_dump(mode="json")
                for m in rows
            ]
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
            event="user_viewed_all_users_predictions",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=action.cost,
            balance_after=balance,
        )

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    def _ensure_feature_keys_match(expected: list[str], provided: Dict[str, Any]) -> list[str]:
        """
        Validate keys match exactly; return the order to feed values to model.
        Pure function → sync.
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
        1) DB: get the user's model only if 'applied'.
        2) FS: load the artifact with clear 500s and log_detail.
        """
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
        Do the actual prediction in-process (sync). To keep the event loop free,
        call it via asyncio.to_thread from async code.
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
        Compose ordered values → run prediction in a worker thread → return string result.
        Raises PredictionFailedException on timeout/other unexpected errors.
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
    async def _fail_prediction_safely(db: AsyncSession, row_id: int) -> None:
        with suppress(SQLAlchemyError):
            await PRepo.mark_failed(db, row_id)
