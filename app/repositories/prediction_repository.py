from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.orm_models.predictions import Prediction
from app.models.orm_models.trained_models import TrainedModel
from app.models.enums import RowStatus


class PredictionRepository:
    """
    DB access layer for Prediction lifecycle.

    Key patterns:
        - Idempotency gate on UNIQUE(user_id, fingerprint).
        - "pending -> applied/failed" state machine.
        - Queries used by both prediction execution and dashboards.
    """

    @staticmethod
    async def try_insert_pending(
            db: AsyncSession,
            user_id: int,
            model_id: int,
            model_type: str,
            feature_values: dict,
            fingerprint: str
    ) -> Optional[int]:
        """
        Insert a new Prediction row in PENDING state if it doesn't already exist.

        Idempotency:
            - Uses ON CONFLICT DO NOTHING on (user_id, fingerprint).
            - If a row already exists (pending/applied/failed), returns None and caller
            must fetch the existing row and decide what to do.

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the prediction.
            model_id: Trained model used for prediction.
            model_type: Stored for display/analytics ("linear", "logistic", etc.).
            feature_values: Raw feature-values dict as submitted by user.
            fingerprint: Deterministic hash identifying "same request".

        Returns:
            Optional[int]:
                - Prediction.id if inserted now
                - None if conflict (row already exists)
        """

        q = (
            pg_insert(Prediction)
            .values(
                user_id=user_id,
                model_id=model_id,
                model_type=model_type,
                feature_values=feature_values,
                prediction_result="",
                fingerprint=fingerprint,
                status=RowStatus.pending
            )
            .on_conflict_do_nothing(index_elements=["user_id", "fingerprint"])
            .returning(Prediction.id)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_by_user_fingerprint(
            db: AsyncSession,
            user_id: int,
            fingerprint: str
    ) -> Optional[Prediction]:
        """
        Fetch the single Prediction row for (user_id, fingerprint), if it exists.

        Used after a conflict in try_insert_pending() to determine whether:
            - another request is still pending
            - a previous request already applied
            - a previous request failed and can be restarted

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the prediction.
            fingerprint: Deterministic request fingerprint.

        Returns:
            Optional[Prediction]: ORM row if found, else None.
        """

        q = (
            select(Prediction).
            where(
                Prediction.user_id == user_id,
                Prediction.fingerprint == fingerprint
            )
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def restart_existing_row(db: AsyncSession, user_id: int, fingerprint: str) -> Optional[int]:
        """
        Atomically flip a FAILED row back to PENDING so a retry can reuse the same row.

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the prediction.
            fingerprint: Deterministic request fingerprint.

        Returns:
            Optional[int]:
                - Prediction.id if flipped now
                - None if no FAILED row matched (already restarted or not failed)
        """

        q = (
            update(Prediction)
            .where(
                Prediction.user_id == user_id,
                Prediction.fingerprint == fingerprint,
                Prediction.status == RowStatus.failed,
            )
            .values(status=RowStatus.pending, prediction_result="")
            .returning(Prediction.id)
        )
        res = await db.execute(q)
        return res.scalar_one_or_none()

    @staticmethod
    async def mark_applied(db: AsyncSession, pred_id: int, result: str) -> Optional[Prediction]:
        """
        Transition PENDING -> APPLIED and store prediction_result.

        Guard:
          - Only updates if current status is PENDING.
          - Returns None if status mismatch (race, reconciler, double-apply attempt).

        Args:
            db: Async SQLAlchemy session.
            pred_id: Prediction primary key.
            result: Final prediction output string.

        Returns:
            Optional[Prediction]: Updated ORM row if applied, else None.
        """

        q = (
            update(Prediction)
            .where(Prediction.id == pred_id, Prediction.status == RowStatus.pending)
            .values(status=RowStatus.applied, prediction_result=result)
            .returning(Prediction)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def mark_failed(db: AsyncSession, pred_id: int) -> bool:
        """
        Mark a prediction as FAILED (idempotent).

        Args:
            db: Async SQLAlchemy session.
            pred_id: Prediction primary key.

        Returns:
            bool: True if a row was updated, False otherwise.
        """

        q = (
            update(Prediction)
            .where(Prediction.id == pred_id)
            .values(status=RowStatus.failed)
        )
        res = await db.execute(q)
        return (res.rowcount or 0) > 0

    @staticmethod
    async def get_latest_created_at_all_users(db: AsyncSession) -> Optional[datetime]:
        """
        Return the latest Prediction.created_at across ACTIVE users.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            Optional[datetime]:
                - max(created_at) across active users
                - None if no predictions exist
        """

        q = (
            select(func.max(Prediction.created_at)).
            where(Prediction.user.has(is_active=True))
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_user_predictions(db: AsyncSession, user_id: int) -> list[Prediction]:
        """
        Return all predictions for a single user, ordered by created_at.

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the predictions.

        Returns:
            list[Prediction]: ORM rows for that user.
        """

        q = (
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at)
        )
        return (await db.execute(q)).scalars().all()

    @staticmethod
    async def get_all_users_predictions(db: AsyncSession) -> list[Prediction]:
        """
        Return all predictions for ACTIVE users, ordered by created_at.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            list[Prediction]: ORM rows for all active users.
        """

        q = (
            select(Prediction).
            where(Prediction.user.has(is_active=True)).
            order_by(Prediction.created_at)
        )
        return (await db.execute(q)).scalars().all()

    @staticmethod
    async def get_model_for_user_applied(
            db: AsyncSession,
            user_id: int,
            model_id: int
    ) -> Optional[TrainedModel]:
        """
        Fetch a user's TrainedModel only if it is in APPLIED status.

        Used by PredictionService to enforce:
          - ownership (authorization)
          - readiness (artifact should exist)

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the model.
            model_id: TrainedModel primary key.

        Returns:
            Optional[TrainedModel]:
                - ORM row if found, owned by user, and status=applied
                - None otherwise
        """

        q = (
            select(TrainedModel).
            where(
                TrainedModel.id == model_id,
                TrainedModel.user_id == user_id,
                TrainedModel.status == RowStatus.applied
            )
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def list_ids_by_status(db: AsyncSession, status: RowStatus) -> list[int]:
        """
        List prediction ids in the given status (ACTIVE users only).

        Args:
            db: Async SQLAlchemy session.
            status: RowStatus to filter by.

        Returns:
            list[int]: Prediction ids.
        """
        q = (
            select(Prediction.id)
            .where(
                Prediction.status == status,
                Prediction.user.has(is_active=True),
            )
        )
        return [int(x) for x in (await db.execute(q)).scalars().all()]