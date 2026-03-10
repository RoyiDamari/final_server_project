from typing import Optional, Any
from datetime import datetime
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.orm_models.trained_models import TrainedModel
from app.models.enums import RowStatus


class TrainModelRepository:
    """
    DB access layer for TrainedModel lifecycle.

    Key patterns:
      - Idempotency gate on UNIQUE(user_id, fingerprint).
      - "pending -> applied/failed" state machine.
      - Queries used by both training execution and dashboards.
    """

    @staticmethod
    async def try_insert_pending(
            db: AsyncSession,
            user_id: int,
            model_type: str,
            features: list[str],
            model_params: dict[str, Any],
            label: str,
            feature_schema: dict[str, Any],
            fingerprint: str,
            model_path: str,
    ) -> Optional[int]:
        """
        Insert a new TrainedModel row in PENDING state if it doesn't already exist.

        Idempotency:
          - Uses ON CONFLICT DO NOTHING on (user_id, fingerprint).
          - If a row already exists (pending/applied/failed), returns None and caller
            must fetch the existing row and decide what to do.

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the model.
            model_type: Stored strategy type ("linear", "logistic", "random_forest", ...).
            features: Clean list of feature column names.
            model_params: Normalized estimator hyperparameters.
            label: Clean target column name.
            feature_schema: Feature name -> "numeric"/"categorical" (for UI/UX).
            fingerprint: Deterministic hash identifying "same training request".
            model_path: Final artifact path where model will live.

        Returns:
            Optional[int]:
              - TrainedModel.id if inserted now
              - None if conflict (row already exists)
        """

        values = dict(
            user_id=user_id,
            model_type=model_type,
            features=list(features),
            model_params=model_params,
            label=label,
            feature_schema=feature_schema,
            fingerprint=fingerprint,
            model_path=model_path,
            status=RowStatus.pending,
        )
        q = (
            pg_insert(TrainedModel)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["user_id", "fingerprint"])
            .returning(TrainedModel.id)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_by_user_fingerprint(
            db: AsyncSession,
            user_id: int,
            fingerprint: str,
    ) -> Optional[TrainedModel]:
        """
        Fetch the single TrainedModel row for (user_id, fingerprint), if it exists.

        Used after a conflict in try_insert_pending() to determine whether:
          - another request is still pending
          - a previous request already applied
          - a previous request failed and can be restarted

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the model.
            fingerprint: Deterministic training fingerprint.

        Returns:
            Optional[TrainedModel]: ORM row if found, else None.
        """

        q = (
            select(TrainedModel)
            .where(
                TrainedModel.user_id == user_id,
                TrainedModel.fingerprint == fingerprint
            )
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def restart_existing_row(
            db: AsyncSession,
            user_id: int,
            fingerprint: str,
    ) -> Optional[int]:
        """
        Atomically flip a FAILED row back to PENDING so a retry can reuse the same row.

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the model.
            fingerprint: Deterministic request fingerprint.

        Returns:
            Optional[int]:
              - id if flipped now
              - None if no FAILED row matched
        """

        q = (
            update(TrainedModel)
            .where(
                TrainedModel.user_id == user_id,
                TrainedModel.fingerprint == fingerprint,
                TrainedModel.status == RowStatus.failed,
            )
            .values(
                status=RowStatus.pending,
                metrics={},
            )
            .returning(TrainedModel.id)
        )
        res = await db.execute(q)
        return res.scalar_one_or_none()

    @staticmethod
    async def mark_applied(
            db: AsyncSession,
            trained_model_id: int,
            metrics: dict[str, Any],
    ) -> Optional[TrainedModel]:
        """
        Transition PENDING -> APPLIED and store metrics.

        Guard:
          - Only updates if current status is PENDING.
          - Returns None if status mismatch (race, reconciler, double-apply attempt).

        Args:
            db: Async SQLAlchemy session.
            trained_model_id: TrainedModel primary key.
            metrics: Final metrics dict to store.

        Returns:
            Optional[TrainedModel]: Updated ORM row if applied, else None.
        """

        q = (
            update(TrainedModel)
            .where(TrainedModel.id == trained_model_id, TrainedModel.status == RowStatus.pending)
            .values(status=RowStatus.applied, metrics=metrics)
            .returning(TrainedModel)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def mark_failed(db: AsyncSession, trained_model_id: int) -> bool:
        """
        Mark a trained model as FAILED (idempotent).

        Args:
            db: Async SQLAlchemy session.
            trained_model_id: Prediction primary key.

        Returns:
            bool: True if a row was updated, False otherwise.
        """

        q = (
            update(TrainedModel)
            .where(TrainedModel.id == trained_model_id)
            .values(status=RowStatus.failed)
        )
        res = await db.execute(q)
        return (res.rowcount or 0) > 0

    @staticmethod
    async def get_latest_created_at_all_users(db: AsyncSession) -> Optional[datetime]:
        """
        Return the most recent TrainedModel.created_at across ACTIVE users.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            Optional[datetime]: max(created_at) or None if no models exist.
        """

        q = (
            select(func.max(TrainedModel.created_at)).
            where(TrainedModel.user.has(is_active=True))
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_user_models(db: AsyncSession, user_id: int) -> list[TrainedModel]:
        """
        Fetch all trained models for a single user.

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner of the trained models.

        Returns:
            list[TrainedModel]: ORM rows ordered by created_at ascending.
        """

        q = (
            select(TrainedModel)
            .where(TrainedModel.user_id == user_id)
            .order_by(TrainedModel.created_at)
        )
        return (await db.execute(q)).scalars().all()

    @staticmethod
    async def get_all_users_models(db: AsyncSession) -> list[TrainedModel]:
        """
        Fetch all trained models across ACTIVE users.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            list[TrainedModel]: ORM rows for active users ordered by created_at ascending.
        """

        q = (
            select(TrainedModel).
            where(TrainedModel.user.has(is_active=True))
            .order_by(TrainedModel.created_at)
        )
        return (await db.execute(q)).scalars().all()

    @staticmethod
    async def list_model_paths_applied(db: AsyncSession) -> set[str]:
        """
        Return model_path values for APPLIED models owned by ACTIVE users.

        Used by startup reconciliation to compute the set of artifact files that should exist.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            set[str]: Non-empty model_path strings from applied rows.
        """

        q = (
            select(TrainedModel.model_path).
            where(
                TrainedModel.status == RowStatus.applied,
                TrainedModel.model_path.is_not(None),
                TrainedModel.user.has(is_active=True)
            )
        )
        result = await db.execute(q)
        return {p for p in result.scalars().all() if p}

    @staticmethod
    async def list_ids_and_paths_by_status(
            db: AsyncSession,
            status: RowStatus,
    ) -> list[tuple[int, Optional[str]]]:
        """
        List (id, model_path) for trained_models rows in the given status (ACTIVE users only).

        Args:
            db: Async SQLAlchemy session.
            status: RowStatus to filter by.

        Returns:
            list[tuple[int, str|None]]: (trained_model_id, model_path)
        """
        q = (
            select(TrainedModel.id, TrainedModel.model_path)
            .where(
                TrainedModel.status == status,
                TrainedModel.user.has(is_active=True),
            )
        )
        rows = (await db.execute(q)).all()
        return [(int(r[0]), r[1]) for r in rows]