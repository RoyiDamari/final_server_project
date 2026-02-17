from typing import Optional, Any
from datetime import datetime
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.orm_models.trained_models import TrainedModel
from app.models.enums import RowStatus


class TrainModelRepository:
    """
    Repository for TrainedModel idempotent lifecycle:
      - insert_pending_returning: gate on (user_id, fingerprint)
      - get_by_user_fingerprint: read current row (any status)
      - restart_existing_row_returning: failed → pending
      - mark_failed: set status failed (idempotent)
      - mark_applied_returning: pending → applied and return the row
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
        Try to create a 'pending' row for (user_id, fingerprint) exactly once.
        Returns:
            int id if inserted now
            None if a row already exists (any status) → caller inspects and decides
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
        Return the single TrainedModel for (user_id, fingerprint) or None.
        Relies on a UNIQUE(user_id, fingerprint) constraint.
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
    async def mark_applied(
        db: AsyncSession,
        trained_model_id: int,
        metrics: dict[str, Any],
    ) -> Optional[TrainedModel]:
        """
        Transition pending → applied and return the ORM row.
        Returns None if the row wasn't pending (e.g., race or reconciler).
        """
        q = (
            update(TrainedModel)
            .where(TrainedModel.id == trained_model_id, TrainedModel.status == RowStatus.pending)
            .values(status=RowStatus.applied, metrics=metrics)
            .returning(TrainedModel)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_latest_created_at_all_users(db: AsyncSession) -> Optional[datetime]:
        """
        Return datetime of the latest created model among ACTIVE users,
        or None if no trained models exist.
        """
        q = (
            select(func.max(TrainedModel.created_at)).
            where(TrainedModel.user.has(is_active=True))
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_user_models(db: AsyncSession, user_id: int) -> list[TrainedModel]:
        """
        Fetch all trained models for a given user.

        Args:
            db: Async session.
            user_id: The user ID.

        Returns:
            List[TrainedModel]: ORM rows for that user.
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
        Fetch all trained models across all users.

        Args:
            db: Async session.

        Returns:
            List[TrainedModel]: ORM rows for all users.
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
        Return a set of model_path strings for rows that are in applied state.
        Paths may be relative ("saved_models/...") or absolute depending on your setup.
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