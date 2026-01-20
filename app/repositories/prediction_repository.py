from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.orm_models.predictions import Prediction
from app.models.orm_models.trained_models import TrainedModel
from app.models.enums import RowStatus

class PredictionRepository:
    @staticmethod
    async def get_model_for_user_applied(
            db: AsyncSession,
            user_id: int,
            model_id: int
    ) -> Optional[TrainedModel]:
        """
        Return the user's TrainedModel row only if it is in 'applied' status.
        Otherwise, None (not found / not owned / not ready).
        """
        stmt = select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.user_id == user_id,
            TrainedModel.status == RowStatus.applied,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def try_insert_pending(
            db: AsyncSession,
            user_id: int,
            model_id: int,
            model_type: str,
            input_data: dict,
            fingerprint: str
    ) -> Optional[int]:
        stmt = (
            pg_insert(Prediction)
            .values(
                user_id=user_id,
                model_id=model_id,
                model_type=model_type,
                input_data=input_data,
                prediction_result="",
                fingerprint=fingerprint,
                status=RowStatus.pending
            )
            .on_conflict_do_nothing(index_elements=["user_id", "fingerprint"])
            .returning(Prediction.id)
        )
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    @staticmethod
    async def get_by_user_fingerprint(
            db: AsyncSession,
            user_id: int,
            findeprint: str
    ) -> Optional[Prediction]:
        q = select(Prediction).where(
            Prediction.user_id == user_id,
            Prediction.fingerprint == findeprint,
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def restart_existing_row(
            db: AsyncSession,
            user_id: int,
            fingerprint: str
    ) -> Optional[int]:
        upd = (
            update(Prediction)
            .where(
                Prediction.user_id == user_id,
                Prediction.fingerprint == fingerprint,
                Prediction.status == RowStatus.failed,
            )
            .values(status=RowStatus.pending)
            .returning(Prediction.id)
        )
        res = await db.execute(upd)
        return res.scalar_one_or_none()

    @staticmethod
    async def mark_applied(db: AsyncSession, pred_id: int, result: str) -> Optional[Prediction]:
        upd = (
            update(Prediction)
            .where(Prediction.id == pred_id, Prediction.status == RowStatus.pending)
            .values(status=RowStatus.applied, prediction_result=result)
            .returning(Prediction.id)
        )
        res = await db.execute(upd)
        new_id = res.scalar_one_or_none()
        return await db.get(Prediction, new_id) if new_id is not None else None

    @staticmethod
    async def mark_failed(db: AsyncSession, pred_id: int) -> None:
        await db.execute(
            update(Prediction)
            .where(Prediction.id == pred_id)
            .values(status=RowStatus.failed)
        )

    @staticmethod
    async def get_latest_created_at_all_users(db: AsyncSession) -> Optional[datetime]:
        """
        Return datetime of the latest prediction created across ACTIVE users,
        or None if no predictions exist.
        """
        stmt = select(func.max(Prediction.created_at)).where(
            Prediction.user.has(is_active=True)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_predictions(db: AsyncSession, user_id: int) -> list[Prediction]:
        result = await db.execute(
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at)
        )
        return result.scalars().all()

    @staticmethod
    async def get_all_users_predictions(db: AsyncSession) -> list[Prediction]:
        stmt = select(Prediction).where(Prediction.user.has(is_active=True)).order_by(Prediction.created_at)
        return (await db.execute(stmt)).scalars().all()
