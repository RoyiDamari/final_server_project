from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.orm_models.trained_models import TrainedModel
from app.models.orm_models.predictions import Prediction
from app.models.enums import RowStatus
from app.maintenance._helpers import (
    finish_publish_or_fail,
    fail_pending_and_clean_tmp,
)


async def reconcile_trained_models_on_startup(db: AsyncSession) -> None:
    applied = await db.execute(
        select(TrainedModel.id, TrainedModel.model_path)
        .where(TrainedModel.status == RowStatus.applied)
    )
    for tm_id, path in applied.all():
        await finish_publish_or_fail(db, tm_id, path)

    pendings = await db.execute(
        select(TrainedModel.id, TrainedModel.model_path)
        .where(TrainedModel.status == RowStatus.pending)
    )
    for tm_id, path in pendings.all():
        await fail_pending_and_clean_tmp(db, tm_id, path)


async def reconcile_predictions_on_startup(db: AsyncSession) -> None:
    await db.execute(
        update(Prediction)
        .where(Prediction.status == RowStatus.pending)
        .values(status=RowStatus.failed)
    )