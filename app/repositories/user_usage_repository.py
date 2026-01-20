from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, cast, Float, case, literal_column
from app.models.orm_models.trained_models import TrainedModel


class UserUsageRepository:
    @staticmethod
    async def get_model_type_distribution(db: AsyncSession) -> list[Dict]:
        result = await db.execute(
            select(TrainedModel.model_type, func.count(TrainedModel.id)).where(
                TrainedModel.user.has(is_active=True)
            ).group_by(TrainedModel.model_type)
        )
        return [{"model_type": row[0], "count": row[1]} for row in result.all()]

    @staticmethod
    async def get_regression_vs_classification_split(db: AsyncSession) -> list[Dict]:
        result = await db.execute(
            select(TrainedModel.model_type, func.count(TrainedModel.id)).where(
                TrainedModel.user.has(is_active=True)
            ).group_by(TrainedModel.model_type)
        )
        split = {"Regression": 0, "Classification": 0}
        for model_type, count in result.all():
            if model_type in ["linear", "ridge", "lasso", "random_forest", "svr"]:
                split["Regression"] += count
            else:
                split["Classification"] += count
        return [{"problem_type": k, "count": v} for k, v in split.items()]

    @staticmethod
    async def get_label_distribution(db: AsyncSession) -> dict:
        metric_type = case(
            (TrainedModel.metrics.has_key("accuracy"), literal_column("'classification'")),
            (TrainedModel.metrics.has_key("r2"), literal_column("'regression'")),
            else_=None,
        ).label("metric_type")

        stmt = (
            select(
                TrainedModel.label,
                metric_type,
                func.count(TrainedModel.id).label("count"),
            )
            .where(
                TrainedModel.user.has(is_active=True),
                metric_type.isnot(None),
            )
            .group_by(TrainedModel.label, metric_type)
        )

        rows = (await db.execute(stmt)).all()

        result = {
            "classification": [],
            "regression": [],
        }

        for label, mtype, count in rows:
            result[mtype].append({
                "label": label,
                "count": count,
            })

        return result

    @staticmethod
    async def get_metric_distribution(
            db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Return global metric distributions for BOTH classification and regression
        in a single request (union dashboard).

        No charging logic here — handled by controller/service wrapper.
        """

        # ---------- Classification: accuracy ----------
        acc_bucket = (
                func.floor(
                    cast(TrainedModel.metrics["accuracy"].astext, Float) * 10
                ) / 10.0
        ).label("bucket")

        acc_stmt = (
            select(acc_bucket, func.count().label("count"))
            .where(
                TrainedModel.user.has(is_active=True),
                TrainedModel.metrics["accuracy"].isnot(None),
            )
            .group_by(acc_bucket)
            .order_by(acc_bucket)
        )

        acc_rows = (await db.execute(acc_stmt)).all()
        accuracy_dist = [
            {"bucket": float(r.bucket), "count": r.count}
            for r in acc_rows
        ]

        # ---------- Regression: r2 ----------
        r2_bucket = (
                func.floor(
                    cast(TrainedModel.metrics["r2"].astext, Float) * 10
                ) / 10.0
        ).label("bucket")

        r2_stmt = (
            select(r2_bucket, func.count().label("count"))
            .where(
                TrainedModel.user.has(is_active=True),
                TrainedModel.metrics["r2"].isnot(None),
            )
            .group_by(r2_bucket)
            .order_by(r2_bucket)
        )

        r2_rows = (await db.execute(r2_stmt)).all()
        r2_dist = [
            {"bucket": float(r.bucket), "count": r.count}
            for r in r2_rows
        ]

        return {
            "classification": accuracy_dist,
            "regression": r2_dist,
        }
