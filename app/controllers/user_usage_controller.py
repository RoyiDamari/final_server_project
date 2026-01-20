from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio.client import Redis
from app.services.auth_service import AuthService
from app.services.user_usage_service import UserUsageService as UUServ
from app.models.pydantic_models.user_usage import (ModelTypeDistributionResponse, TypeSplitResponse,
                                                   GroupedLabelDistributionResponse, GroupedMetricDistributionResponse)
from app.models.pydantic_models.general import ActionResponse, MetadataResponse
from app.models.orm_models import User
from app.models.enums import ActionType
from app.database import get_db
from app.utils.redis import get_redis
from app.utils.rate_limit import rate_limited
from app.config import config


router = APIRouter(
    prefix="/usage",
    tags=["usage"],
    responses={401: {"detail": "Not authorized"}},
)


@router.get("/model_type_distribution",
            status_code=status.HTTP_200_OK,
            response_model=MetadataResponse[ModelTypeDistributionResponse])
@rate_limited("model_type_distribution", **config.RATE_LIMITS["model_type_distribution"])
async def get_model_type_distribution(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user)
):
    return await UUServ.get_model_type_distribution(db, redis, user, ActionType.METADATA)


@router.get("/type_split",
            status_code=status.HTTP_200_OK,
            response_model=MetadataResponse[TypeSplitResponse])
@rate_limited("type_split", **config.RATE_LIMITS["type_split"])
async def get_regression_vs_classification_split(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user)
):
    return await UUServ.get_regression_vs_classification_split(db, redis, user, ActionType.METADATA)


@router.post("/label_distribution",
             status_code=status.HTTP_200_OK,
             response_model=ActionResponse[GroupedLabelDistributionResponse])
@rate_limited("label_distribution", **config.RATE_LIMITS["label_distribution"])
async def get_label_distribution(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user)
):
    return await UUServ.get_label_distribution(db, redis, user, ActionType.METADATA)


@router.post("/metric_distribution",
             status_code=status.HTTP_200_OK,
             response_model=ActionResponse[GroupedMetricDistributionResponse])
@rate_limited("metric_distribution", **config.RATE_LIMITS["metric_distribution"])
async def get_metric_distribution(
    user: User = Depends(AuthService.validate_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    return await UUServ.get_metric_distribution(db, redis, user, ActionType.METADATA)
