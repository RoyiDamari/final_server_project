from redis.asyncio.client import Redis
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth_service import AuthService
from app.services.prediction_service import PredictionService
from app.models.pydantic_models.prediction import PredictionRequest, PredictionResponse
from app.models.pydantic_models.general import MetadataResponse, ActionResponse
from app.utils.rate_limit import rate_limited
from app.models.orm_models import User
from app.models.enums import ActionType
from app.config import config
from app.database import get_db
from app.utils.redis import get_redis


router = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
    responses={401: {"user": "Not authorized"}},
)


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=ActionResponse[PredictionResponse])
@rate_limited("predict", **config.RATE_LIMITS["predict"])
async def predict(
        predict_req: PredictionRequest,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user),
):
    """
    Run a prediction on a user's trained model and log it atomically with the token charge.
    """
    return await PredictionService.predict(
        db=db,
        redis=redis,
        user=user,
        request=predict_req,
        action=ActionType.PREDICTION,
    )


@router.get("/user_predictions", status_code=status.HTTP_200_OK, response_model=list[PredictionResponse])
@rate_limited("user_predictions", **config.RATE_LIMITS["user_predictions"])
async def get_user_predictions(
        db: AsyncSession = Depends(get_db),
        user: User = Depends(AuthService.validate_user),
):
    """
    Charge a metadata token and list the caller’s predictions.
    """
    return await PredictionService.get_user_predictions(db, user)


@router.get("/all_users_predictions", status_code=status.HTTP_200_OK, response_model=MetadataResponse[PredictionResponse])
@rate_limited("all_users_predictions", **config.RATE_LIMITS["all_users_predictions"])
async def get_all_users_predictions(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user),
):
    """
    Charge a metadata token and list all users’ predictions (admin-like view).
    """
    return await PredictionService.get_all_users_predictions(db, redis, user, ActionType.METADATA)
