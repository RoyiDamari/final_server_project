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
    Run a prediction using one of the authenticated user's trained models.

    Behavior:
      - Loads the selected model artifact from disk (must belong to user, status=applied).
      - Creates (or reuses) an idempotent prediction row keyed by fingerprint.
      - Runs prediction computation and, on success, charges tokens and marks the row applied.
      - Invalidates/bump global prediction metadata cache version in Redis after success.

    Args:
        predict_req: PredictionRequest containing model_id and feature_values.
        db: Async SQLAlchemy session (request-scoped).
        redis: Redis client used for cache invalidation/versioning of prediction metadata.
        user: Authenticated user (injected via AuthService.validate_user).

    Returns:
        ActionResponse[PredictionResponse]: The persisted prediction row plus billing metadata
        (charged/balance) according to your ActionResponse schema.

    Raises:
        ModelNotFoundException: If model does not exist / not owned / not applied.
        FeatureMismatchException: If provided feature keys don't match the model's expected features.
        PredictionInProgressException: If a duplicate request is already pending.
        PredictionFailedException: If prediction fails (timeout, artifact issues, unexpected errors).
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
    Return the authenticated user's prediction history.

    Notes:
      - No metadata billing should occur here (based on your service implementation).
      - No Redis caching/versioning is involved in this endpoint.

    Args:
        db: Async SQLAlchemy session (request-scoped).
        user: Authenticated user (injected via AuthService.validate_user).

    Returns:
        list[PredictionResponse]: The user's predictions ordered by created_at (per repository logic).
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
    Return predictions across all active users with per-user-per-version metadata billing.

    Behavior (high level):
      - Uses Redis to cache the global list and store a global version key.
      - Charges the caller at most once per global version (based on Redis "last seen" key).
      - Returns cached list when version matches; refreshes cache when DB version changes.

    Args:
        db: Async SQLAlchemy session (request-scoped).
        redis: Redis client used for caching/versioning and per-user "last seen" tracking.
        user: Authenticated user (injected via AuthService.validate_user).

    Returns:
        MetadataResponse[PredictionResponse]:
            - data: list of all users' predictions (active users only if your repo filters that way)
            - charged: whether the metadata token was charged on this request
            - balance: resulting user token balance
    """
    return await PredictionService.get_all_users_predictions(db, redis, user, ActionType.METADATA)
