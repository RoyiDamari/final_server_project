import os
from redis.asyncio.client import Redis
from contextlib import suppress
from fastapi import APIRouter, UploadFile, File, Form, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth_service import AuthService
from app.services.train_model_service import TrainModelService
from app.models.pydantic_models.train_model import TrainedModelResponse
from app.models.pydantic_models.general import MetadataResponse, ActionResponse
from app.models.orm_models.users import User
from app.models.enums import ActionType
from app.utils.rate_limit import rate_limited
from app.utils.files import save_upload_to_temp_csv
from app.utils.parsing import parse_json_list_strict, parse_json_object_strict
from app.utils.redis import get_redis
from app.config import config
from app.database import get_db


router = APIRouter(
    prefix="/train_model",
    tags=["train_model"],
    responses={401: {"user": "Not authorized"}}
)


@router.post("/train", status_code=status.HTTP_200_OK, response_model=ActionResponse[TrainedModelResponse])
@rate_limited("train", **config.RATE_LIMITS["train"])
async def train_model(
        model_type: str = Form(...),
        features: str = Form(...),
        label: str = Form(...),
        model_params: str = Form('{}'),
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user),
):
    """
    Train a model from an uploaded CSV and provided configuration.

    Args:
        model_type: One of the supported model strategies (e.g., "linear", "logistic", "random_forest").
        features: JSON-encoded list of feature column names.
        label: Name of the label/target column in the CSV.
        model_params: JSON-encoded dict of model hyperparameters (strategy-specific).
        file: The uploaded CSV file.
        db: Async SQLAlchemy session (injected).
        redis: cache instance.
        user: Authenticated user (injected).

    Returns:
        TrainedModelResponse: The persisted model metadata.

    Notes:
        - This endpoint is rate-limited.
        - The CSV is written to a temp file on disk, then removed after training completes (success or error).
    """

    features_list = parse_json_list_strict(features)
    model_params_dict = parse_json_object_strict(model_params)

    tmp_filename = save_upload_to_temp_csv(file, suffix=".csv")
    try:
        result: dict = await TrainModelService.train_model(
            db=db,
            redis=redis,
            user=user,
            file=tmp_filename,
            model_type=model_type,
            features=features_list,
            label=label,
            model_params=model_params_dict,
            action=ActionType.TRAINING
        )

        return result
    finally:
        with suppress(Exception):
            os.remove(tmp_filename)


@router.get("/user_models", status_code=status.HTTP_200_OK, response_model=list[TrainedModelResponse])
@rate_limited("user_models", **config.RATE_LIMITS["user_models"])
async def get_user_models(
        user: User = Depends(AuthService.validate_user),
        db: AsyncSession = Depends(get_db),
):
    """
    List the authenticated user's trained models.

    Args:
        user: Authenticated user (injected).
        db: Async SQLAlchemy session (injected).

    Returns:
        List[TrainedModel]: All models owned by the user.

    Notes:
        - This endpoint is rate-limited.
        - Charges a metadata token before returning results.
    """

    return await TrainModelService.get_user_models(db, user)



@router.get("/all_users_models", status_code=status.HTTP_200_OK, response_model=MetadataResponse[TrainedModelResponse])
@rate_limited("all_users_models", **config.RATE_LIMITS["all_users_models"])
async def get_all_users_models(
        user: User = Depends(AuthService.validate_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    return await TrainModelService.get_all_users_models(db, redis, user, ActionType.METADATA)



@router.get("/user_models_internal", status_code=status.HTTP_200_OK, response_model=list[TrainedModelResponse])
async def get_user_models_internal(
    user: User = Depends(AuthService.validate_user),
    db: AsyncSession = Depends(get_db),
):
    """
    INTERNAL endpoint.

    Used by frontend UX flows (prediction form, feature selection, etc.)
    - No rate limit
    - No token deduction
    - Auth required
    """
    return await TrainModelService.get_user_models_internal(db, user)
