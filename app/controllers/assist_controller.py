from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio.client import Redis
from app.models.pydantic_models.assist import AssistExplainRequest, AssistExplainResponse
from app.models.orm_models import User
from app.models.enums import ActionType
from app.services.auth_service import AuthService
from app.services.assist_service import AssistService
from app.utils.rate_limit import rate_limited
from app.database import get_db
from app.utils.redis import get_redis
from app.config import config

router = APIRouter(prefix="/assist", tags=["assist"])


@router.post("/explain", status_code=status.HTTP_201_CREATED, response_model=AssistExplainResponse)
@rate_limited("explain", **config.RATE_LIMITS["explain"])
async def explain_param_route(
        payload: AssistExplainRequest,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user),
):
    """
    Explain a model/parameter or answer a free-text question using the Assist service.

    This endpoint is token-charged (ActionType.ASSIST) and rate-limited.
    It supports two modes via AssistExplainRequest:
        1) Parameter explanation mode (model_type + param_key)
        2) Free-text question mode (context + optional model_type)

    Args:
        payload: AssistExplainRequest containing model_type/param_key/context fields.
        db: Async SQLAlchemy session dependency.
        redis: Redis client dependency.
        user: Authenticated user resolved from access token.

    Returns:
        AssistExplainResponse containing the explanation text and billing metadata.
    """

    return await AssistService.explain_param(
        db, redis, user, ActionType.ASSIST, payload.model_type, payload.param_key, payload.context)
