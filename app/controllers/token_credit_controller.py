from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio.client import Redis
from app.database import get_db
from app.models.orm_models.users import User
from app.models.pydantic_models.token_credit import BuyTokensRequest, BuyTokensResponse, TokenCreditResponse
from app.models.pydantic_models.general import MetadataResponse
from app.models.enums import ActionType
from app.services.token_credit_service import TokenCreditService as TCservice
from app.services.auth_service import AuthService
from app.utils.rate_limit import rate_limited
from app.utils.redis import get_redis
from app.config import config

router = APIRouter(
    prefix="/token_credit",
    tags=["token_credit"],
)


@router.post("/buy_tokens", status_code=status.HTTP_200_OK, response_model=BuyTokensResponse)
@rate_limited("buy_tokens", **config.RATE_LIMITS["buy_tokens"])
async def buy_tokens(
        buy_request: BuyTokensRequest,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user)
):
    """
    Add tokens to the authenticated user using an idempotency key.

    Args:
        buy_request: Token purchase details (credit_card, amount, idempotency_key).
        db: Async DB session.
        redis: Redis client used for cache invalidation/version bump after a successful purchase.
        user: Authenticated user.

    Returns:
        BuyTokensResponse: Human-readable confirmation message with the new balance.

    Raises:
        BalanceMustBeZeroException: if balance not zero by policy.
        PurchaseInProgressException: if another purchase attempt is pending for this key.
        RequestValidationError: if body invalid (via handler).
    """

    return await TCservice.buy_tokens(db, user, redis, buy_request.amount, buy_request.idempotency_key)


@router.get("/user_tokens", status_code=status.HTTP_200_OK, response_model=list[TokenCreditResponse])
@rate_limited("user_tokens", **config.RATE_LIMITS["user_tokens"])
async def get_user_tokens(
        db: AsyncSession = Depends(get_db),
        user: User = Depends(AuthService.validate_user)
):
    """
    Charge a metadata token and return the caller's current balance.

    Args:
        db: Async DB session.
        user: Authenticated user.

    Returns:
        UserTokensResponse: {"username", "tokens"}.

    Raises:
        NotEnoughTokensException if insufficient balance.
    """
    return await TCservice.get_user_tokens(db, user)


@router.get("/all_users_tokens", status_code=status.HTTP_200_OK, response_model=MetadataResponse[TokenCreditResponse])
@rate_limited("all_users_tokens", **config.RATE_LIMITS["all_users_tokens"])
async def get_all_users_tokens(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user),
):
    """
    Charge a metadata token and return balances of all active users.

    Args:
        db: Async DB session.
        redis: Redis client used for caching/versioning of metadata responses.
        user: Authenticated user.

    Returns:
        MetadataResponse[TokenCreditResponse]: data + charged + balance.
    """
    return await TCservice.get_all_users_tokens(db, redis, user, ActionType.METADATA)
