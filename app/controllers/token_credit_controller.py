from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.orm_models.users import User
from app.models.pydantic_models.token_credit import BuyTokensRequest, BuyTokensResponse, TokenCreditHistoryResponse
from app.services.token_credit_service import TokenCreditService as TCservice
from app.services.auth_service import AuthService
from app.utils.rate_limit import rate_limited
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
        user: User = Depends(AuthService.validate_user)
):
    """
    Add tokens to the authenticated user using an idempotency key.

    Args:
        buy_request: Token purchase details (credit_card, amount, idempotency_key).
        db: Async DB session.
        user: Authenticated user.

    Returns:
        Human-readable confirmation message with the new balance.

    Raises:
        BalanceMustBeZeroException if balance not zero by policy.
        PurchaseInProgressException (depending on your mapping).
        RequestValidationError if body invalid (via handler).
    """

    return await TCservice.buy_tokens(db, user, buy_request.amount, buy_request.idempotency_key)


@router.get("/token_history", status_code=status.HTTP_200_OK, response_model=list[TokenCreditHistoryResponse])
@rate_limited("token_history", **config.RATE_LIMITS["token_history"])
async def get_user_token_history(
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
    return await TCservice.get_user_token_history(db, user)