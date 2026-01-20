from fastapi import APIRouter, Depends, status, Request
from redis.asyncio.client import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from app.services.user_service import UserService
from app.services.auth_service import AuthService
from app.exceptions.user import UsernameTakenException, EmailTakenException
from app.models.pydantic_models.user import (RegisterUserRequest, RegisterUserResponse, DeleteUserResponse,
                                             UserTokensResponse, DeleteUserRequest)
from app.models.pydantic_models.general import MetadataResponse
from app.models.orm_models import User
from app.models.enums import ActionType
from app.utils.rate_limit import rate_limited
from app.utils.redis import get_redis
from app.database import get_db
from app.config import config


router = APIRouter(
    prefix="/user",
    tags=["user"],
    responses={401: {"user": "Not authorized"}}
)


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=RegisterUserResponse)
@rate_limited("register", **config.RATE_LIMITS["register"])
async def register_user(
        reg_request: RegisterUserRequest,
        _request: Request,
        db: AsyncSession = Depends(get_db)
):
    """
    Register a new user and return the public view.

    Args:
        reg_request: Validated user details.
        _request: Request object (used by the rate limiter).
        db: Async DB session.

    Returns:
        RegisterUserResponse: Serialized view of the created user.

    Raises:
        UsernameTakenException, EmailTakenException (via handlers → 409).
    """
    try:
        return await UserService.register_user(db, reg_request)
    except IntegrityError as e:
        orig = getattr(e, "orig", None)
        msg = str(orig).lower() if orig else ""

        if "username" in msg:
            raise UsernameTakenException()
        if "email" in msg:
            raise EmailTakenException()
        raise


@router.delete("/delete", status_code=status.HTTP_200_OK, response_model=DeleteUserResponse)
@rate_limited("delete", **config.RATE_LIMITS["delete"])
async def delete_user(
        del_request: DeleteUserRequest,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        user: User = Depends(AuthService.validate_user)
):
    """
    Soft-delete the authenticated user (set is_active = False).

    Args:
        del_request: Username/password confirmation.
        db: Async DB session.
        redis: cache
        user: Authenticated user.

    Returns:
        DeleteUserResponse: Confirmation message.

    Raises:
        UserCredentialsException (if username/password confirmation fails).
        UserAlreadyDeletedException (if already inactive).
    """
    return await UserService.delete_user(
        db,
        redis,
        user,
        del_request.username,
        del_request.password,
        del_request.confirm_delete_with_balance
    )


@router.get("/all_users_tokens", status_code=status.HTTP_200_OK, response_model=MetadataResponse[UserTokensResponse])
@rate_limited("all_users_tokens", **config.RATE_LIMITS["all_users_tokens"])
async def get_all_users_tokens(
        db: AsyncSession = Depends(get_db),
        user: User = Depends(AuthService.validate_user)
):
    """
    Charge a metadata token and return balances of all active users.

    Args:
        db: Async DB session.
        user: Authenticated user.

    Returns:
        List[UserTokensResponse]: [{"username", "tokens"}, ...].
    """
    return await UserService.get_all_users_tokens(db, user, ActionType.METADATA)

