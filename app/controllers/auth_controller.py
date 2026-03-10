from fastapi import APIRouter, Depends, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth_service import AuthService
from app.models.pydantic_models.auth import (LoginUserRequest, LoginResponse, RefreshResponse,
                                             LogoutRequest, LogoutResponse)
from app.models.orm_models import User
from app.database import get_db
from app.utils.rate_limit import rate_limited
from app.config import config

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", status_code=status.HTTP_200_OK, response_model=LoginResponse)
@rate_limited("login", **config.RATE_LIMITS["login"])
async def login_user(
        req: LoginUserRequest,
        request: Request,
        db: AsyncSession = Depends(get_db)
):
    """
    Authenticate a user and issue an access+refresh token pair.

    Captures client metadata (IP address and User-Agent) for session tracking and
    refresh-token management.

    Args:
        req: LoginUserRequest containing username/password.
        request: FastAPI Request object for extracting IP and User-Agent.
        db: Async SQLAlchemy session dependency.

    Returns:
        LoginResponse with access_token, refresh_token, expires_at, balance, and message.
    """

    ip_address = request.client.host
    user_agent = request.headers.get("user-agent", "")

    return await AuthService.issue_tokens(
        db=db,
        username=req.username,
        password=req.password,
        ip_address=ip_address,
        user_agent=user_agent,
    )


@router.post("/refresh", status_code=status.HTTP_200_OK, response_model=RefreshResponse)
@rate_limited("refresh", **config.RATE_LIMITS["refresh"])
async def rotate_refresh_token(
        request: Request,
        db: AsyncSession = Depends(get_db),
):
    """
    Rotate a refresh token and issue a new access+refresh pair.

    The refresh token is expected in the Authorization header as:
        Authorization: Bearer <refresh_token>

    Args:
        request: FastAPI Request object used to read Authorization header.
        db: Async SQLAlchemy session dependency.

    Returns:
        RefreshResponse with new access_token, refresh_token, and expires_at.
    """

    refresh_token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()

    return await AuthService.rotate_refresh_token(
        db=db,
        refresh_token=refresh_token,
    )


@router.delete("/logout", status_code=status.HTTP_200_OK, response_model=LogoutResponse)
@rate_limited("logout", **config.RATE_LIMITS["logout"])
async def logout_user(
        body: LogoutRequest,
        db: AsyncSession = Depends(get_db),
        user: User = Depends(AuthService.validate_user),
):
    """
    Logout the current session by revoking the provided refresh token.

    Requires a valid access token (user dependency) and a refresh token in the body.

    Args:
        body: LogoutRequest containing the refresh_token to revoke.
        db: Async SQLAlchemy session dependency.
        user: Authenticated user resolved from access token.

    Returns:
        LogoutResponse confirming revocation.
    """

    return await AuthService.revoke_refresh_token(db, user, body.refresh_token)
