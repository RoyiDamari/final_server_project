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
    return await AuthService.revoke_refresh_token(db, user, body.refresh_token)

