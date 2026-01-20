from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError, ExpiredSignatureError
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from fastapi import Depends
from app.database import get_db
from app.models.pydantic_models.auth import LoginResponse, RefreshResponse, LogoutResponse
from app.models.orm_models import User
from app.repositories.auth_repository import AuthRepository as ARepo
from app.repositories.user_repository import UserRepository as URepo
from app.utils.security_utils import generate_id, hash_token
from app.services.user_service import UserService
from app.utils.password_hashing import verify_password
from app.core.logs import log_action
from app.config import config
from app.exceptions.auth import (
    TokenGenerationException, ExpiredTokenException, UserCredentialsException,
    InvalidTokenException, ReusedTokenException
)

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/auth/login")


class AuthService:
    @staticmethod
    async def _try_generate_unique_refresh_token_with_retries(
            db: AsyncSession,
            user_id: int,
            ip_address: str,
            user_agent: str,
            session_id: str | None = None,
            rotate: bool = False,
            last_token_hash: str | None = None,
    ) -> str:
        for _ in range(config.MAX_TOKEN_GENERATION_RETRIES):
            try:
                raw_refresh = generate_id()
                hashed_refresh = hash_token(raw_refresh)
                expiry = datetime.now(timezone.utc) + timedelta(hours=1)

                if rotate:
                    await ARepo.rotate_refresh_token(
                        db=db,
                        session_id=session_id,
                        new_token_hash=hashed_refresh,
                        last_token_hash=last_token_hash,
                        new_expiry=expiry,
                    )
                else:
                    await ARepo.insert_new_refresh_token(
                        db=db,
                        session_id=generate_id(),
                        user_id=user_id,
                        refresh_hash=hashed_refresh,
                        expires_at=expiry,
                        absolute_expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                        ip_address=ip_address,
                        user_agent=user_agent,
                    )

                return raw_refresh

            except IntegrityError:
                continue

        raise TokenGenerationException()

    @staticmethod
    async def issue_tokens(
            db: AsyncSession,
            username: str,
            password: str,
            ip_address: str,
            user_agent: str,
    ) -> LoginResponse:

        user = await URepo.get_user_by_username(db, username)
        if user is None:
            raise UserCredentialsException()

        if not user.hashed_password:
            raise UserCredentialsException()

        if not verify_password(password, user.hashed_password):
            raise UserCredentialsException()

        refresh_token = await AuthService._try_generate_unique_refresh_token_with_retries(
            db=db,
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        access_token, expires_at = AuthService._create_access_token(user.username, user.id)

        log_action(
            event="user_has_been_login",
            user_id=user.id,
            username=user.username,
        )

        return LoginResponse(
            message=f"Login successful. wellcome {user.username}",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            balance=user.tokens,
        )

    @staticmethod
    async def rotate_refresh_token(
            db: AsyncSession,
            refresh_token: str,
    ) -> RefreshResponse:

        last_refresh_token = hash_token(refresh_token)

        row = await ARepo.get_refresh_token(db, last_refresh_token)
        if not row or not row.user or not row.user.is_active:
            raise InvalidTokenException()

        row_user = row.user

        if row.revoked:
            raise ReusedTokenException(log_detail="Reused token from revoked session")
        if row.expires_at < datetime.now(timezone.utc):
            raise ExpiredTokenException()
        if row.absolute_expires_at < datetime.now(timezone.utc):
            await ARepo.revoke_by_session(db, row.session_id)
            raise ExpiredTokenException()
        if row.last_token_hash == last_refresh_token:
            await ARepo.revoke_by_session(db, row.session_id)
            raise ReusedTokenException(log_detail="Refresh token reuse detected — session revoked")

        refresh_token = await AuthService._try_generate_unique_refresh_token_with_retries(
            db=db,
            user_id=row_user.id,
            ip_address=row.ip_address,
            user_agent=row.user_agent,
            session_id=row.session_id,
            rotate=True,
            last_token_hash=last_refresh_token
        )

        access_token, expires_at = AuthService._create_access_token(row_user.username, row_user.id)

        return RefreshResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )

    @staticmethod
    async def revoke_refresh_token(db: AsyncSession, user: User, refresh_token: str) -> LogoutResponse:
        hashed = hash_token(refresh_token)

        row = await ARepo.get_refresh_token(db, hashed)
        if row:
            await ARepo.revoke_by_session(db, row.session_id)

        log_action(
            event="user_has_been_logout",
            user_id=user.id,
            username=user.username,
        )

        return LogoutResponse(message="Logout successful.")

    @staticmethod
    def _create_access_token(username: str, user_id: int) -> tuple[str, int]:
        exp = datetime.now(timezone.utc) + timedelta(minutes=config.TOKEN_EXPIRY_TIME)
        payload = {
            "sub": username,
            "uid": user_id,
            "exp": exp,
            "iat": datetime.now(timezone.utc)
        }

        token = jwt.encode(payload, config.SECRET_KEY, algorithm=config.ALGORITHM)
        return token, int(exp.timestamp())

    @staticmethod
    async def validate_user(token: str = Depends(oauth2_bearer), db: AsyncSession = Depends(get_db)):
        return await AuthService.validate_user_check(token, db)

    @staticmethod
    async def validate_user_check(token: str, db: AsyncSession):
        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
            uid = payload.get("uid")
            if not uid:
                raise InvalidTokenException()

            user = await UserService.get_user_by_id(db, uid)
            if not user:
                raise InvalidTokenException()
            return user

        except ExpiredSignatureError:
            raise ExpiredTokenException()
        except JWTError:
            raise InvalidTokenException()
