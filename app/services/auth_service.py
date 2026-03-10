from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError, ExpiredSignatureError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.pydantic_models.auth import LoginResponse, RefreshResponse, LogoutResponse
from app.models.orm_models import User
from app.repositories.auth_repository import AuthRepository as ARepo
from app.repositories.user_repository import UserRepository as URepo
from app.utils.security_utils import generate_id, hash_token
from app.utils.password_hashing import verify_password
from app.core.logs import log_action, errors
from app.config import config
from app.exceptions.auth import (
    TokenGenerationException, ExpiredTokenException, UserCredentialsException,
    InvalidTokenException, ReusedTokenException
)

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/auth/login")


class AuthService:
    """
    Service layer for authentication.

    Design goals:
      - Keep DB transactions short.
      - Use SELECT ... FOR UPDATE for refresh token rotation safety.
      - Retry refresh token generation on rare unique constraint collisions.
    """

    @staticmethod
    async def issue_tokens(
            db: AsyncSession,
            username: str,
            password: str,
            ip_address: str,
            user_agent: str,
    ) -> LoginResponse:
        """
        Authenticate a user and issue a new access token + refresh token.

        Flow:
          1) Validate username/password against stored hash.
          2) Create a new refresh-token session row (new session_id).
          3) Create a JWT access token with expiry.
          4) Return tokens and current balance.

        Args:
            db: Async SQLAlchemy session.
            username: Username (normalized upstream or by caller).
            password: Plain password to verify.
            ip_address: Client IP for audit.
            user_agent: Client user-agent for audit.

        Returns:
            LoginResponse: Access token, refresh token, expiry timestamp, and balance.

        Raises:
            UserCredentialsException: If username does not exist, has no password, or password is invalid.
            TokenGenerationException: If refresh token generation collides too many times (unique constraint collisions).
        """

        async with db.begin():
            user = await URepo.get_user_by_username(db, username)
            if user is None or not user.hashed_password:
                raise UserCredentialsException()
            if not verify_password(password, user.hashed_password):
                raise UserCredentialsException()

            raw_refresh = await AuthService._create_refresh_session_with_retries(
                db=db,
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
            )

        access_token, expires_at = AuthService._create_access_token(user.username, user.id)

        try:
            log_action(
                event="user_has_been_login",
                user_id=user.id,
                username=user.username
            )
        except Exception as e:
            errors.exception("log_action failed in issue_tokens: %r", e)

        return LoginResponse(
            message=f"Login successful. wellcome {user.username}",
            access_token=access_token,
            refresh_token=raw_refresh,
            expires_at=expires_at,
            balance=user.tokens,
        )

    @staticmethod
    async def rotate_refresh_token(
            db: AsyncSession,
            refresh_token: str,
    ) -> RefreshResponse:
        """
        Rotate a refresh token and issue a new access token (race-safe).

        Concurrency:
          - Locks the AuthSession row using SELECT ... FOR UPDATE.
          - Ensures only one refresh rotation can succeed at a time for the same session.

        Validation:
          - Session must exist and user must be active.
          - Session must not be revoked.
          - Token must not be expired or absolute-expired.
          - Reuse detection revokes the session.

        Args:
            db: Async SQLAlchemy session.
            refresh_token: Raw refresh token presented by the client.

        Returns:
            RefreshResponse: New access token + new refresh token + expiry timestamp.

        Raises:
            InvalidTokenException: If token hash is not found or user is inactive.
            ExpiredTokenException: If refresh token is expired or absolute-expired (session revoked on absolute expiry).
            ReusedTokenException: If token reuse is detected (session revoked).
            TokenGenerationException: If new token hash collides too many times.
        """

        presented_hash = hash_token(refresh_token)

        async with db.begin():
            row = await ARepo.get_refresh_token(db, presented_hash)
            if not row or not row.user or not row.user.is_active:
                raise InvalidTokenException()

            if row.revoked:
                raise ReusedTokenException(log_detail="Reused token from revoked session")

            now = datetime.now(timezone.utc)

            if row.expires_at < now:
                raise ExpiredTokenException()

            if row.absolute_expires_at < now:
                await ARepo.revoke_by_session(db, row.session_id)
                raise ExpiredTokenException()

            if row.last_token_hash == presented_hash:
                await ARepo.revoke_by_session(db, row.session_id)
                raise ReusedTokenException(log_detail="Refresh token reuse detected — session revoked")

            raw_new = await AuthService._rotate_refresh_token_with_retries(
                db=db,
                session_id=row.session_id,
                last_token_hash=presented_hash,
            )

        access_token, expires_at = AuthService._create_access_token(row.user.username, row.user.id)

        return RefreshResponse(
            access_token=access_token,
            refresh_token=raw_new,
            expires_at=expires_at,
        )

    @staticmethod
    async def revoke_refresh_token(
            db: AsyncSession,
            user: User,
            refresh_token: str,
    ) -> LogoutResponse:
        """
        Revoke the refresh session associated with the provided refresh token.

        Notes:
          - Best-effort: if token is not found, behaves like already logged out.
          - Revoke is idempotent (revoked=True).

        Args:
            db: Async SQLAlchemy session.
            user: Authenticated user performing logout.
            refresh_token: Raw refresh token to revoke.

        Returns:
            LogoutResponse: Confirmation message.

        Raises:
            Exception: Not raised intentionally; unexpected DB errors propagate.
        """

        hashed = hash_token(refresh_token)

        async with db.begin():
            row = await ARepo.get_refresh_token(db, hashed)
            if row:
                await ARepo.revoke_by_session(db, row.session_id)

        try:
            log_action(
                event="user_has_been_logout",
                user_id=user.id,
                username=user.username
            )
        except Exception as e:
            errors.exception("log_action failed in revoke_refresh_token: %r", e)

        return LogoutResponse(message="Logout successful.")

    @staticmethod
    def _create_access_token(username: str, user_id: int) -> tuple[str, int]:
        """
        Create a signed JWT access token with an expiry claim.

        Args:
            username: Subject username to embed in JWT ("sub").
            user_id: User id to embed in JWT ("uid").

        Returns:
            tuple[str, int]:
                - token: Encoded JWT string.
                - expires_at: Expiry timestamp as UNIX seconds (int).
        """

        exp = datetime.now(timezone.utc) + timedelta(minutes=config.TOKEN_EXPIRY_TIME)
        payload = {
            "sub": username,
            "uid": user_id,
            "exp": exp,
            "iat": datetime.now(timezone.utc),
        }
        token = jwt.encode(payload, config.SECRET_KEY, algorithm=config.ALGORITHM)
        return token, int(exp.timestamp())

    @staticmethod
    async def _create_refresh_session_with_retries(
            db: AsyncSession,
            user_id: int,
            ip_address: str,
            user_agent: str,
    ) -> str:
        """
        Create a new refresh session with retry on unique collisions.

        Uses short transactions:
          - each attempt runs inside `async with db.begin(): ...`
          - on IntegrityError the tx rolls back and we retry

        Returns:
            raw refresh token (not hashed)

        Raises:
            TokenGenerationException: after MAX retries.
        """

        for _ in range(config.MAX_TOKEN_GENERATION_RETRIES):
            raw = generate_id()
            hashed = hash_token(raw)
            expiry = datetime.now(timezone.utc) + timedelta(hours=1)
            abs_exp = datetime.now(timezone.utc) + timedelta(hours=24)

            try:
                await ARepo.insert_new_refresh_token(
                    db=db,
                    session_id=generate_id(),
                    user_id=user_id,
                    refresh_hash=hashed,
                    expires_at=expiry,
                    absolute_expires_at=abs_exp,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                return raw
            except IntegrityError:
                continue

        raise TokenGenerationException()

    @staticmethod
    async def _rotate_refresh_token_with_retries(
            db: AsyncSession,
            session_id: str,

            last_token_hash: str,
    ) -> str:
        """
        Rotate refresh token hash for an existing session with retry on collisions.

        Must be called *inside an existing transaction* that already locked the session row.

        Returns:
            raw refresh token (not hashed)

        Raises:
            TokenGenerationException: after MAX retries.
        """

        for _ in range(config.MAX_TOKEN_GENERATION_RETRIES):
            raw = generate_id()
            hashed = hash_token(raw)
            expiry = datetime.now(timezone.utc) + timedelta(hours=1)

            try:
                await ARepo.rotate_refresh_token(
                    db=db,
                    session_id=session_id,
                    new_token_hash=hashed,
                    last_token_hash=last_token_hash,
                    new_expiry=expiry,
                )
                return raw
            except IntegrityError:
                continue

        raise TokenGenerationException()

    @staticmethod
    async def validate_user(token: str = Depends(oauth2_bearer), db: AsyncSession = Depends(get_db)) -> User:
        """
        Dependency wrapper.
        """

        return await AuthService.validate_user_check(token, db)

    @staticmethod
    async def validate_user_check(token: str, db: AsyncSession) -> User:
        """
        Validate a JWT access token and return the corresponding active user.

        Steps:
          1) Decode JWT and extract uid claim.
          2) Fetch the user from DB and ensure is_active=True.

        Args:
            token: JWT access token string (Bearer payload without prefix).
            db: Async SQLAlchemy session.

        Returns:
            User: Active user ORM instance.

        Raises:
            ExpiredTokenException: If JWT exp is in the past.
            InvalidTokenException: If JWT is invalid/malformed or user does not exist/is inactive.
        """

        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
            uid = payload.get("uid")
            if not uid:
                raise InvalidTokenException()

            async with db.begin():
                user = await URepo.get_user_by_id(db, uid)

            if not user:
                raise InvalidTokenException()

            return user

        except ExpiredSignatureError:
            raise ExpiredTokenException()
        except JWTError:
            raise InvalidTokenException()
