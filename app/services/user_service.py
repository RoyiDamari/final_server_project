import asyncio
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio.client import Redis
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone
from contextlib import suppress
from app.repositories.user_repository import UserRepository as UserRepo
from app.repositories.auth_repository import AuthRepository as ARepo
from app.models.pydantic_models.user import RegisterUserRequest, RegisterUserResponse, DeleteUserResponse
from app.models.orm_models.users import User
from app.exceptions.user import (UsernameTakenException, EmailTakenException, UserAlreadyDeletedException,
                                 UserHasRemainingTokensException, DeleteUserConfirmationException)
from app.utils.password_hashing import get_password_hash, verify_password
from app.utils.cache_invalidation import (invalidate_global_predictions_cache, invalidate_global_models_cache,
                                          invalidate_global_token_credits_cache, invalidate_global_user_usage_cache)
from app.core.logs import log_action, errors


class UserService:
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """
        Fetch one active user by id.

        Args:
            db: Async SQLAlchemy session.
            user_id: User primary key.

        Returns:
            User | None: Active user row if found, else None.
        """

        return await UserRepo.get_user_by_id(db, user_id)

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """
        Fetch one active user by username.

        Args:
            db: Async SQLAlchemy session.
            username: Username (case-insensitive via CITEXT).

        Returns:
            User | None: Active user row if found, else None.
        """

        return await UserRepo.get_user_by_username(db, username)

    @staticmethod
    async def register_user(db: AsyncSession, req: RegisterUserRequest) -> RegisterUserResponse:
        """
        Create and persist a new user.

        Transactions & logging:
            - Inserts inside a DB transaction.
            - After commit, logs event: "user_registered".

        Args:
            db: SQLAlchemy async session.
            req: Validated registration payload.

        Returns:
            RegisterUserResponse: Success message payload.
        """

        user = User(
            first_name=req.first_name,
            last_name=req.last_name,
            username=req.username,
            email=req.email,
            hashed_password=get_password_hash(req.password),
            is_active=True,
        )

        try:
            async with db.begin():
                await UserRepo.create_user(db, user)
        except IntegrityError as e:
            orig = getattr(e, "orig", None)
            msg = str(orig).lower() if orig else ""

            if "username" in msg:
                raise UsernameTakenException()
            if "email" in msg:
                raise EmailTakenException()
            raise

        try:
            log_action(
                "user_has_been_registered",
                user_id=user.id,
                username=user.username,
            )
        except Exception as e:
            errors.exception("log_action failed in register_user: %r", e)

        return RegisterUserResponse(
            message=f"{user.username} has registered successfully",
        )

    @staticmethod
    async def delete_user(
            db: AsyncSession,
            redis: Redis,
            user: User,
            confirm_username: str,
            confirm_password: str,
            confirm_delete_with_balance: bool,
    ) -> DeleteUserResponse:
        """
        Soft-delete (is_active=False) after confirming credentials.

        DB side effects (atomic):
          - Mark user inactive (soft delete)
          - Revoke all refresh-token sessions for that user

        Redis side effects (best-effort):
          - Invalidate global caches that depend on active users/models/preds/tokens
          - Cleanup per-user seen keys (if you keep that Redis-based scheme)

        Args:
            db: Async DB session.
            redis: Redis client for cache invalidation and version bumping.
            user: Authenticated user (must match confirmation fields).
            confirm_username: Must equal user.username.
            confirm_password: Must verify against user.hashed_password.
            confirm_delete_with_balance: Must verify by the user in case his token balance is positive

        Returns:
            DeleteUserResponse with success message.

        Raises:
            DeleteUserConfirmationException: If username/password confirmation fails.
            UserHasRemainingTokensException: If user has tokens and did not confirm deletion with balance.
            UserAlreadyDeletedException: If the user is already inactive (soft-deleted).
        """

        if user.username != confirm_username or not verify_password(confirm_password, user.hashed_password):
            raise DeleteUserConfirmationException()

        if user.tokens > 0 and not confirm_delete_with_balance:
            raise UserHasRemainingTokensException(
                detail=f"User has {user.tokens} remaining tokens"
            )

        async with db.begin():
            deleted = await UserRepo.delete_user(db, user.id)
            if not deleted:
                raise UserAlreadyDeletedException()

            await ARepo.revoke_all_sessions_by_user(db, user.id)

        ts = datetime.now(timezone.utc).isoformat()
        with suppress(RedisError, asyncio.TimeoutError):
            await invalidate_global_models_cache(redis, ts)
            await invalidate_global_predictions_cache(redis, ts)
            await invalidate_global_token_credits_cache(redis, ts)
            await invalidate_global_user_usage_cache(redis, ts)

        try:
            log_action(
                "user_has_been_deleted_his_account",
                user_id=user.id,
                username=user.username,
                balance_after=user.tokens,
            )
        except Exception as e:
            errors.exception("log_action failed in delete_user: %r", e)

        return DeleteUserResponse(message="User deleted successfully")
