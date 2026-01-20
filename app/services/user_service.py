from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio.client import Redis
from datetime import datetime, timezone
from app.repositories.user_repository import UserRepository as UserRepo
from app.repositories.auth_repository import AuthRepository as ARepo
from app.models.pydantic_models.user import RegisterUserRequest, RegisterUserResponse, DeleteUserResponse
from app.models.orm_models.users import User
from app.models.enums import ActionType
from app.exceptions.user import (UserAlreadyDeletedException, UserHasRemainingTokensException,
                                 DeleteUserConfirmationException)
from app.utils.password_hashing import get_password_hash, verify_password
from app.utils.cache_invalidation import invalidate_global_predictions_cache, invalidate_global_models_cache
from app.core.logs import log_action


class UserService:
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """Fetch a single active user by id."""
        return await UserRepo.get_user_by_id(db, user_id)

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """Fetch a single active user by username."""
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
            The persisted ORM User (id populated).

        Raises:
            UsernameTakenException: Username already exists (unique violation).
            EmailTakenException: Email already exists (unique violation).
            IntegrityError: Re-raised for unexpected constraint issues.
        """
        user = User(
            first_name=req.first_name,
            last_name=req.last_name,
            username=req.username,
            email=req.email,
            hashed_password=get_password_hash(req.password),
            is_active=True,
        )

        await UserRepo.create_user(db, user)

        log_action(
            "user_has_been_registered",
            user_id=user.id,
            username=user.username,
        )
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

        Also deletes all refresh tokens for this user and invalidates
        related caches (models, predictions, tokens) using Version-D rules.

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
            UserCredentialsException: If username or password confirmation fails.
            UserAlreadyDeletedException: If user is already inactive.
        """

        if user.username != confirm_username or not verify_password(confirm_password, user.hashed_password):
            raise DeleteUserConfirmationException()

        if user.tokens > 0 and not confirm_delete_with_balance:
            raise UserHasRemainingTokensException(
                detail=f"User has {user.tokens} remaining tokens"
            )

        deleted = await UserRepo.delete_user(db, user.id)
        if not deleted:
            raise UserAlreadyDeletedException()

        await ARepo.revoke_all_session_by_user(db, user.id)

        ts = datetime.now(timezone.utc).isoformat()
        await invalidate_global_models_cache(redis, ts)
        await invalidate_global_predictions_cache(redis, ts)

        log_action(
            "user_has_been_deleted_his_account",
            user_id=user.id,
            username=user.username,
            balance_after=user.tokens,
        )

        return DeleteUserResponse(message="User deleted successfully")

    @staticmethod
    async def get_all_users_tokens(
            db: AsyncSession,
            user: User,
            action: ActionType,
    ) -> dict:
        """
        Return all users' active token balances.

        Charging rules:
        - Fetch FIRST
        - If no rows → no charge
        - If rows exist → charge metadata token
        """

        all_users_tokens = await UserRepo.get_all_users_tokens(db)

        if not all_users_tokens:
            return {"data": [], "charged": False, "balance": user.tokens}

        balance = await UserRepo.update_tokens(db, user.id, action.cost)

        log_action(
            event="user_viewed_all_users_tokens",
            user_id=user.id,
            username=user.username,
            action=action,
            charged=action.cost,
            balance_after=balance,
        )

        return {"data": all_users_tokens, "charged": True, "balance": balance}

