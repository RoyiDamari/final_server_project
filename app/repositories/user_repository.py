from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.exceptions.user import NotEnoughTokensException
from app.exceptions.token_credit import TokenBalanceCapExceededException
from app.models.orm_models.users import User
from app.config import config


class UserRepository:
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """
        Fetch a single active user by id.

        Args:
            db: Async SQLAlchemy session.
            user_id: User primary key.

        Returns:
            User | None: Active user row if found, else None.
        """
        q = (
            select(User).
            where(User.id == user_id, User.is_active == True)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """
        Fetch a single active user by username.

        Args:
            db: Async SQLAlchemy session.
            username: Username (CITEXT, case-insensitive).

        Returns:
            User | None: Active user row if found, else None.
        """
        q = (
            select(User).
            where(User.username == username, User.is_active == True)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_tokens_by_id(db: AsyncSession, user_id: int) -> int:
        """
        Fetch current token balance for a user (active or inactive).

        Args:
            db: Async SQLAlchemy session.
            user_id: User primary key.

        Returns:
            int: Current token balance.
        """
        q = (
            select(User.tokens).
            where(User.id == user_id)
        )
        return (await db.execute(q)).scalar_one()

    @staticmethod
    async def create_user(db: AsyncSession, user: User) -> User:
        """
        Add a new User ORM object to the session and flush it.

        Notes:
            - This does NOT commit by itself.
            - Caller should wrap in `async with db.begin():` to commit, and to roll back on error.

        Args:
            db: Async SQLAlchemy session.
            user: User ORM instance (not yet persisted).

        Returns:
            User: The same ORM instance, with `id` populated after flush.
        """
        db.add(user)
        await db.flush()
        return user

    @staticmethod
    async def add_tokens(db: AsyncSession, user_id: int, amount: int) -> int | None:
        """
        Increase user token balance atomically, enforcing MAX_TOKENS cap.

        Args:
            db: Async SQLAlchemy session.
            user_id: User primary key.
            amount: Tokens to add.

        Returns:
            int: New token balance after increment.

        Raises:
            TokenBalanceCapExceededException: If the update did not apply (cap exceeded / inactive / not found).
        """
        q = (
            update(User)
            .where(
                User.id == user_id,
                User.is_active == True,
                (User.tokens + amount) <= config.MAX_TOKENS
            )
            .values(tokens=User.tokens + amount)
            .returning(User.tokens)
        )
        result = (await db.execute(q)).scalar_one_or_none()
        if result is None:
            raise TokenBalanceCapExceededException()
        return result

    @staticmethod
    async def update_tokens(db: AsyncSession, user_id: int, cost: int) -> int:
        """
        Deduct tokens atomically if balance is sufficient.

        Args:
            db: Async SQLAlchemy session.
            user_id: User primary key.
            cost: Tokens to deduct.

        Returns:
            int: Remaining token balance after deduction.

        Raises:
            NotEnoughTokensException: If balance < cost or user not found.
        """
        q = (
            update(User)
            .where(User.id == user_id, User.tokens >= cost)
            .values(tokens=User.tokens - cost)
            .returning(User.tokens)
        )
        result = (await db.execute(q)).fetchone()
        if not result:
            raise NotEnoughTokensException()
        return result[0]

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: int) -> bool:
        """
        Soft-delete a user (set is_active = False) if currently active.

        Args:
            db: Async SQLAlchemy session.
            user_id: User primary key.

        Returns:
            bool:
                - True if the row transitioned from active -> inactive now
                - False if no row was updated (already inactive or not found)
        """
        q = await db.execute(
            update(User)
            .where(User.id == user_id, User.is_active == True)
            .values(is_active=False)
        )
        return q.rowcount == 1
