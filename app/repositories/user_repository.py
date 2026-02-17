from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.exceptions.user import NotEnoughTokensException
from app.exceptions.token_credit import TokenBalanceCapExceededException
from app.models.orm_models.users import User
from app.config import config


class UserRepository:
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """Return active user by id, or None."""
        q = (
            select(User).
            where(User.id == user_id, User.is_active == True)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """Return active user by username, or None."""
        q = (
            select(User).
            where(User.username == username, User.is_active == True)
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_tokens_by_id(db: AsyncSession, user_id: int) -> int:
        q = (
            select(User.tokens).
            where(User.id == user_id)
        )
        return (await db.execute(q)).scalar_one()

    @staticmethod
    async def create_user(db: AsyncSession, user: User) -> User:
        db.add(user)
        await db.flush()
        return user

    @staticmethod
    async def add_tokens(db: AsyncSession, user_id: int, amount: int) -> int | None:
        """
        Add tokens if current tokens == 0 (policy). Returns new balance or None if not applied.
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
        Deduct tokens atomically if balance >= cost. Returns remaining or raises.
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
        Soft delete (set is_active = False). Returns True if changed, False if already inactive.
        """
        q = await db.execute(
            update(User)
            .where(User.id == user_id, User.is_active == True)
            .values(is_active=False)
        )
        return q.rowcount == 1


