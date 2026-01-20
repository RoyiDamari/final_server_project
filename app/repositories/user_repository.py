from typing import Mapping, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.exceptions.user import NotEnoughTokensException
from app.models.orm_models.users import User


class UserRepository:
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """Return active user by id, or None."""
        result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """Return active user by username, or None."""
        result = await db.execute(select(User).where(User.username == username, User.is_active == True))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_tokens_by_id(db, user_id: int) -> int:
        stmt = select(User.tokens).where(User.id == user_id)
        return (await db.execute(stmt)).scalar_one()

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
        row = await db.execute(
            update(User)
            .where(User.id == user_id, User.tokens == 0, User.is_active == True)
            .values(tokens=User.tokens + amount)
            .returning(User.tokens)
        )
        rec = row.fetchone()
        return rec[0] if rec else None

    @staticmethod
    async def update_tokens(db: AsyncSession, user_id: int, cost: int) -> int:
        """
        Deduct tokens atomically if balance >= cost. Returns remaining or raises.
        """
        stmt = (
            update(User)
            .where(User.id == user_id, User.tokens >= cost)
            .values(tokens=User.tokens - cost)
            .returning(User.tokens)
        )
        row = (await db.execute(stmt)).fetchone()
        if not row:
            raise NotEnoughTokensException()
        return row[0]

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: int) -> bool:
        """
        Soft delete (set is_active = False). Returns True if changed, False if already inactive.
        """
        res = await db.execute(
            update(User)
            .where(User.id == user_id, User.is_active == True)
            .values(is_active=False)
        )
        return res.rowcount == 1

    @staticmethod
    async def get_all_users_tokens(db: AsyncSession) -> list[Mapping[str, Any]]:
        """Return [{username, tokens}, ...] for all active users."""
        result = await db.execute(
            select(User.username, User.tokens)
            .where(User.is_active == True)
            .order_by(User.created_at)
        )
        return result.mappings().all()
