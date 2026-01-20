from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Result
from app.models.orm_models.token_credits import TokenCredit
from app.models.orm_models.users import User
from app.models.enums import RowStatus
from typing import Mapping, Any


class TokenCreditRepository:
    @staticmethod
    async def try_insert_pending(db: AsyncSession, user_id: int, key: str) -> bool:
        """
        Insert a 'pending' ledger row once (idempotency key).
        Returns True iff inserted; False if duplicate key (ignored).
        """
        stmt = (
            pg_insert(TokenCredit)
            .values(user_id=user_id, key=key, status=RowStatus.pending)
            .on_conflict_do_nothing(index_elements=["user_id", "key"])
        )
        res: Result = await db.execute(stmt)
        return res.rowcount == 1

    @staticmethod
    async def get_by_key_status_open_balance(db: AsyncSession, user_id: int, key: str) -> Mapping[str, Any] | None:
        """Return {'status': CreditStatus, 'open_balance': int|None} for this key, or None."""
        q = select(
            TokenCredit.status.label("status"),
            TokenCredit.open_balance.label("open_balance"),
        ).where(
            TokenCredit.user_id == user_id,
            TokenCredit.key == key,
        )
        return (await db.execute(q)).mappings().first()

    @staticmethod
    async def mark_applied(db: AsyncSession, user_id: int, key: str, open_balance: int) -> None:
        """Set status=applied and persist open_balance snapshot."""
        await db.execute(
            update(TokenCredit)
            .where(
                TokenCredit.user_id == user_id,
                TokenCredit.key == key,
            )
            .values(status=RowStatus.applied, open_balance=open_balance)
        )

    @staticmethod
    async def mark_failed(db: AsyncSession, user_id: int, key: str) -> None:
        """Set status=failed for this key."""
        await db.execute(
            update(TokenCredit)
            .where(
                TokenCredit.user_id == user_id,
                TokenCredit.key == key,
            )
            .values(status=RowStatus.failed)
        )

    @staticmethod
    async def get_user_token_history(db: AsyncSession, user_id: int) -> list[Mapping[str, Any]]:
        """
        Return token credit history for a given user_id including username from User.
        """
        result = await db.execute(
            select(
                TokenCredit.open_balance,
                TokenCredit.status,
                TokenCredit.created_at,
                User.username,
            )
            .join(User, User.id == TokenCredit.user_id)
            .where(TokenCredit.user_id == user_id)
            .order_by(TokenCredit.created_at)
        )
        return result.mappings().all()
