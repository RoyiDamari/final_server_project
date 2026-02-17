from sqlalchemy import select, update, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.orm_models.token_credits import TokenCredit
from app.models.orm_models.users import User
from app.models.enums import RowStatus
from typing import Mapping, Any, Optional
from datetime import datetime


class TokenCreditRepository:
    @staticmethod
    async def try_insert_pending(
            db: AsyncSession,
            user_id: int,
            key: str,
    ) -> Optional[int]:
        """
        Idempotency gate: insert a pending row once per (user_id, key).
        Returns:
          - id if inserted now
          - None if already exists (pending or applied)
        """
        q = (
            pg_insert(TokenCredit)
            .values(user_id=user_id, key=key, status=RowStatus.pending)
            .on_conflict_do_nothing(index_elements=["user_id", "key"])
            .returning(TokenCredit.id)
        )
        try:
            return (await db.execute(q)).scalar_one_or_none()

        except IntegrityError as e:
            orig = getattr(e, "orig", None)

            constraint = getattr(orig, "constraint_name", None)

            if constraint == "ux_token_credits_one_pending_per_user":
                return None

            raise

    @staticmethod
    async def get_by_key_status_amount_balance_after(
        db: AsyncSession,
        user_id: int,
        key: str
    ) -> dict | None:
        q = (
            select(
                TokenCredit.status.label("status"),
                TokenCredit.amount.label("amount"),
                TokenCredit.balance_after.label("balance_after"),
                TokenCredit.created_at.label("created_at"),
            )
            .where(TokenCredit.user_id == user_id, TokenCredit.key == key)
        )
        row = (await db.execute(q)).mappings().first()
        return dict(row) if row else None

    @staticmethod
    async def mark_applied(
        db: AsyncSession,
        token_credit_id: int,
        amount: int,
        balance_after: int,
    ) -> dict | None:
        q = (
            update(TokenCredit)
            .where(TokenCredit.id == token_credit_id, TokenCredit.status == RowStatus.pending)
            .values(
                status=RowStatus.applied,
                amount=amount,
                balance_after=balance_after,
            )
            .returning(TokenCredit.created_at, TokenCredit.amount, TokenCredit.balance_after)
        )
        row = (await db.execute(q)).mappings().first()
        return dict(row) if row else None

    @staticmethod
    async def get_latest_created_at_all_users(db: AsyncSession) -> Optional[datetime]:
        q = (
            select(func.max(TokenCredit.created_at)).
            where(
            TokenCredit.status == RowStatus.applied,
            TokenCredit.user.has(is_active=True)
            )
        )
        return (await db.execute(q)).scalar_one_or_none()

    @staticmethod
    async def get_user_tokens(
        db: AsyncSession,
        user_id: int,
    ) -> list[Mapping[str, Any]]:
        """
        Token credit history for one user.

        Shows:
        - amount: tokens bought in this purchase (applied rows)
        - balance_after: user's balance immediately after this purchase (applied rows)
        - current_tokens: only on the latest row (per user); 0 on older rows
        """

        rn = func.row_number().over(
            partition_by=TokenCredit.user_id,
            order_by=TokenCredit.created_at.desc()
        )

        current_tokens = case(
            (rn == 1, User.tokens),
            else_=0
        ).label("current_tokens")

        q = (
            select(
                User.username.label("username"),
                TokenCredit.amount.label("amount"),
                TokenCredit.balance_after.label("balance_after"),
                current_tokens,
                TokenCredit.status.label("status"),
                TokenCredit.created_at.label("created_at"),
            )
            .join(User, User.id == TokenCredit.user_id)
            .where(TokenCredit.user_id == user_id)
            .order_by(TokenCredit.created_at.asc())
        )

        return (await db.execute(q)).mappings().all()

    @staticmethod
    async def get_all_users_tokens(db: AsyncSession) -> list[Mapping[str, Any]]:
        rn = func.row_number().over(
            partition_by=TokenCredit.user_id,
            order_by=TokenCredit.created_at.desc()
        )

        current_tokens = case(
            (rn == 1, User.tokens),
            else_=0
        ).label("current_tokens")

        q = (
            select(
                User.username.label("username"),
                TokenCredit.amount.label("amount"),
                TokenCredit.balance_after.label("balance_after"),
                current_tokens,
                TokenCredit.status.label("status"),
                TokenCredit.created_at.label("created_at"),
            )
            .join(User, User.id == TokenCredit.user_id)
            .where(User.is_active.is_(True))
            .order_by(User.username, TokenCredit.created_at.asc())
        )

        return (await db.execute(q)).mappings().all()