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
        Idempotency gate: create a PENDING TokenCredit row for (user_id, key) exactly once.

        Important nuance:
        - Returns None both when:
            (a) (user_id, key) already exists (duplicate/replay), OR
            (b) a different PENDING row already exists for this user due to the
                partial-unique constraint (one pending per user).

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner user id.
            key: Idempotency key for this purchase attempt.

        Returns:
            int | None:
                - int: TokenCredit.id if inserted now (this request owns the purchase)
                - None: if the row already exists OR if another pending row blocks insertion
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
        """
        Fetch a token-purchase row by (user_id, key), returning only the fields
        the service needs to decide idempotency behavior.

        This is used to disambiguate cases where try_insert_pending() returns None:
            - same key already exists (replay), OR
            - a different pending row exists (one-pending-per-user constraint).

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner user id.
            key: Idempotency key for the purchase attempt.

        Returns:
            dict[str, Any] | None:
                - dict with keys: status, amount, balance_after, created_at
                - None if no row exists for (user_id, key)

        """

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
        """
        Transition a TokenCredit row from PENDING -> APPLIED and record purchase results.

        This update is conditional:
            - It only applies if the row is currently PENDING.
            - If the row is not PENDING (already applied/failed), returns None.

        Args:
            db: Async SQLAlchemy session.
            token_credit_id: TokenCredit primary key id.
            amount: Number of tokens purchased.
            balance_after: User token balance immediately after applying the purchase.

        Returns:
            dict[str, Any] | None:
                - dict with keys: created_at, amount, balance_after if the update succeeded
                - None if no row matched (wrong id or status not pending)

        """

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
    async def mark_failed(db: AsyncSession, token_credit_id: int) -> bool:
        """
        Mark a TokenCredit row as FAILED (best-effort state transition).

        Why:
          - If a purchase attempt crashes mid-flow, we can stop it from remaining "pending forever".
          - Keeps the table consistent with the CHECK constraint:
              for non-applied statuses, amount and balance_after must be NULL.

        Args:
            db: Async SQLAlchemy session.
            token_credit_id: TokenCredit primary key id.

        Returns:
            bool: True if a row was updated, False otherwise.
        """

        q = (
            update(TokenCredit)
            .where(TokenCredit.id == token_credit_id, TokenCredit.status == RowStatus.pending)
            .values(
                status=RowStatus.failed,
                amount=None,
                balance_after=None,
            )
        )
        res = await db.execute(q)
        return (res.rowcount or 0) > 0

    @staticmethod
    async def get_latest_created_at_all_users(db: AsyncSession) -> Optional[datetime]:
        """
        Return the latest TokenCredit.created_at across ACTIVE users, considering only APPLIED rows.

        This is used as a dataset "version" for global dashboards and cache invalidation.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            datetime | None:
                - datetime of max(created_at) among applied purchases for active users
                - None if no applied purchases exist
        """

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
        Fetch token-credit history for a single user (chronological).

        Output includes:
          - amount: tokens bought in this purchase (applied rows)
          - balance_after: user's balance immediately after this purchase (applied rows)
          - current_tokens: shown only on the latest row for this user; 0 on older rows

        Args:
            db: Async SQLAlchemy session.
            user_id: Owner user id.

        Returns:
            list[Mapping[str, Any]]:
                List of rows (mapping objects) containing:
                  username, amount, balance_after, current_tokens, status, created_at
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
        """
        Fetch token-credit history across all ACTIVE users (grouped by user, chronological per user).

        Output includes:
            - amount, balance_after, status, created_at
            - current_tokens only on each user's latest row (0 on older rows)

        Args:
            db: Async SQLAlchemy session.

        Returns:
            list[Mapping[str, Any]]:
                List of rows (mapping objects) containing:
                    username, amount, balance_after, current_tokens, status, created_at
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
            .where(User.is_active.is_(True))
            .order_by(User.username, TokenCredit.created_at.asc())
        )

        return (await db.execute(q)).mappings().all()

    @staticmethod
    async def list_ids_by_status(db: AsyncSession, status: RowStatus) -> list[int]:
        """
        List token_credit ids in the given status (ACTIVE users only).

        Args:
            db: Async SQLAlchemy session.
            status: RowStatus to filter by.

        Returns:
            list[int]: TokenCredit ids.
        """
        q = (
            select(TokenCredit.id)
            .where(
                TokenCredit.status == status,
                TokenCredit.user.has(is_active=True),
            )
        )
        return [int(x) for x in (await db.execute(q)).scalars().all()]