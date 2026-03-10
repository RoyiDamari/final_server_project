from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete
from app.models.orm_models.auth_sessions import AuthSession
from datetime import datetime
from typing import Optional


class AuthRepository:
    """
    DB access layer for AuthSession lifecycle (refresh token sessions).

    Key patterns:
        - Insert new session rows for login.
        - Row-lock a session for refresh rotation (FOR UPDATE).
        - Revoke by session or by user.
    """

    @staticmethod
    async def insert_new_refresh_token(
            db: AsyncSession,
            session_id: str,
            user_id: int,
            refresh_hash: str,
            expires_at: datetime,
            absolute_expires_at: datetime,
            ip_address: str,
            user_agent: str,
    ) -> None:
        """
        Add a new AuthSession ORM object to the current SQLAlchemy session.

        Caller responsibilities:
          - Run inside a transaction (`async with db.begin(): ...`) so the insert commits.
          - Handle unique constraint collisions at the service layer if desired.

        Args:
            db: Async SQLAlchemy session.
            session_id: Unique session identifier for this login session.
            user_id: Owner user id.
            refresh_hash: Hash of the raw refresh token (never store raw token).
            expires_at: Sliding expiry timestamp.
            absolute_expires_at: Absolute expiry timestamp (max lifetime).
            ip_address: Client IP (optional, for audit).
            user_agent: Client user-agent string (optional, for audit).

        Returns:
            None
        """

        token = AuthSession(
            session_id=session_id,
            user_id=user_id,
            refresh_token_hash=refresh_hash,
            expires_at=expires_at,
            absolute_expires_at=absolute_expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(token)

    @staticmethod
    async def get_refresh_token(
            db: AsyncSession,
            token_hash: str,
    ) -> Optional[AuthSession]:
        """
        Fetch an AuthSession by refresh_token_hash and lock it FOR UPDATE.

        Why FOR UPDATE:
          - Prevent concurrent refresh requests from rotating the same session row.
          - Makes reuse detection and revoke logic race-safe.

        Args:
            db: Async SQLAlchemy session.
            token_hash: Hash of the presented refresh token.

        Returns:
            Optional[AuthSession]: Session row with AuthSession.user eager-loaded,
            or None if not found.
        """

        q = (
            select(AuthSession)
            .where(AuthSession.refresh_token_hash == token_hash)
            .options(selectinload(AuthSession.user))
            .with_for_update()
        )
        return (await db.execute(q)).unique().scalars().first()

    @staticmethod
    async def rotate_refresh_token(
            db: AsyncSession,
            session_id: str,
            new_token_hash: str,
            last_token_hash: str,
            new_expiry: datetime,
    ) -> None:
        """
        Update an AuthSession row with the newly rotated refresh token hash.

        Caller responsibilities:
          - Ensure concurrency safety (normally by holding FOR UPDATE lock).
          - Perform validation checks (revoked/expired/reuse) in the service layer.

        Args:
            db: Async SQLAlchemy session.
            session_id: Session identifier to update.
            new_token_hash: Hash of the new refresh token.
            last_token_hash: Hash of the presented token (stored for reuse detection).
            new_expiry: New sliding expiry timestamp.

        Returns:
            None
        """

        q = (
            update(AuthSession)
            .where(AuthSession.session_id == session_id)
            .values(
                refresh_token_hash=new_token_hash,
                last_token_hash=last_token_hash,
                expires_at=new_expiry,
            )
        )
        await db.execute(q)

    @staticmethod
    async def revoke_by_session(db: AsyncSession, session_id: str) -> None:
        """
        Mark a session revoked=True (idempotent).

        Args:
            db: Async SQLAlchemy session.
            session_id: Session identifier to revoke.

        Returns:
            None
        """

        q = (
            update(AuthSession)
            .where(AuthSession.session_id == session_id)
            .values(revoked=True)
        )
        await db.execute(q)

    @staticmethod
    async def revoke_all_sessions_by_user(db: AsyncSession, user_id: int) -> None:
        """
        Revoke all sessions for a user (idempotent).

        Args:
            db: Async SQLAlchemy session.
            user_id: User id whose sessions should be revoked.

        Returns:
            None
        """

        q = (
            update(AuthSession)
            .where(AuthSession.user_id == user_id)
            .values(revoked=True)
        )
        await db.execute(q)
