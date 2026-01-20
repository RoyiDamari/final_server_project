from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete
from app.models.orm_models.auth_sessions import AuthSession
from datetime import datetime
from typing import Optional


class AuthRepository:
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
        stmt = (
            select(AuthSession)
            .where(AuthSession.refresh_token_hash == token_hash)
            .options(selectinload(AuthSession.user))
            .with_for_update()
        )
        result = await db.execute(stmt)
        return result.unique().scalars().first()

    @staticmethod
    async def rotate_refresh_token(
        db: AsyncSession,
        session_id: str,
        new_token_hash: str,
        last_token_hash: str,
        new_expiry: datetime,
    ) -> None:
        stmt = (
            update(AuthSession)
            .where(AuthSession.session_id == session_id)
            .values(
                refresh_token_hash=new_token_hash,
                last_token_hash=last_token_hash,
                expires_at=new_expiry,
            )
        )
        await db.execute(stmt)

    @staticmethod
    async def revoke_by_session(db: AsyncSession, session_id: str) -> None:
        stmt = (
            update(AuthSession)
            .where(AuthSession.session_id == session_id)
            .values(revoked=True)
        )
        await db.execute(stmt)

    @staticmethod
    async def revoke_all_session_by_user(db: AsyncSession, user_id: int) -> None:
        stmt = (update(AuthSession)
                .where(AuthSession.user_id == user_id)
                .values(revoked=True)
                )
        await db.execute(stmt)
