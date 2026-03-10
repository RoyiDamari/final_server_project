from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models.orm_models.seen_versions import SeenVersion


class SeenVersionRepository:
    @staticmethod
    async def insert_seen(
            db: AsyncSession,
            user_id: int,
            resource: str,
            version: str,
    ) -> bool:
        """
        Insert a durable "seen version" marker for per-user billing.

        Args:
            db: Async SQLAlchemy session.
            user_id: The viewer user id (the user who may be charged).
            resource: Logical resource name (e.g., "models:all", "preds:all", "usage:model_type").
            version: Dataset version string (typically ISO timestamp).

        Returns:
            bool:
                - True if inserted now (first time this user sees this resource+version)
                - False if it already existed (already charged/seen)
        """

        q = (
            pg_insert(SeenVersion)
            .values(user_id=user_id, resource=resource, version=version)
            .on_conflict_do_nothing(index_elements=["user_id", "resource", "version"])
            .returning(SeenVersion.id)
        )
        inserted_id = (await db.execute(q)).scalar_one_or_none()
        return inserted_id is not None
