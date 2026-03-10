from datetime import datetime
from sqlalchemy import (
    DateTime, ForeignKey, Integer, String, UniqueConstraint, Index, func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class SeenVersion(Base):
    __tablename__ = "seen_versions"
    __table_args__ = (
        UniqueConstraint("user_id", "resource", "version", name="uq_seen_user_resource_version"),
        Index("ix_seen_user_resource", "user_id", "resource"),
        Index("ix_seen_resource_version", "resource", "version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    resource: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="seen_versions")
