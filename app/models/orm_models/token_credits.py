from datetime import datetime
from sqlalchemy import (
    DateTime, Enum as SAEnum, ForeignKey, Integer, String,
    UniqueConstraint, CheckConstraint, func, Index, text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base
from app.models.enums import RowStatus


class TokenCredit(Base):
    __tablename__ = "token_credits"
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="ux_token_credits_user_key"),
        CheckConstraint(
            "(status <> 'applied' AND open_balance IS NULL) "
            "OR (status = 'applied' AND open_balance BETWEEN 1 AND 100)",
            name="ck_token_credits_open_balance_when_applied",
        ),
        Index(
            "ux_token_credits_one_pending_per_user",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    open_balance: Mapped[int] = mapped_column(Integer, nullable=True)
    status: Mapped[RowStatus] = mapped_column(
        SAEnum(RowStatus, name="row_status", native_enum=True),
        nullable=False,
        default=RowStatus.pending,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # many-to-one → user
    user: Mapped["User"] = relationship(
        back_populates="token_credits",
    )
