from datetime import datetime
from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Integer, String, func, Index, text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import CITEXT
from .base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("tokens >= 0 AND tokens <= 1000", name="ck_users_tokens_0_1000"),
        Index(
            "ux_users_username_active",
            "username",
            unique=True,
            postgresql_where=text("is_active = true")
        ),
        Index(
        "ux_users_email_active",
        "email",
        unique=True,
        postgresql_where=text("is_active = true"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    username: Mapped[str] = mapped_column(CITEXT, unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(CITEXT, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # One-to-many: a user owns many trained models, predictions, and token credits
    trained_models: Mapped[list["TrainedModel"]] = relationship(
        back_populates="user",
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="user",
    )
    token_credits: Mapped[list["TokenCredit"]] = relationship(
        back_populates="user",
    )
    auth_sessions: Mapped[list["AuthSession"]] = relationship(
        back_populates="user",
    )
