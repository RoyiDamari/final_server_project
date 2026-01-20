from datetime import datetime
from typing import Dict, Any
from sqlalchemy import (
    DateTime, ForeignKey, Integer, String, UniqueConstraint, func, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.enums import RowStatus
from .base import Base


class TrainedModel(Base):
    __tablename__ = "trained_models"
    __table_args__ = (
        UniqueConstraint("user_id", "fingerprint", name="uq_user_model_fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    features: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    model_params: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    feature_schema: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
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
        back_populates="trained_models",
    )

    # one-to-many → predictions
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model",
    )
