from datetime import datetime
from sqlalchemy import (
    DateTime, ForeignKey, Integer, JSON, String, func, UniqueConstraint, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.enums import RowStatus
from .base import Base


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("user_id", "fingerprint", name="uq_user_prediction_fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("trained_models.id", ondelete="RESTRICT"), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(255), nullable=False)
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction_result: Mapped[str] = mapped_column(String(255), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[RowStatus] = mapped_column(
        SAEnum(RowStatus, name="row_status", native_enum=True),
        nullable=False,
        default=RowStatus.pending
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # many-to-one
    user: Mapped["User"] = relationship(
        back_populates="predictions",
    )
    # many-to-one
    model: Mapped["TrainedModel"] = relationship(
        back_populates="predictions",
    )
