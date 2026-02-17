from pydantic import BaseModel, EmailStr, UUID4, Field, ConfigDict, field_validator
from app.exceptions.token_credit import InvalidCreditCardException, InvalidPurchaseAmountException
from datetime import datetime
from app.models.enums import RowStatus
from app.config import config
import re


class BuyTokensRequest(BaseModel):
    credit_card: str
    amount: int
    idempotency_key: UUID4

    @field_validator("credit_card", mode="before")
    @classmethod
    def _normalize_and_validate_card(cls, value: str) -> str:
        digits = re.sub(r"\D", "", str(value or ""))
        if len(digits) != 16 or not digits.isdigit():
            raise InvalidCreditCardException()
        return digits

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, value: int) -> int:
        if value is None or value <= 0 or value > config.MAX_TOKENS_PER_PURCHASE:
            raise InvalidPurchaseAmountException()
        return value


class BuyTokensResponse(BaseModel):
    message: str
    balance: int


class TokenCreditResponse(BaseModel):
    username: str
    amount: int | None
    balance_after: int | None
    current_tokens: int
    status: RowStatus
    created_at: datetime
