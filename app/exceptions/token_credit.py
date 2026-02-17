from fastapi import status
from .base import BaseAppException
from app.config import config


class InvalidCreditCardException(BaseAppException):
    """
    Raised when an invalid credit card number is provided (e.g., not 16 digits).

    Default HTTP status: 400 Bad Request
    """

    def __init__(self):
        super().__init__(
            detail="Invalid credit card format. Must be 16 digits.",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class InvalidPurchaseAmountException(BaseAppException):
    """
    Raised when an invalid token amount is provided during purchase.

    Default HTTP status: 400 Bad Request
    """

    def __init__(self):
        super().__init__(
            detail=(
                f"Token purchase exceeds the allowed per-purchase limit. "
                f"You can buy up to {config.MAX_TOKENS_PER_PURCHASE} tokens per purchase."
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class TokenBalanceCapExceededException(BaseAppException):
    def __init__(self):
        super().__init__(
            detail=(
                f"Purchase would exceed your account token cap of {config.MAX_TOKENS} which is not allowed"
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class PurchaseInProgressException(BaseAppException):
    """
    Raised when a user tries to start a new purchase while another is still pending.
    HTTP 409 Conflict.
    """

    def __init__(self):
        super().__init__(
            detail="Another purchase is already in progress. Retry with the same key or wait.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class PurchaseFailedException(BaseAppException):
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Purchase tokens failed due to internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
