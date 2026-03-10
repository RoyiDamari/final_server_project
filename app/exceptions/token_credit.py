from fastapi import status
from .base import BaseAppException
from app.config import config


class InvalidCreditCardException(BaseAppException):
    """
    Raised when a credit card value is not a valid 16-digit format.
    Returns HTTP 400 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Invalid credit card format. Must be 16 digits.",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class InvalidPurchaseAmountException(BaseAppException):
    """
    Raised when the requested purchase amount violates configured limits.
    Returns HTTP 400 and does not log.
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
    """
    Raised when the requested purchase amount violates configured limits.
    Returns HTTP 400 and does not log.
    """

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
    Raised when a token purchase would exceed the user’s max token cap.
    Returns HTTP 400 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Another purchase is already in progress. Retry with the same key or wait. "
                   "or click ‘Fetch Tokens’ to see when it finishes.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class PurchaseFailedException(BaseAppException):
    """
    Raised when a user starts a purchase while another purchase is pending.
    Returns HTTP 409 and does not log.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Purchase tokens failed due to internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
