from fastapi import status
from .base import BaseAppException


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


class BuyTokensException(BaseAppException):
    """
    Raised when an invalid token amount is provided during purchase.

    Default HTTP status: 400 Bad Request
    """

    def __init__(self):
        super().__init__(
            detail="Invalid token amount request",
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


class BalanceMustBeZeroException(BaseAppException):
    """
    Raised when top-up is attempted while tokens != 0 (your zero-balance policy).
    HTTP 409 Conflict.
    """

    def __init__(self):
        super().__init__(
            detail="Cannot top up: balance must be 0.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )
