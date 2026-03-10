from fastapi import status
from .base import BaseAppException


class TokenGenerationException(BaseAppException):
    """
    Raised when a refresh/session token cannot be generated uniquely after retries.
    Returns HTTP 401 (by your design) and does not log.
    """

    def __init__(self, detail: str = "Unable to generate unique session token pair."):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class InvalidTokenException(BaseAppException):
    """
    Raised when an access/refresh token is invalid or cannot be validated.
    Returns HTTP 401 and does not log (common auth failure).
    """

    def __init__(self, detail: str = "Your session could not be validated. Please re-authenticate."):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class ExpiredTokenException(BaseAppException):
    """
    Raised when an access/refresh token is expired (or absolute expiry reached).
    Returns HTTP 401 and does not log.
    """

    def __init__(self, detail: str = "Session has expired. Please log in again"):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class UserCredentialsException(BaseAppException):
    """
    Raised when login fails due to incorrect username/password.
    Returns HTTP 401 and does not log.
    """

    def __init__(self, detail: str = "Incorrect username or password"):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class ReusedTokenException(BaseAppException):
    """
    Raised when refresh-token reuse is detected and the session is revoked.
    Returns HTTP 401 and logs internal details (security signal).
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="⚠️ Your session was revoked due to suspicious activity. "
                   "Another device may have tried to use your credentials. "
                   "Please log in again to continue.",
            status_code=status.HTTP_401_UNAUTHORIZED,
            log_detail=log_detail,
            suppress_log=False,
        )
