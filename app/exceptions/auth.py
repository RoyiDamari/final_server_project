from fastapi import status
from .base import BaseAppException


class TokenGenerationException(BaseAppException):
    """
    Raised when a JWT token is invalid, malformed, or missing required claims.

    Default HTTP status: 401 Unauthorized
    """

    def __init__(self, detail: str = "Unable to generate unique session token pair."):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class InvalidTokenException(BaseAppException):
    """
    Raised when a JWT token is invalid, malformed, or missing required claims.

    Default HTTP status: 401 Unauthorized
    """

    def __init__(self, detail: str = "Your session could not be validated. Please re-authenticate."):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class ExpiredTokenException(BaseAppException):
    """
    Raised when a JWT token has expired.

    Default HTTP status: 401 Unauthorized
    """

    def __init__(self, detail: str = "Session has expired. Please log in again"):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class UserCredentialsException(BaseAppException):
    """
    Raised when user authentication fails due to invalid credentials.

    Default HTTP status: 401 Unauthorized
    """

    def __init__(self, detail: str = "Incorrect username or password"):
        super().__init__(
            detail,
            status_code=status.HTTP_401_UNAUTHORIZED,
            suppress_log=True,
        )


class ReusedTokenException(BaseAppException):
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="⚠️ Your session was revoked due to suspicious activity. "
                   "Another device may have tried to use your credentials. "
                   "Please log in again to continue.",
            status_code=status.HTTP_401_UNAUTHORIZED,
            log_detail=log_detail,
            suppress_log=False,
        )
