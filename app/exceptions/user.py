from fastapi import status
from .base import BaseAppException


class UsernameFormatException(BaseAppException):
    """
    Raised when training fails unexpectedly (worker, publish, DB apply).
    Returns HTTP 422 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Username must be 3–20 characters and include only letters, digits, underscores, or hyphens.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            suppress_log=True,
        )


class PasswordFormatException(BaseAppException):
    """
    Raised when username does not meet formatting rules.
    Returns HTTP 422 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Password must be 6–20 chars and include at least one letter and one number.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            suppress_log=True,
        )


class UsernameTakenException(BaseAppException):
    """
    Raised when registering with an existing active username.
    Returns HTTP 400 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Username already taken",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class EmailTakenException(BaseAppException):
    """
    Raised when registering with an existing active username.
    Returns HTTP 400 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Email already taken",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class NotEnoughTokensException(BaseAppException):
    """
    Raised when a user attempts an action without sufficient tokens.
    Returns HTTP 400 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Not enough tokens for this action",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class UserNotFoundException(BaseAppException):
    """
    Raised when a requested user does not exist or is inactive.
    Returns HTTP 404 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="User not found",
            status_code=status.HTTP_404_NOT_FOUND,
            suppress_log=True,
        )


class UserAlreadyDeletedException(BaseAppException):
    """
    Raised when a requested user does not exist or is inactive.
    Returns HTTP 404 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="User is already deleted.",
            status_code=status.HTTP_404_NOT_FOUND,
            suppress_log=True,
        )


class UserHasRemainingTokensException(BaseAppException):
    """
    Raised when deletion is blocked because the user still has tokens.
    Returns HTTP 409 and does not log.
    """

    def __init__(self, detail: str | None = None):
        super().__init__(
            detail,
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class DeleteUserConfirmationException(BaseAppException):
    """
    Raised when delete-account confirmation credentials do not match.
    Returns HTTP 403 and does not log.
    """

    def __init__(self, detail: str = "Incorrect username or password confirmation"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_403_FORBIDDEN,
            suppress_log=True,
        )


class UserDisconnectedException(BaseAppException):
    """
    Raised when the client disconnects during a long-running operation.
    Returns HTTP 499 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="User disconnected. Training cancelled.",
            status_code=499,
            suppress_log=True,
        )
