from fastapi import status
from .base import BaseAppException


class UsernameFormatException(BaseAppException):
    def __init__(self):
        super().__init__(
            detail="Username must be 3–20 characters and include only letters, digits, underscores, or hyphens.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            suppress_log=True,
        )


class PasswordFormatException(BaseAppException):
    def __init__(self):
        super().__init__(
            detail="Password must be 6–20 chars and include at least one letter and one number.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            suppress_log=True,
        )


class UsernameTakenException(BaseAppException):
    """
    Raised when trying to register a user with a username that already exists.

    Default HTTP status: 400 Bad Request
    """

    def __init__(self):
        super().__init__(
            detail="Username already taken",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class EmailTakenException(BaseAppException):
    """
    Raised when trying to register a user with an email that already exists.

    Default HTTP status: 400 Bad Request
    """

    def __init__(self):
        super().__init__(
            detail="Email already taken",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class NotEnoughTokensException(BaseAppException):
    """
    Raised when a user tries to perform an action but lacks enough tokens.

    Default HTTP status: 400 Bad Request
    """

    def __init__(self):
        super().__init__(
            detail="Not enough tokens for this action",
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class UserNotFoundException(BaseAppException):
    """
    Raised when the requested user does not exist (or is not accessible).

    Default HTTP status: 404 Not Found
    """

    def __init__(self):
        super().__init__(
            detail="User not found",
            status_code=status.HTTP_404_NOT_FOUND,
            suppress_log=True,
        )


class UserAlreadyDeletedException(BaseAppException):
    def __init__(self):
        super().__init__(
            detail="User is already deleted.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class UserHasRemainingTokensException(BaseAppException):
    def __init__(self, detail: str | None = None):
        super().__init__(
            detail,
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class DeleteUserConfirmationException(BaseAppException):
    """
    Raised when username/password confirmation fails
    during delete-account flow (user is already authenticated).
    """

    def __init__(self, detail: str = "Incorrect username or password confirmation"):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_403_FORBIDDEN,
            suppress_log=True,
        )