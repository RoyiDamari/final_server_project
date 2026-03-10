from fastapi import status
from .base import BaseAppException


class AssistInputException(BaseAppException):
    """
    Raised when assist endpoint inputs do not match any supported mode.
    Returns HTTP 422 and does not log (user-correctable).
    """

    def __init__(self, detail: str | None = None):
        super().__init__(
            detail,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            suppress_log=True,
        )


class AssistUnavailableException(BaseAppException):
    """
    Raised when assist endpoint inputs do not match any supported mode.
    Returns HTTP 422 and does not log (user-correctable).
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="OpenAI service unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            log_detail=log_detail,
            suppress_log=False,
        )


class OpenAIConfigException(BaseAppException):
    """
    Raised when the OpenAI dependency is misconfigured (missing key/SDK).
    Returns HTTP 502 and logs internal details.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="OpenAI configuration error",
            status_code=status.HTTP_502_BAD_GATEWAY,
            log_detail=log_detail,
            suppress_log=False,
        )


class OpenAIRequestException(BaseAppException):
    """
    Raised when the OpenAI request fails (network/timeout/empty response).
    Returns HTTP 502 Bad Gateway (upstream dependency failed).
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="OpenAI request failed",
            status_code=status.HTTP_502_BAD_GATEWAY,
            log_detail=log_detail,
            suppress_log=False,
        )


class AssistInProgressException(BaseAppException):
    """
    Raised when an identical Assist request is already being processed.
    Returns HTTP 409 and does not log.
    """

    def __init__(self) -> None:
        super().__init__(
            detail="Assist request is already in progress. Please wait a moment and try again.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class AssistFailedException(BaseAppException):
    """
    Raised when Assist fails due to internal errors not related to OpenAI configuration.
    Returns HTTP 500 and logs internal details.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Assist failed due to an internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
