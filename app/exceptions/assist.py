from fastapi import status
from .base import BaseAppException


class AssistInputException(BaseAppException):
    """
    Raised when input to assist endpoints/services is invalid.
    HTTP 422 Unprocessable Entity.
    """
    def __init__(self, detail: str | None = None):
        super().__init__(
            detail,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            suppress_log=True,
        )


class AssistUnavailableException(BaseAppException):
    """
    Raised when the assist service cannot operate (e.g., not initialized).
    HTTP 503 Service Unavailable.
    """
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Assist service unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            log_detail=log_detail,
            suppress_log=False,
        )


class OpenAIConfigException(BaseAppException):
    """
    Raised when OpenAI is not configured correctly (missing key/SDK).
    HTTP 502 Bad Gateway (upstream dependency misconfigured).
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
    HTTP 502 Bad Gateway (upstream dependency failed).
    """
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="OpenAI request failed",
            status_code=status.HTTP_502_BAD_GATEWAY,
            log_detail=log_detail,
            suppress_log=False,
        )
