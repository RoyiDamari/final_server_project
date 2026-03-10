from fastapi import status
from .base import BaseAppException


class ArtifactWriteException(BaseAppException):
    """
    Raised when persisting (moving/writing) a model artifact to disk fails.
    Returns HTTP 500 and logs internal details for debugging.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Failed to persist model artifact",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )


class ArtifactMissingException(BaseAppException):
    """
    Raised when a model artifact is missing or inaccessible on disk.
    Returns HTTP 500 and logs internal details for debugging.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Model artifact is missing",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
