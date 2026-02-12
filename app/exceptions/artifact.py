from fastapi import status
from .base import BaseAppException


class ArtifactWriteException(BaseAppException):
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Failed to persist model artifact",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )


class ArtifactMissingException(BaseAppException):
    """Raised when the model artifact path is missing or the file does not exist on disk."""

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Model artifact is missing",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )