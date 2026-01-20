from fastapi import status
from .base import BaseAppException


class ModelNotFoundException(BaseAppException):
    """Raised when the requested model row does not exist or does not belong to the user."""

    def __init__(self):
        super().__init__(
            detail="Model not found",
            status_code=status.HTTP_404_NOT_FOUND,
            suppress_log=True,
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


class FeatureMismatchException(BaseAppException):
    """Raised when provided feature keys do not match the model's expected features."""

    def __init__(self, detail: str | None = None):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class PredictionInProgressException(BaseAppException):
    def __init__(self):
        super().__init__(
            detail="Prediction already in progress for this idempotency key. Please wait and retry.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class PredictionFailedException(BaseAppException):
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Prediction failed due to internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
