from fastapi import status
from .base import BaseAppException


class ModelNotFoundException(BaseAppException):
    """
    Raised when a requested trained model is missing, not owned, or not applied.
    Returns HTTP 404 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Model not found",
            status_code=status.HTTP_404_NOT_FOUND,
            suppress_log=True,
        )


class FeatureMismatchException(BaseAppException):
    """
    Raised when provided feature keys do not match the model’s expected feature set.
    Returns HTTP 400 and does not log (user-correctable).
    """

    def __init__(self, detail: str | None = None):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class PredictionInProgressException(BaseAppException):
    """
    Raised when an identical prediction request is already pending (idempotency collision).
    Returns HTTP 409 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Prediction already in progress for this idempotency key. Please wait and retry. "
                   "or click ‘Fetch Predictions’ to see when it finishes.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class PredictionFailedException(BaseAppException):
    """
    Raised when an identical prediction request is already pending (idempotency collision).
    Returns HTTP 500 and logs internal details.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Prediction failed due to internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
