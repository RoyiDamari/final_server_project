from fastapi import status
from .base import BaseAppException


class InvalidFormatException(BaseAppException):
    """
    Raised when request payload JSON/format is invalid.
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="Invalid payload format"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class MissingDataException(BaseAppException):
    """
    Raised when request payload JSON/format is invalid.
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="You must select data", ):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class InvalidFeatureException(BaseAppException):
    """
    Raised when required inputs are missing (e.g., no features/label).
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="Some features are missing in the dataset"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class InvalidLabelException(BaseAppException):
    """
    Raised when requested feature columns are missing/invalid for the dataset.
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="Label column not found in dataset"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class InvalidParamException(BaseAppException):
    """
    Raised when model hyperparameters are unknown/invalid.
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="Invalid model parameters"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class UnsupportedModelTypeException(BaseAppException):
    """
    Raised when model hyperparameters are unknown/invalid.
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="The requested model type is not supported"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )


class ModelTypeMismatchException(BaseAppException):
    """
    Raised when model_type is not one of the supported strategies.
    Returns HTTP 400 and does not log.
    """

    def __init__(self, detail="Selected model type does not match the target variable type."):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True
        )


class TrainModelInProgressException(BaseAppException):
    """
    Raised when an identical training request is already pending (idempotency collision).
    Returns HTTP 409 and does not log.
    """

    def __init__(self):
        super().__init__(
            detail="Training is already in progress for these parameters. Please wait and retry shortly. "
                   "or click ‘Fetch Models’ to see when it finishes.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )


class TrainingFailedException(BaseAppException):
    """
    Raised when training fails unexpectedly (worker, publish, DB apply).
    Returns HTTP 500 and logs internal details.
    """

    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Model training failed due to internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )
