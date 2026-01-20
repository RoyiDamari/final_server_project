from fastapi import status
from .base import BaseAppException


class InvalidFormatException(BaseAppException):
    def __init__(self, detail="Invalid payload format"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class MissingDataException(BaseAppException):
    def __init__(self, detail="You must select data",):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class InvalidFeatureException(BaseAppException):
    def __init__(self, detail="Some features are missing in the dataset"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class InvalidLabelException(BaseAppException):
    def __init__(self, detail="Label column not found in dataset"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class InvalidParamException(BaseAppException):
    def __init__(self, detail="Invalid model parameters"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class UnsupportedModelTypeException(BaseAppException):
    def __init__(self, detail="The requested model type is not supported"):
        super().__init__(
            detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True,
        )

class ModelTypeMismatchException(BaseAppException):
    def __init__(self, detail="Selected model type does not match the target variable type."):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
            suppress_log=True
        )

class TrainModelInProgressException(BaseAppException):
    def __init__(self):
        super().__init__(
            detail="Training is already in progress for these parameters. Please wait and retry shortly.",
            status_code=status.HTTP_409_CONFLICT,
            suppress_log=True,
        )

class TrainingFailedException(BaseAppException):
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Model training failed due to internal error. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )

class ArtifactWriteException(BaseAppException):
    def __init__(self, log_detail: str | None = None):
        super().__init__(
            detail="Failed to persist model artifact",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_detail=log_detail,
            suppress_log=False,
        )