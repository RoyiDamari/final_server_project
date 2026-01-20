from fastapi import status


class BaseAppException(Exception):
    """Base exception class for all custom application exceptions."""

    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        log_detail: str | None = None,
        suppress_log: bool = False,
    ):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.log_detail = log_detail
        self.suppress_log = suppress_log
