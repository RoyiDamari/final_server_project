from fastapi import status


class BaseAppException(Exception):
    """
    Base class for application-level exceptions that map cleanly to HTTP responses.

    Attributes:
        detail: User-facing message returned in JSON {"detail": ...}.
        status_code: HTTP status code for the response.
        log_detail: Optional internal detail for server logs (not returned to user).
        suppress_log: If True, the global handler will not log the exception.
    """

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
