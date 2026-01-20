from fastapi import status
from .base import BaseAppException


class RateLimitException(BaseAppException):
    """
    Raised when a user or IP has exceeded the maximum allowed number of requests
    within a defined time window.

    This is used for both authenticated (user.id) and unauthenticated (IP) rate limiting.

    Returns:
        HTTP 429 Too Many Requests
    """
    def __init__(self, retry_after: float):
        super().__init__(
            detail="Rate limit exceeded. Please try again later.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            suppress_log=True,
        )
        self.retry_after = retry_after

