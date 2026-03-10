from ui.api.base import api_call


def get_ready() -> dict | None:
    """
    Call backend readiness endpoint.

    Returns:
        dict | None:
            - {"status_code": 200, ...} on success
            - {"status_code": 503, ...} if not ready
            - None on network/timeout
    """
    return api_call("/ready", method="GET", token=None, allow_refresh=False)
