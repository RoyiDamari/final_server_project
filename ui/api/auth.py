import requests
from typing import Any
from requests.exceptions import RequestException, Timeout
from ui.utils.api_helpers import logout_and_stop
from ui.config import API_BASE_URL
from ui.api.base import api_call


def login_user(username: str, password: str) -> dict | None:
    """
    Authenticate a user and obtain access/refresh tokens.

    Args:
        username: Username (expected lowercase in caller).
        password: Plaintext password.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/auth/login",
        method="POST",
        allow_refresh=False,
        json={"username": username, "password": password}
    )


def refresh_token(refresh_tok: str) -> dict[str, Any] | None:
    """
    Exchange a refresh token for a new access token (and usually a rotated refresh token).

    Important:
        This function intentionally does NOT call api_call() to avoid recursion loops,
        because api_call() may itself attempt a refresh on 401.

    Failure behavior:
        - On missing token, network error, non-200 response, or invalid JSON:
          calls logout_and_stop(...) and returns None.

    Args:
        refresh_tok: Refresh token string sent as Bearer in the Authorization header.

    Returns:
        A dict parsed from backend JSON on success (expected keys like:
        "access_token", "refresh_token", "expires_at"),
        or None on failure (after logging out).
    """

    if not refresh_tok:
        logout_and_stop("Session expired. Please log in again.")
        return

    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/refresh",
            headers={"Authorization": f"Bearer {refresh_tok}"},
            timeout=10,
        )
    except (RequestException, Timeout):
        logout_and_stop("Server unavailable. Please log in again.")
        return

    if response.status_code != 200:
        logout_and_stop("Session expired. Please log in again.")
        return

    try:
        return response.json()
    except ValueError:
        logout_and_stop("Invalid refresh response from server.")
        return


def logout_user(access_tok: str, refresh_tok: str) -> None:
    """
    Notify backend to revoke the current refresh token (best-effort).

    Args:
        access_tok: Current access token.
        refresh_tok: Current refresh token.

    Returns:
        None.
    """

    try:
        requests.delete(
            f"{API_BASE_URL}/auth/logout",
            headers={
                "Authorization": f"Bearer {access_tok}",
            },
            json={
                "refresh_token": refresh_tok,
            },
            timeout=10,
        )
    except RequestException:
        pass
