import requests
from requests.exceptions import RequestException, Timeout
from ui.utils.api_helpers import logout_and_stop
from ui.config import API_BASE_URL
from ui.api.base import api_call


def login_user(username: str, password: str):
    return api_call(
        "/auth/login",
        method="POST",
        allow_refresh=False,
        json={"username": username, "password": password}
    )

def refresh_token(refresh_tok: str) -> dict | None:
    """
    Send refresh token to backend to get a new access + refresh token.
    IMPORTANT: This must NOT call api_call() to avoid recursion loops.
    Returns dict on success, None on failure (after logging out).
    """

    if not refresh_tok:
        logout_and_stop("Session expired. Please log in again.")
        return

    # Network-level failure
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/refresh",
            headers={"Authorization": f"Bearer {refresh_tok}"},
            timeout=10,
        )
    except (RequestException, Timeout):
        logout_and_stop("Server unavailable. Please log in again.")
        return

    # HTTP-level failure
    if response.status_code != 200:
        logout_and_stop("Session expired. Please log in again.")
        return

    # JSON decode
    try:
        return response.json()
    except ValueError:
        logout_and_stop("Invalid refresh response from server.")
        return


def logout_user(access_tok: str, refresh_tok: str) -> None:
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

