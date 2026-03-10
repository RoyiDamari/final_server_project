from typing import Any
from ui.api.base import api_call


def register_user(
        first_name: str,
        last_name: str,
        username: str,
        email: str,
        password: str,
) -> dict[str, Any] | None:
    """
    Register a new user account via the backend.

    This calls the public endpoint:
        POST /user/register

    Notes:
        - allow_refresh=False because this is an unauthenticated flow.
        - Backend is expected to validate and return a standard response wrapper.

    Args:
        first_name: User's first name.
        last_name: User's last name.
        username: Desired username (caller usually lowercases/strips).
        email: User email (caller usually lowercases/strips).
        password: Plaintext password to send to backend (HTTPS required).

    Returns:
        API response wrapper dict on success/error, or None on network failure.
    """

    return api_call(
        "/user/register",
        method="POST",
        allow_refresh=False,
        json={
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "email": email,
            "password": password,
        }
    )


def delete_user(
        token: str,
        username: str,
        password: str,
        confirm_delete_with_balance: bool,
) -> dict[str, Any] | None:
    """
    Delete the current user's account via the backend.

    This calls the authenticated endpoint:
        DELETE /user/delete

    The backend may require explicit confirmation when the user still has a token balance.

    Args:
        token: User access token (Authorization Bearer).
        username: Username confirmation (for safety).
        password: Password confirmation (for safety).
        confirm_delete_with_balance: If True, confirms deletion even if token balance > 0.

    Returns:
        API response wrapper dict on success/error, or None on network failure.
    """

    return api_call(
        "/user/delete",
        method="DELETE",
        token=token,
        json={
            "username": username,
            "password": password,
            "confirm_delete_with_balance": confirm_delete_with_balance,
        }
    )
