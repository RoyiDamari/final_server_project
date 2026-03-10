from typing import Any
from ui.api.base import api_call


def buy_tokens(token: str, credit_card: str, amount: int, idempotency_key: str) -> dict[str, Any] | None:
    """
    Purchase tokens for the current user.

    Args:
        token: User access token.
        credit_card: Credit card number (usually digits-only on client).
        amount: Number of tokens to buy.
        idempotency_key: Client-generated key to prevent duplicate charges.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/token_credit/buy_tokens",
        method="POST",
        token=token,
        json={"credit_card": credit_card, "amount": amount, "idempotency_key": idempotency_key}
    )


def get_user_tokens(token: str) -> dict[str, Any] | None:
    """
    Fetch token purchase/usage history for the current user.

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/token_credit/user_tokens",
        method="GET",
        token=token,
    )


def get_all_users_tokens(token: str) -> dict[str, Any] | None:
    """
    Fetch token history for all users (admin/privileged endpoint).

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/token_credit/all_users_tokens",
        method="GET",
        token=token,
    )
