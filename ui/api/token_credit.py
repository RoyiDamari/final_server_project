from ui.api.base import api_call


def buy_tokens(token: str, credit_card: str, amount: int, idempotency_key: str):
    return api_call(
        "/token_credit/buy_tokens",
        method="POST",
        token=token,
        json={"credit_card": credit_card, "amount": amount, "idempotency_key": idempotency_key}
    )


def get_user_tokens(token: str):
    return api_call(
        "/token_credit/user_tokens",
        method="GET",
        token=token,
    )


def get_all_users_tokens(token: str):
    return api_call(
        "/token_credit/all_users_tokens",
        method="GET",
        token=token,
    )
