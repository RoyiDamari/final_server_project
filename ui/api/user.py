from ui.api.base import api_call


def register_user(first_name: str, last_name: str, username: str, email: str, password: str):
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


def delete_user(token: str, username: str, password: str, confirm_delete_with_balance: bool):
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


def get_all_users_tokens(token: str):
    return api_call(
        "/user/all_users_tokens",
        method="GET",
        token=token,
    )
