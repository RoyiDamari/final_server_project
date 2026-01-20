from ui.api.base import api_call


def predict(token: str, model_id: str, feature_values: dict):
    return api_call(
        "/prediction/predict",
        method="POST",
        token=token,
        json={"model_id": model_id, "feature_values": feature_values}
    )


def get_user_predictions(token: str):
    return api_call(
        "/prediction/user_predictions",
        method="GET",
        token=token,
    )


def get_all_users_predictions(token: str):
    return api_call(
        "/prediction/all_users_predictions",
        method="GET",
        token=token,
    )
