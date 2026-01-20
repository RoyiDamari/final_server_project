import json
from ui.api.base import api_call


def train_model(token: str, file, model_type: str, features: list[str], label: str, model_params: dict):
    # file is a BytesIO — convert to multipart tuple
    file_bytes = file.read()
    file.seek(0)

    return api_call(
        "/train_model/train",
        method="POST",
        token=token,
        files={"file": ("data.csv", file_bytes, "text/csv")},
        data={
            "model_type": model_type,
            "features": json.dumps(features),
            "label": label,
            "model_params": json.dumps(model_params),
        }
    )


def get_user_models(token: str):
    return api_call(
        "/train_model/user_models",
        method="GET",
        token=token,
    )


def get_all_users_models(token: str):
    return api_call(
        "/train_model/all_users_models",
        method="GET",
        token=token,
    )


def get_user_models_internal(token: str):
    """
    INTERNAL metadata fetch:
    - no rate limit
    - no token charge
    - used for UX composition (prediction form, feature selection)
    """
    return api_call(
        "/train_model/user_models_internal",
        method="GET",
        token=token,
    )