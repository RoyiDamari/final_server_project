import json
from typing import Any
from ui.api.base import api_call


def train_model(
        token: str,
        file: Any,
        model_type: str,
        features: list[str],
        label: str,
        model_params: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Train a model on uploaded CSV data.

    Sends a multipart/form-data request:
      - file is uploaded as "data.csv"
      - features/model_params are JSON-encoded strings in form fields

    Args:
        token: User access token.
        file: File-like object (BytesIO). Will be read and reset to position 0.
        model_type: Model type ("linear", "logistic", "random_forest").
        features: List of feature column names.
        label: Target column name.
        model_params: Model parameters dict (will be JSON-encoded).

    Returns:
        API response dict (wrapper) or None on network failure.
    """

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


def get_user_models(token: str) -> dict[str, Any] | None:
    """
    Fetch trained models for the current authenticated user.

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/train_model/user_models",
        method="GET",
        token=token,
    )


def get_all_users_models(token: str) -> dict[str, Any] | None:
    """
    Fetch trained models for all users (admin/privileged endpoint).

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/train_model/all_users_models",
        method="GET",
        token=token,
    )


def get_user_models_internal(token: str) -> dict[str, Any] | None:
    """
    Fetch trained models for internal UX composition.

    Notes:
        - No rate limit
        - No token charge
        - Used for UI tasks like populating dropdowns (prediction form, feature selection)

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/train_model/user_models_internal",
        method="GET",
        token=token,
    )
