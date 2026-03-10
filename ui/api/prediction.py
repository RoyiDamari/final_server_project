from typing import Any
from ui.api.base import api_call


def predict(token: str, model_id: str, feature_values: dict[str, Any]) -> dict[str, Any] | None:
    """
    Request a prediction for a given trained model and feature values.

    Args:
        token: User access token.
        model_id: Trained model identifier.
        feature_values: Mapping from feature name to value.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/prediction/predict",
        method="POST",
        token=token,
        json={"model_id": model_id, "feature_values": feature_values}
    )


def get_user_predictions(token: str) -> dict[str, Any] | None:
    """
    Fetch prediction history for the current authenticated user.

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/prediction/user_predictions",
        method="GET",
        token=token,
    )


def get_all_users_predictions(token: str) -> dict[str, Any] | None:
    """
    Fetch prediction history for all users (admin/privileged endpoint).

    Args:
        token: User access token.

    Returns:
        API response dict (wrapper) or None on network failure.
    """

    return api_call(
        "/prediction/all_users_predictions",
        method="GET",
        token=token,
    )
