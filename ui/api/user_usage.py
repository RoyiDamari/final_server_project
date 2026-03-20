from typing import Any
from ui.api.base import api_call


def get_model_type_distribution(token: str) -> dict[str, Any] | None:
    """
    Fetch the distribution of trained model types for usage analytics.

    Calls:
        GET /usage/model_type_distribution

    Args:
        token: User access token.

    Returns:
        API response wrapper dict on success/error, or None on network failure.
    """

    return api_call(
        "/usage/model_type_distribution",
        method="GET",
        token=token,
    )


def get_regression_vs_classification_split(token: str) -> dict[str, Any] | None:
    """
    Fetch the regression vs classification split for usage analytics.

    Calls:
        GET /usage/type_split

    Args:
        token: User access token.

    Returns:
        API response wrapper dict on success/error, or None on network failure.
    """

    return api_call(
        "/usage/type_split",
        method="GET",
        token=token,
    )


def get_label_distribution(token: str) -> dict[str, Any] | None:
    """
    Fetch the distribution of labels used in training for usage analytics.

    Calls:
        POST /usage/label_distribution

    Notes:
        - Uses POST (as defined by your backend) even though this is a read-like operation.

    Args:
        token: User access token.

    Returns:
        API response wrapper dict on success/error, or None on network failure.
    """

    return api_call(
        "/usage/label_distribution",
        method="GET",
        token=token,
    )


def get_metric_distribution(token: str) -> dict[str, Any] | None:
    """
    Fetch the distribution of model metrics (e.g., accuracy/mae/r2) for usage analytics.

    Calls:
        POST /usage/metric_distribution

    Args:
        token: User access token.

    Returns:
        API response wrapper dict on success/error, or None on network failure.
    """

    return api_call(
        "/usage/metric_distribution",
        method="GET",
        token=token,
    )
