from typing import Any
from ui.api.base import api_call


def explain(
        token: str,
        model_type: str | None,
        param_key: str | None,
        question: str | None = None,
) -> dict[str, Any] | None:
    """
    Call the backend /assist/explain endpoint.

    This endpoint supports two modes:

    1) Parameter explanation mode:
        - model_type: required
        - param_key: required
        - question: must be None
        Result: explanation text for a specific model parameter (or model overview if param_key=None).

    2) Free-text question mode:
        - question: required (sent as "context")
        - model_type: optional (helps the assistant tailor the answer)
        - param_key: must be None
        Result: assistant answer based on the user's question.

    Args:
        token: User access token.
        model_type: Model type context ("linear"/"logistic"/"random_forest") or None.
        param_key: Parameter key name or None.
        question: Free-text question (optional). If provided, sent under "context".

    Returns:
        API response dict (wrapper) on success/error, or None on network failure.
    """

    payload: dict = {}

    if model_type:
        payload["model_type"] = model_type

    if param_key:
        payload["param_key"] = param_key

    if question:
        payload["context"] = question

    return api_call(
        "/assist/explain",
        method="POST",
        token=token,
        json=payload,
    )
