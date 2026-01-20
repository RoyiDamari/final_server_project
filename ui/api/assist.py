from ui.api.base import api_call


def explain(
    token: str,
    model_type: str | None,
    param_key: str | None,
    question: str | None = None,
):
    """
    Single backend endpoint (/assist/explain) with two modes:

    1) Parameter mode:
        - model_type required
        - param_key required
        - question must be None

    2) Free-text question mode:
        - question required
        - model_type optional (helps answer)
        - param_key must be None
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
