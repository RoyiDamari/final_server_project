import streamlit as st
from typing import Callable, Any
from ui.utils.params.linear import render_linear_params_ui
from ui.utils.params.logistic import render_logistic_params_ui
from ui.utils.params.random_forest import render_rf_params_ui
from ui.utils.api_helpers import handle_api_error
from ui.utils.display_helpers import handle_usage_balance
from ui.utils.widgets_guard import has_enough_tokens, render_not_enough_tokens_warning
from ui.config import ASSIST_COST


def ask_chatgpt_button(
        label: str,
        model_type: str,
        token: str,
        explain_fn: Callable[..., dict[str, Any]],
        param_key: str | None,
) -> None:
    """
    Render a popover button that calls the assist explanation endpoint.

    Behavior:
        - Opens a popover for the given label.
        - Checks whether the user has enough tokens for the assist action.
        - Renders a button whose text depends on whether the request is for the
            model itself or for a preset/parameter.
        - On click, calls the provided explain function, handles API errors,
            updates the token balance display, and renders the returned explanation.

    Args:
        label: UI label for the current model / preset / parameter.
        model_type: Selected model type context (for example "linear", "logistic", or "random_forest").
        token: Current user access token.
        explain_fn: Callable used to call the backend explanation endpoint.
            It must accept token, model_type, param_key, and question keyword arguments
            and return a response dict.
        param_key: Parameter or preset key to explain. If None, the request is treated
            as a model-level explanation.

    Returns:
        None
    """

    with st.popover(f"More about `{label}`"):
        st.caption("Ask ChatGPT (costs tokens)")

        if not has_enough_tokens(ASSIST_COST):
            render_not_enough_tokens_warning(ASSIST_COST)
            return

        button_text = (
            f"Explain {label} model"
            if label == model_type
            else f"Explain {label} for {model_type}"
        )

        if st.button(button_text, key=f"ask_{model_type}_{label}"):
            with st.spinner("Contacting ChatGPT…"):
                resp = explain_fn(
                    token=token,
                    model_type=model_type,
                    param_key=param_key,
                    question=None,
                )

            handle_api_error(resp)
            handle_usage_balance(resp)

            text = resp.get("data")
            if text:
                st.markdown("**📘 Explanation**")
                st.markdown(text)
            else:
                st.error(f"Unexpected empty response: {resp}")


def render_custom_params_ui(
        model_type: str,
        base_params: dict[str, Any],
        token: str,
        explain_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """
    Render the model-specific parameter customization UI and return updated parameters.

    Dispatches to the correct renderer based on model_type:
        - "linear" -> render_linear_params_ui(...)
        - "logistic" -> render_logistic_params_ui(...)
        - "random_forest" -> render_rf_params_ui(...)

    Each renderer:
        - starts from base_params (preset)
        - renders Streamlit widgets to modify parameters
        - attaches "Ask ChatGPT" helper controls via ask_chatgpt_button

    Args:
        model_type: Selected model type ("linear", "logistic", "random_forest").
        base_params: Preset/base parameters dict (may be empty).
        token: User access token for the explanation endpoint.
        explain_fn: Callable used by ask_chatgpt_button to call the backend explain endpoint.

    Returns:
        A dict of model parameters updated according to UI selections.
    """

    if model_type == "linear":
        return render_linear_params_ui(model_type, base_params, token, explain_fn, ask_chatgpt_button)
    if model_type == "logistic":
        return render_logistic_params_ui(model_type, base_params, token, explain_fn, ask_chatgpt_button)
    if model_type == "random_forest":
        return render_rf_params_ui(model_type, base_params, token, explain_fn, ask_chatgpt_button)
    return dict(base_params or {})
