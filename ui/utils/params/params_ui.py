import streamlit as st
from ui.utils.params.linear import render_linear_params_ui
from ui.utils.params.logistic import render_logistic_params_ui
from ui.utils.params.random_forest import  render_rf_params_ui
from ui.utils.api_helpers import handle_api_error
from ui.utils.display_helpers import handle_usage_balance
from ui.utils.widgets_guard import has_enough_tokens, render_not_enough_tokens_warning
from ui.config import ASSIST_COST


def ask_chatgpt_button(
    label: str,
    model_type: str,
    token: str,
    explain_fn,
    param_key: str | None,
):
    """
    Generic ChatGPT helper:
    - label controls UI wording
    - param_key controls backend mode
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
                with st.expander("📘 Explanation", expanded=True):
                    st.markdown(text)
            else:
                st.error(f"Unexpected empty response: {resp}")

def render_custom_params_ui(model_type: str, base_params: dict, token: str, explain_fn):
    if model_type == "linear":
        return render_linear_params_ui(model_type, base_params, token, explain_fn, ask_chatgpt_button)
    if model_type == "logistic":
        return render_logistic_params_ui(model_type, base_params, token, explain_fn, ask_chatgpt_button)
    if model_type == "random_forest":
        return render_rf_params_ui(model_type, base_params, token, explain_fn, ask_chatgpt_button)
    return dict(base_params or {})