import pandas as pd
import streamlit as st
import io
from ui.api.train_model import train_model, get_user_models, get_all_users_models
from ui.api.assist import explain
from ui.utils.params.params_ui import render_custom_params_ui, ask_chatgpt_button
from ui.utils.params.presets import MODEL_PRESETS, PARAM_HELP
from ui.utils.api_helpers import handle_api_error
from ui.utils.session_guard import ensure_authenticated
from ui.utils.widgets_guard import has_enough_tokens, render_not_enough_tokens_warning, render_token_guarded_button
from ui.utils.display_helpers import (handle_usage_balance, format_ts, render_table,
                                      model_to_row, render_metrics_summary)
from ui.config import TRAINING_COST, METADATA_COST, ASSIST_COST


@st.cache_data
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def train_model_logic(token, uploaded_file, model_type, features, label, customized_params):
    """Extracted training logic to keep main() clean."""
    try:
        uploaded_file.seek(0)
    except (AttributeError, IOError) as e:
        st.error(f"Could not reset file pointer: {e}")

    with st.spinner("Training your model..."):
        resp = train_model(
            token=token,
            file=uploaded_file,
            model_type=model_type,
            features=features,
            label=label,
            model_params=customized_params,
        )

    handle_api_error(resp)
    handle_usage_balance(resp)

    return resp


def render_free_question_box(token: str, model_type: dict | None):
    """
    One model-level question box (NOT tied to param_key buttons).
    """
    st.markdown("---")
    st.subheader("💬 Ask about your model or results")

    question = st.text_area(
        "Write your question (metrics, performance, next steps, etc.)",
        placeholder="Example: Why is my MAE high? What should I tune next?",
        height=90,
        key="assist_free_question",
        value=st.session_state.get("assist_free_question", ""),
    )

    if st.button("Ask ChatGPT", key="assist_free_ask"):
        if not question.strip():
            st.warning("Please write a question.")
            return

        if not has_enough_tokens(ASSIST_COST):
            render_not_enough_tokens_warning(ASSIST_COST)
            return

        with st.spinner("Contacting ChatGPT…"):
            resp = explain(
                token=token,
                model_type=model_type,
                param_key=None,
                question=question,
            )

        handle_api_error(resp)
        handle_usage_balance(resp)

        if resp is not None:
            st.session_state["assist_free_answer"] = resp.get("data")

    if "assist_free_answer" in st.session_state:
        st.info(st.session_state["assist_free_answer"])


def render_training_form(token: str):
    st.header("📈 Train a Model")

    customized_params: dict = {}

    model_type = st.selectbox(
        "Model Type",
        ["linear", "random_forest", "logistic"],
        key="model_type_select",
        on_change=invalidate_model_context
    )

    st.markdown(f"{model_type}", help=PARAM_HELP[model_type])

    ask_chatgpt_button(
        label=model_type,
        model_type=model_type,
        token=token,
        explain_fn=explain,
        param_key=None,
    )

    uploaded_file = st.file_uploader("📂 Upload CSV Data", type=["csv"])

    if uploaded_file is not None:
        st.session_state["train_file_bytes"] = uploaded_file.getvalue()
    else:
        st.session_state.pop("train_file_bytes", None)

    file_bytes = st.session_state.get("train_file_bytes")
    df = None
    cols: list = []

    if file_bytes:
        try:
            df = read_csv_cached(file_bytes)
            st.write("### Preview")
            st.dataframe(df.head())
            cols = list(df.columns)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

    with st.form("train_form", clear_on_submit=False):
        features: list = []
        label = ""

        if df is not None:
            features = st.multiselect(
                "Select features",
                options=cols,
                key=f"features_{model_type}",
            )

            label = st.selectbox(
                "Select label",
                options=cols,
                index=None,
                placeholder="Choose option",
                key=f"label_{model_type}",
            )

        submitted = st.form_submit_button("Train Model")

    preset_names = list(MODEL_PRESETS[model_type].keys())
    preset_name = st.selectbox(
        "Parameter preset",
        options=preset_names,
        key=f"preset_{model_type}",
        on_change=invalidate_model_context
    )

    is_default = preset_name.lower().startswith("default")

    if not is_default:
        st.markdown(
            f"{preset_name}",
            help=PARAM_HELP[preset_name]
        )

        ask_chatgpt_button(
            label=preset_name,
            model_type=model_type,
            token=token,
            explain_fn=explain,
            param_key=preset_name,
        )

    chosen_params = dict(MODEL_PRESETS[model_type][preset_name])

    if not is_default:
        with st.expander("Customize parameters (optional)"):
            customized_params = render_custom_params_ui(
                model_type,
                chosen_params,
                token=token,
                explain_fn=explain,
            )
            customized_params = dict(customized_params or {})

            if model_type == "random_forest":
                rf_task = st.selectbox(
                    "Random Forest task",
                    ["Auto (detect)", "classification", "regression"],
                    key=f"rf_task_{model_type}"
                )
                customized_params["task"] = (
                    "auto" if rf_task.startswith("Auto") else rf_task
                )

            if model_type == "linear" and "kind" in chosen_params:
                customized_params["kind"] = chosen_params["kind"]

            if model_type == "logistic" and "penalty" in chosen_params:
                customized_params["penalty"] = chosen_params["penalty"]

    if submitted:
        if not has_enough_tokens(TRAINING_COST):
            render_not_enough_tokens_warning(TRAINING_COST)
            return

        if file_bytes is None or len(file_bytes) == 0:
            st.warning("Please upload a CSV file.")
            return

        if not features:
            st.warning("Please select at least one feature.")
            return

        if not label:
            st.warning("Please select a label.")
            return

        resp = train_model_logic(
            token=token,
            uploaded_file=io.BytesIO(file_bytes),
            model_type=model_type,
            features=features,
            label=label,
            customized_params=customized_params,
        )

        if resp is not None:
            st.session_state["last_trained_model"] = resp.get("data")
            st.session_state["show_train_success"] = True

    if st.session_state.get("show_train_success"):
        model = st.session_state.get("last_trained_model")

        if model:
            st.success(
                f"✅ Model Information!\n\n"
                f"**Type:** {model['model_type']}\n\n"
                f"**Label:** {model['label']}\n\n"
                f"**Features:** {', '.join(model['features'])}\n\n"
                f"**Status:** {model['status']}\n\n"
                f"**Created at:** {format_ts(model['created_at'])}"
            )

            params = model.get("model_params", {})
            if params:
                with st.expander("⚙️ Model Parameters"):
                    render_table(params)

            metrics = model.get("metrics", {})
            if metrics:
                with st.expander("📊 Training Metrics"):
                    render_metrics_summary(model.get("metrics", {}))

    render_free_question_box(
        token=token,
        model_type=st.session_state.get(
            "last_trained_model", {}
        ).get("model_type", model_type),
    )


def render_model_viewer(token: str):
    st.markdown("---")
    st.header("🧠 Trained Models Viewer")

    view_choice = st.radio(
        "Choose which models to view:",
        ["My Models", "All Users' Models"],
        key="model_viewer_choice"
    )

    want_all = view_choice == "All Users' Models"

    button_clicked = (
        render_token_guarded_button("🔄 Fetch Models", min_tokens=METADATA_COST)
        if want_all
        else st.button("🔄 Fetch Models")
    )

    if button_clicked:
        with st.spinner("Loading models..."):
            resp = (
                get_user_models(token)
                if not want_all
                else get_all_users_models(token)
            )

        handle_api_error(resp)

        if not want_all:
            st.info(f"Remaining balance: {st.session_state.get('token_balance', 0)}")
        else:
            handle_usage_balance(resp)

        models = resp.get("data", [])

        if models is None:
            st.info("Click the button to load models.")
            return

        if not models:
            st.info("No models have been trained yet.")
            return

        rows = [
            model_to_row(m, include_user=want_all)
            for m in models
        ]

        df = pd.DataFrame(rows)

        df = df.astype(str)

        if not want_all:
            df = df.drop(columns=["User ID"])

        st.dataframe(
            df,
            hide_index=True,
        )


def invalidate_model_context():
    # Model results
    st.session_state.pop("last_trained_model", None)
    st.session_state.pop("show_train_success", None)

    # Free-text chat
    st.session_state["assist_free_question"] = ""
    st.session_state.pop("assist_free_answer", None)


def main():
    ensure_authenticated()

    token = st.session_state["jwt_token"]

    render_training_form(token)
    st.divider()
    render_model_viewer(token)
