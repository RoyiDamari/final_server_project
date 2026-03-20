import io
from typing import Any
import pandas as pd
import streamlit as st
from ui.api.train_model import train_model, get_user_models, get_all_users_models
from ui.api.assist import explain
from ui.utils.params.params_ui import render_custom_params_ui, ask_chatgpt_button
from ui.utils.params.presets import MODEL_PRESETS, PARAM_HELP
from ui.utils.api_helpers import handle_api_error
from ui.utils.session_guard import ensure_authenticated
from ui.utils.widgets_guard import (
    has_enough_tokens,
    render_not_enough_tokens_warning,
    render_token_guarded_button,
)
from ui.utils.display_helpers import (
    handle_usage_balance,
    format_ts,
    render_table,
    model_to_row,
    render_metrics_summary,
)
from ui.config import TRAINING_COST, METADATA_COST, ASSIST_COST


@st.cache_data
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    """
    Read CSV bytes into a pandas DataFrame with Streamlit caching.

    Args:
        file_bytes: Raw CSV file content as bytes.

    Returns:
        Parsed pandas DataFrame.
    """

    return pd.read_csv(io.BytesIO(file_bytes))


def train_model_logic(
        token: str,
        uploaded_file: io.BytesIO,
        model_type: str,
        features: list[str],
        label: str,
        customized_params: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Call the backend train_model endpoint and handle the response.

    Behavior:
        - Resets the uploaded file pointer to the beginning before sending it.
        - Calls the backend training endpoint inside a Streamlit spinner.
        - Delegates response validation to handle_api_error().
        - Updates the token balance display through handle_usage_balance().
        - Returns the raw backend response dict.

    Args:
        token: Current user access token.
        uploaded_file: In-memory CSV file object to send to the backend.
        model_type: Selected model type (for example "linear", "logistic", or "random_forest").
        features: Selected feature column names.
        label: Selected target column name.
        customized_params: Final parameter dict to send to the backend.

    Returns:
        The backend response dict on success, or None if the lower-level API call
        returned None.
    """

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
    Render a free-text "Ask ChatGPT" box for model-level questions.

    Behavior:
        - Renders a text area for a user question and a submit button.
        - Validates that the question is non-empty and the user has enough tokens.
        - Calls the backend assist endpoint in free-question mode (context=question).
        - Uses handle_api_error(..., allow_in_progress=True) to treat "in progress"
          conflicts as non-fatal (returns early without stopping the page).
        - Updates the sidebar balance and shows the assistant answer if available.

    Args:
        token: Current user access token.
        model_type: Optional model-type context (e.g., "linear", "logistic").
            Passed to the assist endpoint to help tailor the answer. Maybe None.

    Returns:
        None.
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


def render_training_inputs(
    token: str,
) -> tuple[str, bytes | None, list[str], str, dict[str, Any], bool]:
    """
    Render the Train Model input UI and return the current user selections.

    This function is UI-only. It does not call the backend.

    It renders:
        - Model type selector (+ Ask ChatGPT button)
        - CSV uploader + preview
        - Feature multiselect and label select box inside a form
        - Preset selector (+ optional parameter customization UI)

    Args:
        token: User access token. Used only for Ask ChatGPT helper controls.

    Returns:
        A tuple:
            model_type: The selected model type ("linear", "random_forest", "logistic").
            file_bytes: Uploaded CSV bytes (stored in session_state) or None if not uploaded.
            features: Selected feature column names (empty if not selected / no CSV).
            label: Selected label column name ("" if not selected / no CSV).
            customized_params: Final parameter dict (maybe empty).
            submitted: True if the Train Model form was submitted, else False.

    Raises:
        None directly. Any Streamlit rendering/IO errors will surface naturally.
    """
    customized_params: dict[str, Any] = {}

    model_type = st.selectbox(
        "Model Type",
        ["linear", "random_forest", "logistic"],
        key="model_type_select",
        on_change=invalidate_model_context,
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
    cols: list[str] = []

    if file_bytes:
        try:
            df = read_csv_cached(file_bytes)
            st.write("### Preview")
            st.dataframe(df.head())
            cols = list(df.columns)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return model_type, file_bytes, [], "", {}, False

    with st.form("train_form", clear_on_submit=False):
        features: list[str] = []
        label: str = ""

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
    )

    is_default = preset_name.lower().startswith("default")

    if not is_default:
        st.markdown(f"{preset_name}", help=PARAM_HELP[preset_name])

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
                    key=f"rf_task_{model_type}",
                )
                customized_params["task"] = "auto" if rf_task.startswith("Auto") else rf_task

            if model_type == "linear" and "kind" in chosen_params:
                customized_params["kind"] = chosen_params["kind"]

            if model_type == "logistic" and "penalty" in chosen_params:
                customized_params["penalty"] = chosen_params["penalty"]

    return model_type, file_bytes, features, label, customized_params, submitted


def render_training_form(token: str) -> None:
    """
    Render the full Train Model page: inputs, training submission, results, and the free-question box.

    Behavior:
        - Calls render_training_inputs() to draw the UI and collect inputs.
        - If the form was submitted:
            * validates token balance and required fields
            * calls train_model_logic() to execute training
            * stores the last trained model in session_state on success
        - If a model was trained successfully, renders its summary, params, and metrics.
        - Always renders the free-question ("Ask ChatGPT") box at the bottom.

    Args:
        token: User access token.

    Returns:
        None.

    Raises:
        None directly.
        Note: train_model_logic() calls handle_api_error(), which may stop execution (st.stop)
        or reroute to log in (st.rerun via logout flow) on auth failures.
    """
    st.header("📈 Train a Model")

    model_type, file_bytes, features, label, customized_params, submitted = render_training_inputs(token)

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
                    render_metrics_summary(metrics)

    render_free_question_box(
        token=token,
        model_type=st.session_state.get("last_trained_model", {}).get("model_type", model_type),
    )


def render_model_viewer(token: str) -> None:
    """
    Render the trained models viewer with an optional max-rows display limit.

    Stores fetched models in session_state so the table persists across reruns.

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.markdown("---")
    st.header("🧠 Trained Models Viewer")

    st.session_state.setdefault("models_view_rows", None)
    st.session_state.setdefault("models_view_want_all", False)

    view_choice = st.radio(
        "Choose which models to view:",
        ["My Models", "All Users' Models"],
        key="model_viewer_choice",
        on_change=clear_models_viewer_cache,
    )
    want_all = view_choice == "All Users' Models"

    clicked = (
        render_token_guarded_button("🔄 Fetch Models", min_tokens=METADATA_COST)
        if want_all
        else st.button("🔄 Fetch Models")
    )

    if clicked:
        with st.spinner("Loading models..."):
            resp = get_all_users_models(token) if want_all else get_user_models(token)

        handle_api_error(resp)

        if want_all:
            handle_usage_balance(resp)

        models = resp.get("data", [])
        if not models:
            st.info("No models have been trained yet.")
            st.session_state["models_view_rows"] = None
            return

        st.session_state["models_view_rows"] = models
        st.session_state["models_view_want_all"] = want_all

    models = st.session_state.get("models_view_rows")
    if not models:
        st.info("Click the button to load models.")
        return

    include_user = bool(st.session_state.get("models_view_want_all", False))
    rows = [model_to_row(m, include_user=include_user) for m in models]
    df = pd.DataFrame(rows).astype(str)

    if not include_user and "User ID" in df.columns:
        df = df.drop(columns=["User ID"])

    count = len(df)
    if count == 1:
        st.dataframe(df, hide_index=True, use_container_width=True)
        return

    max_rows = st.slider(
        "Max models to show",
        min_value=1,
        max_value=count,
        value=min(20, count),
        key="models_viewer_max_rows",
    )
    st.dataframe(df.iloc[:max_rows], hide_index=True, use_container_width=True)


def invalidate_model_context() -> None:
    """
    Clear training-result and free-question state from Streamlit session state.

    Behavior:
        - Removes the last trained model and its success flag.
        - Clears the free-text assist question input and stored answer.

    This is intended to be used when the model context changes, such as when
    switching model type.

    Returns:
        None.
    """

    # Model results
    st.session_state.pop("last_trained_model", None)
    st.session_state.pop("show_train_success", None)

    # Free-text chat
    st.session_state["assist_free_question"] = ""
    st.session_state.pop("assist_free_answer", None)


def clear_models_viewer_cache() -> None:
    st.session_state["models_view_rows"] = None
    st.session_state["models_view_want_all"] = False
    st.session_state.pop("models_viewer_max_rows", None)


def main() -> None:
    """
    Fragment entry point for the Train Model page.

    Behavior:
        - Ensures the user is authenticated.
        - Retrieves the current access token from session_state.
        - Renders the training form (upload/data/params + train action).
        - Renders the trained models viewer section under the form.

    Returns:
        None.
    """

    ensure_authenticated()
    token = st.session_state["jwt_token"]

    render_training_form(token)
    st.divider()
    render_model_viewer(token)
