import json
import pandas as pd
from typing import Any
import streamlit as st
from ui.api.train_model import get_user_models_internal
from ui.api.prediction import predict, get_user_predictions, get_all_users_predictions
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error
from ui.utils.widgets_guard import has_enough_tokens, render_not_enough_tokens_warning, render_token_guarded_button
from ui.utils.display_helpers import handle_usage_balance, format_ts, render_table, prediction_to_row
from ui.config import PREDICTION_COST, METADATA_COST


def _ensure_features_list(m: dict) -> list[str]:
    feats = m.get("features", [])
    if isinstance(feats, str):
        try:
            feats = json.loads(feats)
        except (json.JSONDecodeError, TypeError):
            feats = []
    return feats


def _pretty_label(m: dict) -> str:
    m_id = m.get("id", "Unnamed")
    name = m.get("model_type", "Unnamed")
    ts = format_ts(m.get("created_at"))
    return f"{m_id}    •    {name}_model    •    {ts}"


def render_prediction_form(token: str):
    st.header("🔮 Make a Prediction")

    resp = get_user_models_internal(token)

    handle_api_error(resp)

    my_models = resp["data"]

    if not my_models:
        st.warning("No trained models found. Train a model first!")
        return

    choices = [(m["id"], _pretty_label(m)) for m in my_models]
    labels = [label for _, label in choices]
    ids = [model_id for model_id, _ in choices]

    chosen_idx = st.selectbox("Select your model", options=range(len(labels)), format_func=lambda i: labels[i])
    chosen_model_id = ids[chosen_idx]
    chosen_model = next(m for m in my_models if m["id"] == chosen_model_id)

    st.write(f"**Model Type:** {chosen_model.get('model_type', '—')}")
    st.write(f"**Label:** {chosen_model.get('label', '—')}")
    st.caption(f"Trained at: {format_ts(chosen_model.get('created_at'))}")

    metrics = chosen_model.get("metrics", {}) or {}

    # ---- Task-aware metrics ----
    if "accuracy" in metrics:
        cols = st.columns(2)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")

        if "cv_mean" in metrics:
            cols[1].metric("CV Mean", f"{metrics['cv_mean']:.4f}")

    elif "r2" in metrics:
        cols = st.columns(2)
        cols[0].metric("R²", f"{metrics['r2']:.4f}")

        if "mae" in metrics:
            cols[1].metric("MAE", f"{metrics['mae']:.4f}")

    features = _ensure_features_list(chosen_model)
    feature_schema = chosen_model.get("feature_schema", {})

    if not features:
        st.error("This model has no features metadata.")
        return

    with st.form("predict_form"):
        st.subheader("Input feature values")

        feature_values: dict[str, Any] = {}

        for feat in features:
            f_type = feature_schema.get(feat, "numeric")
            label = f"{feat} ({f_type})"

            if f_type == "numeric":
                feature_values[feat] = st.number_input(
                    label,
                    key=f"prediction_{chosen_model_id}_{feat}",
                    value=0.0
                )

            elif f_type == "categorical":
                feature_values[feat] = st.text_input(
                    label,
                    key=f"prediction_{chosen_model_id}_{feat}"
                )

            else:
                st.warning(f"Unknown feature type for '{feat}', defaulting to text")
                feature_values[feat] = st.text_input(
                    label,
                    key=f"prediction_{chosen_model_id}_{feat}"
                )

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:

        if not has_enough_tokens(PREDICTION_COST):
            render_not_enough_tokens_warning(PREDICTION_COST)
            return

        with st.spinner("Calculating prediction..."):
            resp = predict(token, model_id=chosen_model_id, feature_values=feature_values)

        handle_api_error(resp)

        handle_usage_balance(resp)

        prediction = resp["data"]

        st.session_state["last_prediction"] = prediction
        st.session_state["show_predict_success"] = True

    if st.session_state.get("show_predict_success"):
        prediction = st.session_state.get("last_prediction")

        if prediction:
            st.success(
                f"✅ Prediction Information!\n\n"
                f"Type:** {prediction['model_type']}\n\n"
                f"**Prediction Results:** {prediction['prediction_result']}\n\n"
                f"**Status:** {prediction['status']}\n\n"
                f"Created at: {format_ts(prediction['created_at'])}")

            with st.expander("📥 Input Data"):
                render_table(prediction["feature_values"])

        st.session_state.pop("show_predict_success", None)


def render_prediction_viewer(token: str) -> None:
    """
    Render prediction history with an optional max-rows display limit.

    Behavior:
        - Lets the user choose between "My Predictions" and "All Users' Predictions".
        - Fetches predictions on button click and stores them in session state.
        - Shows a slider to limit displayed rows.
        - For all-users view, uses token-guarded button and shows token charge/balance updates.

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.markdown("---")
    st.header("📜 Predictions History Viewer")

    st.session_state.setdefault("predictions_view_rows", None)
    st.session_state.setdefault("predictions_view_want_all", False)

    view_choice = st.radio(
        "Choose which predictions to view:",
        ["My Predictions", "All Users' Predictions"],
        key="prediction_viewer_choice",
        on_change=clear_predictions_viewer_cache,
    )
    want_all = view_choice == "All Users' Predictions"

    button_clicked = (
        render_token_guarded_button("🔄 Fetch Predictions", min_tokens=METADATA_COST)
        if want_all
        else st.button("🔄 Fetch Predictions")
    )

    if button_clicked:
        with st.spinner("Loading predictions..."):
            resp = get_all_users_predictions(token) if want_all else get_user_predictions(token)

        handle_api_error(resp)

        if want_all:
            handle_usage_balance(resp)

        predictions = resp.get("data", [])
        if not predictions:
            st.info("No predictions have been made yet.")
            st.session_state["predictions_view_rows"] = None
            return

        st.session_state["predictions_view_rows"] = predictions
        st.session_state["predictions_view_want_all"] = want_all

    predictions = st.session_state.get("predictions_view_rows")
    if predictions is None:
        st.info("Click the button above to load predictions.")
        return
    if not predictions:
        st.info("No predictions have been made yet.")
        return

    include_user = bool(st.session_state.get("predictions_view_want_all", False))
    rows = [prediction_to_row(p, include_user=include_user) for p in predictions]
    df = pd.DataFrame(rows).astype(str)

    if not include_user and "User ID" in df.columns:
        df = df.drop(columns=["User ID"])

    count = len(df)
    if count == 1:
        st.dataframe(df, hide_index=True, use_container_width=True)
        return

    max_rows = st.slider(
        "Max predictions to show",
        min_value=1,
        max_value=count,
        value=min(20, count),
        key="predictions_viewer_max_rows",
    )
    st.dataframe(df.iloc[:max_rows], hide_index=True, use_container_width=True)


def clear_predictions_viewer_cache() -> None:
    st.session_state["predictions_view_rows"] = None
    st.session_state["predictions_view_want_all"] = False
    st.session_state.pop("predictions_viewer_max_rows", None)


def main() -> None:
    """
    Fragment entry point for the Prediction page.

    Behavior:
        - Ensures the user is authenticated.
        - Retrieves the current access token from session_state.
        - Renders the prediction UI (select model, input features, submit prediction).
        - Renders prediction history viewer section (user / all users depending on role).

    Returns:
        None.
    """

    ensure_authenticated()

    token = st.session_state["jwt_token"]

    render_prediction_form(token)
    st.divider()
    render_prediction_viewer(token)
