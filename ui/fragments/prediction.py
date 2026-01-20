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
    st.write(f"**Label:** {chosen_model.get('label','—')}")
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
                render_table(prediction["input_data"])

        st.session_state.pop("show_predict_success", None)


def render_prediction_viewer(token: str):
    st.markdown("---")
    st.header("📜 Predictions History Viewer")

    # Ensure session state exists
    st.session_state.setdefault("predictions_history", None)

    # Radio selection identical to model viewer
    view_choice = st.radio(
        "Choose which predictions to view:",
        ["My Predictions", "All Users' Predictions"],
        key="prediction_viewer_choice"
    )

    want_all = view_choice == "All Users' Predictions"

    if want_all:
        button_clicked = render_token_guarded_button(
            "🔄 Fetch Predictions",
            min_tokens=METADATA_COST
        )
    else:
        button_clicked = st.button("🔄 Fetch Predictions")

    if button_clicked:
        with st.spinner("Loading predictions..."):
            resp = (
                get_user_predictions(token)
                if view_choice == "My Predictions"
                else get_all_users_predictions(token)
            )

        handle_api_error(resp)

        if not want_all:
            st.info(f"Remaining balance: {st.session_state.get('token_balance', 0)}")
        else:
            handle_usage_balance(resp)

        predictions = resp.get("data", [])

        if predictions is None:
            st.info("Click the button above to load predictions.")
            return

        if not predictions:
            st.info("No predictions have been made yet.")
            return

        rows = [
            prediction_to_row(p, include_user=want_all)
            for p in predictions
        ]

        df = pd.DataFrame(rows)

        df = df.astype(str)

        if not want_all:
            df = df.drop(columns=["User ID"])

        st.dataframe(
            df,
            hide_index=True,
        )


def main():
    ensure_authenticated()

    token = st.session_state["jwt_token"]

    render_prediction_form(token)
    st.divider()
    render_prediction_viewer(token)



