import streamlit as st
import plotly.express as px
import pandas as pd
from ui.api.user_usage import (
    get_model_type_distribution,
    get_regression_vs_classification_split,
    get_label_distribution,
    get_metric_distribution,
)
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error
from ui.utils.widgets_guard import render_token_guarded_button
from ui.utils.display_helpers import handle_usage_balance
from ui.config import METADATA_COST


def render_model_type_distribution(token: str) -> None:
    """
    Render a bar chart of model counts grouped by model type.

    Fetches data from the backend only when the user clicks the guarded button.
    The endpoint is token-charged (METADATA_COST), so we use render_token_guarded_button()
    and then update the sidebar/message via handle_usage_balance().

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.subheader("Distribution by Model Type")

    clicked = render_token_guarded_button(
        "📊 Fetch Model Type Distribution",
        min_tokens=METADATA_COST
    )

    if not clicked:
        st.info("Click the button above to load model type distribution.")
        return

    with st.spinner("Loading model type distribution..."):
        resp = get_model_type_distribution(token)
        handle_api_error(resp)

    handle_usage_balance(resp)

    data = resp.get("data")
    if not data:
        st.warning("No model type data found.")
        return

    df = pd.DataFrame(data)
    fig = px.bar(df, x="model_type", y="count", color="model_type")
    st.plotly_chart(fig, use_container_width=True)


def render_regression_vs_classification_split(token: str) -> None:
    """
    Render a pie chart showing regression vs classification model counts.

    Fetches data only when the user clicks the guarded button and updates balance
    via handle_usage_balance().

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.subheader("Regression vs. Classification Split")

    clicked = render_token_guarded_button(
        "📈 Fetch Regression/Classification Split",
        min_tokens=METADATA_COST
    )

    if not clicked:
        st.info("Click the button above to load the regression/classification split.")
        return

    with st.spinner("Loading split..."):
        resp = get_regression_vs_classification_split(token)
        handle_api_error(resp)

    handle_usage_balance(resp)

    data = resp.get("data")
    if not data:
        st.warning("No split data found.")
        return

    df = pd.DataFrame(data)
    fig = px.pie(df, names="problem_type", values="count")
    st.plotly_chart(fig, use_container_width=True)


def render_label_distribution(token: str) -> None:
    """
    Render global label distributions for classification and regression models.

    The backend returns a dict with keys like "classification" and "regression",
    each containing (label, count) rows. The function plots separate bar charts.

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.subheader("🔎 Global Label Distribution")

    clicked = render_token_guarded_button(
        "🔍 Fetch Label Distribution",
        min_tokens=METADATA_COST
    )

    if not clicked:
        st.info("Click the button above to load label distribution.")
        return

    with st.spinner("Fetching label distribution..."):
        resp = get_label_distribution(token)
        handle_api_error(resp)

    handle_usage_balance(resp)

    data = resp.get("data", {}) or {}

    col1, col2 = st.columns(2)

    # ---------------- Classification ----------------
    with col1:
        st.markdown("### 🧠 Classification Labels")
        cls_data = data.get("classification", []) or []

        if cls_data:
            df = pd.DataFrame(cls_data)
            fig = px.bar(
                df,
                x="label",
                y="count",
                labels={"label": "Label", "count": "Models"},
            )
            fig.update_layout(xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No classification models found.")

    # ---------------- Regression ----------------
    with col2:
        st.markdown("### 📈 Regression Labels")
        reg_data = data.get("regression", []) or []

        if reg_data:
            df = pd.DataFrame(reg_data)
            fig = px.bar(
                df,
                x="label",
                y="count",
                labels={"label": "Label", "count": "Models"},
            )
            fig.update_layout(xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regression models found.")


def render_metric_distribution(token: str) -> None:
    """
    Render global performance distributions for classification (accuracy) and regression (R²).

    The backend returns bucketed distributions (bucket, count) for each problem type.
    This function plots bar charts for accuracy and R² in two columns.

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.subheader("📊 Global Model Performance Distribution")

    clicked = render_token_guarded_button(
        "🔍 Fetch Metric Distribution",
        min_tokens=METADATA_COST
    )

    if not clicked:
        st.info("Click the button above to load metric distribution.")
        return

    with st.spinner("Fetching metric distribution..."):
        resp = get_metric_distribution(token)
        handle_api_error(resp)

    handle_usage_balance(resp)

    data = resp.get("data", {}) or {}

    col1, col2 = st.columns(2)

    # ---------- Classification ----------
    with col1:
        st.markdown("### 🎯 Accuracy Distribution")
        acc = data.get("classification", []) or []

        if acc:
            df = pd.DataFrame(acc).dropna(subset=["bucket", "count"])
            df["bucket"] = df["bucket"].astype(float)
            df["count"] = df["count"].astype(int)

            fig = px.bar(
                df,
                x="bucket",
                y="count",
                labels={"bucket": "Accuracy", "count": "Models"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis=dict(dtick=0.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No classification models found.")

    # ---------- Regression ----------
    with col2:
        st.markdown("### 📉 R² Distribution")
        r2 = data.get("regression", []) or []

        if r2:
            df = pd.DataFrame(r2).dropna(subset=["bucket", "count"])
            df["bucket"] = df["bucket"].astype(float)
            df["count"] = df["count"].astype(int)

            fig = px.bar(
                df,
                x="bucket",
                y="count",
                labels={"bucket": "R²", "count": "Models"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis=dict(dtick=0.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regression models found.")


def main() -> None:
    """
    Fragment entry point for the All-Users activity dashboard.

    Ensures authentication, then renders four analytics sections:
        - model type distribution
        - regression vs classification split
        - label distribution
        - metric distribution

    Returns:
        None.
    """

    ensure_authenticated()

    token = st.session_state["jwt_token"]

    st.header("📊 User Activity Dashboard (All Users)")

    render_model_type_distribution(token)
    st.divider()

    render_regression_vs_classification_split(token)
    st.divider()

    render_label_distribution(token)
    st.divider()

    render_metric_distribution(token)
