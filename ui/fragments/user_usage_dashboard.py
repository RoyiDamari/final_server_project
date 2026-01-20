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


def render_model_type_distribution(token: str):
    st.subheader("Distribution by Model Type")

    if render_token_guarded_button("📊 Fetch Model Type Distribution", min_tokens=METADATA_COST):
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
        st.plotly_chart(fig)


def render_regression_vs_classification_split(token: str):
    st.subheader("Regression vs. Classification Split")

    if render_token_guarded_button("📈 Fetch Regression/Classification Split", min_tokens=METADATA_COST):
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
        st.plotly_chart(fig)


def render_label_distribution(token: str):
    st.subheader("🔎 Global Label Distribution")

    col1, col2 = st.columns(2)

    if render_token_guarded_button("🔍 Load Label Distribution", min_tokens=METADATA_COST):
        with st.spinner("Fetching label distribution..."):
            resp = get_label_distribution(token)
            handle_api_error(resp)
            handle_usage_balance(resp)

        data = resp.get("data", {})

        # ---------------- Classification ----------------
        with col1:
            st.markdown("### 🧠 Classification Labels")
            cls_data = data.get("classification", [])

            if cls_data:
                df = pd.DataFrame(cls_data)
                fig = px.bar(
                    df,
                    x="label",
                    y="count",
                    labels={"label": "Label", "count": "Models"},
                )

                fig.update_layout(
                    xaxis_tickangle=-30,
                    showlegend=False,
                )

                st.plotly_chart(fig)
            else:
                st.info("No classification models found.")

        # ---------------- Regression ----------------
        with col2:
            st.markdown("### 📈 Regression Labels")
            reg_data = data.get("regression", [])

            if reg_data:
                df = pd.DataFrame(reg_data)
                fig = px.bar(
                    df,
                    x="label",
                    y="count",
                    labels={"label": "Label", "count": "Models"},
                )

                fig.update_layout(
                    xaxis_tickangle=-30,
                    showlegend=False,
                )

                st.plotly_chart(fig)
            else:
                st.info("No regression models found.")


def render_metric_distribution(token: str):
    st.subheader("📊 Global Model Performance Distribution")

    col1, col2 = st.columns(2)

    if render_token_guarded_button("🔍 Load metric Distribution", min_tokens=METADATA_COST):
        with st.spinner("Fetching metric distribution..."):
            resp = get_metric_distribution(token)
            handle_api_error(resp)
            handle_usage_balance(resp)

        data = resp.get("data", {})

        # ---------- Classification ----------
        with col1:
            st.markdown("### 🎯 Accuracy Distribution")
            acc = data.get("classification", [])

            if acc:
                df = pd.DataFrame(acc)
                df = df.dropna(subset=["bucket", "count"])
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

                st.plotly_chart(fig)
            else:
                st.info("No classification models found.")

        # ---------- Regression ----------
        with col2:
            st.markdown("### 📉 R² Distribution")
            r2 = data.get("regression", [])

            if r2:
                df = pd.DataFrame(r2)
                df = df.dropna(subset=["bucket", "count"])
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

                st.plotly_chart(fig)
            else:
                st.info("No regression models found.")


def main():
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


