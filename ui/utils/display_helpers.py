import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil import tz


def init_sidebar_balance_slot():
    # create per-run slot; safe because we recreate each rerun
    st.session_state["_balance_slot"] = st.sidebar.empty()

def show_sidebar_balance():
    slot = st.session_state.get("_balance_slot")
    if slot is None:
        # fallback if someone forgot to init
        slot = st.sidebar.empty()
        st.session_state["_balance_slot"] = slot

    slot.metric("💰 Tokens", st.session_state.get("token_balance", 0))

def handle_usage_balance(resp: dict):
    charged = resp.get("charged")
    balance = resp.get("balance")
    if balance is None:
        return

    st.session_state["token_balance"] = balance
    show_sidebar_balance()

    if charged:
        st.success(f"💳 Tokens charged. New balance: {balance}")
    else:
        st.info(f"Remaining balance: {balance}")


def format_ts(ts: str) -> str:
    """
    Convert ISO UTC string to: YYYY-MM-DD HH:MM:SS
    Drops milliseconds and timezone.
    """
    try:
        utc_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        local_dt = utc_dt.astimezone(tz.tzlocal())
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def render_table(data: dict):
    if not data:
        return

    df = pd.DataFrame(
        [{"Field": k, "Value": v} for k, v in data.items()]
    )

    # 🔑 fix the real problem
    df["Value"] = df["Value"].astype(str)

    st.table(df)


def model_to_row(m: dict, include_user: bool) -> dict:
    metrics = m.get("metrics", {}) or {}

    return {
        "User ID": m.get("user_id") if include_user else None,
        "Model ID": m.get("id"),
        "Type": m.get("model_type"),
        "Label": m.get("label"),
        "Features": ", ".join(m.get("features", [])),
        "Created At": format_ts(m.get("created_at")),
        "CV Mean": metrics.get("cv_mean"),
        "CV Std": metrics.get("cv_std"),
    }


def prediction_to_row(p: dict, include_user: bool) -> dict:
    return {
        "User ID": p.get("user_id") if include_user else None,
        "Prediction ID": p.get("id"),
        "Model Type": p.get("model_type"),
        "Input Data": ", ".join(p.get("input_data", [])),
        "Prediction": p.get("prediction_result"),
        "Created At": format_ts(p.get("created_at")),
    }


def render_metrics_summary(metrics: dict):
    metrics = metrics or {}

    cols = st.columns(3)

    if "accuracy" in metrics:
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")

    if "r2" in metrics:
        cols[0].metric("R²", f"{metrics['r2']:.4f}")

    if "mae" in metrics:
        cols[1].metric("MAE", f"{metrics['mae']:.4f}")

    if "cv_mean" in metrics:
        cols[2].metric(
            "CV Score",
            f"{metrics['cv_mean']:.4f}",
            delta=f"± {metrics['cv_std']:.4f}" if "cv_std" in metrics else None,
        )


