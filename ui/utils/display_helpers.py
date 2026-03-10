import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil import tz
from typing import Any


def init_sidebar_balance_slot() -> None:
    """
    Initialize a placeholder slot in the Streamlit sidebar for the token balance metric.

    This should be called once per rerun before any code tries to render the balance,
    so that subsequent updates can reuse the same placeholder.

    Returns:
        None.
    """

    st.session_state["_balance_slot"] = st.sidebar.empty()


def show_sidebar_balance() -> None:
    """
    Render (or re-render) the current token balance in the sidebar placeholder.

    If the placeholder was not initialized for some reason, this function will
    create a fallback placeholder and store it in session_state.

    Returns:
        None.
    """

    slot = st.session_state.get("_balance_slot")
    if slot is None:
        slot = st.sidebar.empty()
        st.session_state["_balance_slot"] = slot

    slot.metric("💰 Tokens", st.session_state.get("token_balance", 0))


def handle_usage_balance(resp: dict[str, Any]) -> None:
    """
    Update the user's token balance from an API response and show a charge/remaining message.

    Expected response keys:
        - "balance": int (required to update session_state)
        - "charged": bool (optional; determines message type)

    Side effects:
        - Updates st.session_state["token_balance"]
        - Refreshes the sidebar balance via show_sidebar_balance()
        - Displays a message in the main page using st.success/st.info

    Args:
        resp: API response dict.

    Returns:
        None.
    """

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
    Convert an ISO UTC timestamp string into local time formatted as YYYY-MM-DD HH:MM:SS.

    Behavior:
        - Converts "Z" to "+00:00" for parsing.
        - Converts from UTC to local timezone.
        - Drops milliseconds and timezone.

    Args:
        ts: Timestamp string in ISO format (e.g. "2026-03-04T12:34:56.789Z").

    Returns:
        A formatted local timestamp string, or the original input if parsing fails.
    """

    try:
        utc_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        local_dt = utc_dt.astimezone(tz.tzlocal())
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def render_table(data: dict[str, Any]) -> None:
    """
    Render a simple key/value table using Streamlit.

    Converts values to strings to avoid mixed-type rendering issues.

    Args:
        data: Dictionary to render as a table.

    Returns:
        None.
    """

    if not data:
        return

    df = pd.DataFrame(
        [{"Field": k, "Value": v} for k, v in data.items()]
    )

    # 🔑 fix the real problem
    df["Value"] = df["Value"].astype(str)

    st.table(df)


def model_to_row(m: dict[str, Any], include_user: bool) -> dict[str, Any]:
    """
    Convert a model dict from the API into a flat row dict suitable for a DataFrame.

    Args:
        m: Model dict returned from backend.
        include_user: If True, include user_id in the row.

    Returns:
        A flat dict representing one row for display.
    """

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


def prediction_to_row(p: dict[str, Any], include_user: bool) -> dict[str, Any]:
    """
    Convert a prediction dict from the API into a flat row dict suitable for a DataFrame.

    Args:
        p: Prediction dict returned from backend.
        include_user: If True, include user_id in the row.

    Returns:
        A flat dict representing one row for display.
    """

    fv = p.get("feature_values", {}) or {}

    return {
        "User ID": p.get("user_id") if include_user else None,
        "Prediction ID": p.get("id"),
        "Model Type": p.get("model_type"),
        "feature_values": ", ".join(f"{k}={v}" for k, v in fv.items()),
        "Prediction": p.get("prediction_result"),
        "Created At": format_ts(p.get("created_at")),
    }


def render_metrics_summary(metrics: dict[str, Any]) -> None:
    """
    Render a compact metrics summary using Streamlit metric widgets.

    Displays common fields if they exist:
        - accuracy, r2, mae, cv_mean (+/- cv_std)

    Args:
        metrics: Metrics dict from backend.

    Returns:
        None.
    """

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
