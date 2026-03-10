import streamlit as st
import pandas as pd
from typing import Any
from ui.api.token_credit import get_all_users_tokens
from ui.api.token_credit import get_user_tokens
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error
from ui.utils.widgets_guard import render_token_guarded_button
from ui.utils.display_helpers import handle_usage_balance
from ui.config import METADATA_COST


def render_user_tokens(token: str) -> None:
    """
    Render the current user's token history with an optional max-rows display limit.

    Stores fetched rows in session_state so the table persists across reruns.

    Args:
        token: User access token.

    Returns:
        None.
    """
    st.subheader("🧍 My Token History")

    st.session_state.setdefault("user_tokens_rows", None)

    if st.button("💰 Fetch My Token History"):
        with st.spinner("Fetching your token history..."):
            resp = get_user_tokens(token)
        handle_api_error(resp)

        data = resp.get("data", [])
        if not data:
            st.info("No token credit records found.")
            st.session_state["user_tokens_rows"] = None
            return

        rows: list[dict[str, Any]] = []
        for r in data:
            row: dict[str, Any] = dict(r)
            row["status"] = str(row["status"]).capitalize()
            row["created_at"] = str(row["created_at"]).replace("T", " ").split(".")[0]
            rows.append(row)

        st.session_state["user_tokens_rows"] = rows

    rows = st.session_state.get("user_tokens_rows")
    if not rows:
        st.info("Click the button to load your token history.")
        return

    df = pd.DataFrame(rows)
    count = len(df)

    if count == 1:
        st.dataframe(df, use_container_width=True)
        return

    max_rows = st.slider(
        "Max rows to show",
        min_value=1,
        max_value=count,
        value=min(20, count),
        key="user_tokens_max_rows",
    )
    st.dataframe(df.iloc[:max_rows], use_container_width=True)


def render_all_users_tokens(token: str) -> None:
    """
    Render a table of token balances/history for all users (admin/privileged view).

    Behavior:
        - Uses a token-guarded button (METADATA_COST) to fetch data.
        - Stores the fetched rows in st.session_state["all_users_tokens_rows"]
          so the table persists across reruns.
        - Provides a slider to limit the number of displayed rows (max rows).

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.subheader("🔍 All Users' Token Balances")

    st.session_state.setdefault("all_users_tokens_rows", None)

    if render_token_guarded_button("📄 Fetch All Users' Tokens", min_tokens=METADATA_COST):

        with st.spinner("Fetching all users' token balances..."):
            resp = get_all_users_tokens(token)
        handle_api_error(resp)

        handle_usage_balance(resp)

        data = resp.get("data", [])

        if not data:
            st.warning("No user token data available.")
            st.session_state["all_users_tokens_rows"] = None
            return

        rows: list[dict[str, Any]] = []

        for r in data:
            row: dict[str, Any] = dict(r)
            row["status"] = str(row["status"]).capitalize()
            row["created_at"] = str(row["created_at"]).replace("T", " ").split(".")[0]
            rows.append(row)

        st.session_state["all_users_tokens_rows"] = rows

    rows = st.session_state.get("all_users_tokens_rows")
    if not rows:
        st.info("Click the button to view all users' token balances.")
        return

    df = pd.DataFrame(rows)
    count = len(df)

    if count == 1:
        st.dataframe(df, use_container_width=True)
        return

    max_rows = st.slider(
        "Max users to show",
        min_value=1,
        max_value=count,
        value=min(20, count),
        key="all_users_tokens_max_rows"
    )
    st.dataframe(df.iloc[:max_rows], use_container_width=True)


def clear_tokens_view_cache() -> None:
    """Clear cached token tables when switching the radio view."""
    st.session_state["user_tokens_rows"] = None
    st.session_state["all_users_tokens_rows"] = None
    st.session_state.pop("user_tokens_max_rows", None)
    st.session_state.pop("all_users_tokens_max_rows", None)


def main() -> None:
    """
    Fragment entry point for the Tokens Dashboard page.

    Behavior:
        - Ensures the user is authenticated.
        - Retrieves the current access token from session_state.
        - Renders the current user's token history table.
        - Optionally renders an all-users token table (if your UI exposes it).

    Returns:
        None.
    """

    ensure_authenticated()

    token = st.session_state["jwt_token"]

    st.header("🪙 User Tokens Dashboard")

    selected = st.radio(
        "Choose view:",
        ["My Tokens", "All Users' Tokens"],
        key="tokens_view_choice",
        on_change=clear_tokens_view_cache,
    )

    if selected == "All Users' Tokens":
        render_all_users_tokens(token)
    else:
        render_user_tokens(token)
