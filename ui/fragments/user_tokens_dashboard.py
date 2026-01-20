import streamlit as st
import pandas as pd
from typing import Any
from ui.api.user import get_all_users_tokens
from ui.api.token_credit import get_user_token_history
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error
from ui.utils.widgets_guard import render_token_guarded_button
from ui.utils.display_helpers import handle_usage_balance
from ui.config import METADATA_COST


def render_all_users_tokens(token: str):
    st.subheader("🔍 All Users' Token Balances")

    if render_token_guarded_button("📄 Fetch All Users' Tokens", min_tokens=METADATA_COST):

        with st.spinner("Fetching all users' token balances..."):
            resp = get_all_users_tokens(token)
            handle_api_error(resp)
            handle_usage_balance(resp)

        data = resp.get("data", [])

        if not data:
            st.warning("No user token data available.")
            return

        df = pd.DataFrame(data)
        count = len(df)

        if count == 1:
            st.dataframe(df, use_container_width=True)
        else:
            max_rows = st.slider(
                "Max users to show",
                min_value=1,
                max_value=count,
                value=min(20, count)
            )
            st.dataframe(df.iloc[:max_rows], use_container_width=True)
    else:
        st.info("Click the button to view all users' token balances.")

def render_user_token_history(token: str):
    st.subheader("🧍 My Token History")

    if st.button("💰 Fetch My Token History"):
        with st.spinner("Fetching your token history..."):
            resp = get_user_token_history(token)
            handle_api_error(resp)
            handle_usage_balance(resp)

        data = resp.get("data")

        if not data:
            st.info("No token credit records found.")
            return

        rows: list[dict[str, Any]] = []

        for r in data:
            row: dict[str, Any] = dict(r)
            row["status"] = str(row["status"]).capitalize()
            row["created_at"] = str(row["created_at"]).replace("T", " ").split(".")[0]
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Click the button to load your token history.")

def main():
    ensure_authenticated()

    token = st.session_state["jwt_token"]

    st.header("🪙 User Tokens Dashboard")

    selected = st.radio("Choose view:", ["My Tokens", "All Users' Tokens"])

    if selected == "All Users' Tokens":
        render_all_users_tokens(token)
    else:
        render_user_token_history(token)

