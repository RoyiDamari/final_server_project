import streamlit as st
import time
from ui.api.auth import refresh_token
from ui.utils.api_helpers import logout_and_stop
from ui.config import TOKEN_REFRESH_THRESHOLD_SECONDS


def ensure_authenticated() -> None:
    """
    Ensure the user is logged in.
    """
    if "jwt_token" not in st.session_state:
        logout_and_stop("Please log in to continue.")



def ensure_token_fresh():
    """
    Proactively refresh access token if it's about to expire.

    Requires:
    - st.session_state["jwt_expires_at"]: int (unix timestamp)
    - st.session_state["refresh_token"]: str
    """
    exp = st.session_state.get("jwt_expires_at")
    refresh = st.session_state.get("refresh_token")

    if not exp or not refresh:
        return

    now = int(time.time())

    if now >= exp - TOKEN_REFRESH_THRESHOLD_SECONDS:
        new_data = refresh_token(refresh)

        if not new_data or "access_token" not in new_data:
            logout_and_stop("Session expired. Please log in again.")

        st.session_state["jwt_token"] = new_data["access_token"]
        st.session_state["refresh_token"] = new_data["refresh_token"]
        st.session_state["jwt_expires_at"] = new_data["expires_at"]

