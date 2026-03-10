import streamlit as st
import time
from ui.api.auth import refresh_token
from ui.utils.api_helpers import logout_and_stop
from ui.config import TOKEN_REFRESH_THRESHOLD_SECONDS


def ensure_authenticated() -> None:
    """
    Ensure the current Streamlit session is authenticated.

    This is intended to be called at the start of protected fragments/pages.
    If the access token is missing, the user is logged out (session cleared),
    a one-time message is stored, and the app reruns to the login page.

    Returns:
        None.
    """

    if "jwt_token" not in st.session_state:
        logout_and_stop("Please log in to continue.")


def ensure_token_fresh() -> None:
    """
    Proactively refresh the access token when it is near expiration.

    Reads the current token expiry and refresh token from session_state.
    If the access token is within TOKEN_REFRESH_THRESHOLD_SECONDS of expiring,
    requests a new token pair from the backend and updates session_state.

    Requires (in st.session_state):
        - "jwt_expires_at": int (unix timestamp)
        - "refresh_token": str

    Side effects:
        - May call refresh_token(refresh_token)
        - On failure, logs the user out via logout_and_stop()
        - On success, updates:
            - st.session_state["jwt_token"]
            - st.session_state["refresh_token"]
            - st.session_state["jwt_expires_at"]

    Returns:
        None.
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
