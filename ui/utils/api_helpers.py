import streamlit as st


# ----------------------------
# Shared helpers
# ----------------------------
def logout_and_stop(msg: str) -> None:
    st.session_state.clear()
    st.session_state["logout_message"] = msg
    st.rerun()

def _format_retry_after_minutes(seconds: int | float) -> str:
    """Format seconds → minutes with 2 decimal places."""
    try:
        minutes = float(seconds) / 60
        return f"{minutes:.2f} minutes"
    except (TypeError, ValueError):
        return "a few minutes"

# ----------------------------
# For LOGIN / REGISTER
# ----------------------------
def handle_api_response(resp: dict) -> bool:
    """
    Non-fatal response handler.
    Used for login / register flows.

    Returns:
        True  -> caller may continue
        False -> caller should stop
    """
    if not isinstance(resp, dict):
        st.error("Unexpected server response.")
        return False

    # Backend business / auth errors
    if "detail" in resp:
        if "retry_after" in resp:
            retry_msg = _format_retry_after_minutes(resp["retry_after"])
            st.warning(
                f"{resp['detail']} Try again in {retry_msg}."
            )
        else:
            st.error(resp["detail"])
        return False

    # Transport / wrapper errors
    if "error" in resp:
        st.error(resp["error"])
        return False

    return True


# ----------------------------
# For PROTECTED FRAGMENTS
# ----------------------------
def handle_api_error(resp: dict) -> None:
    """
    Fatal handler.
    Used in authenticated / protected pages only.
    """
    if not isinstance(resp, dict):
        st.error("Unexpected server response.")
        st.stop()

    status = resp.get("status_code")
    detail = resp.get("detail")

    # Authentication → logout
    if status == 401:
        logout_and_stop(detail or "Session expired. Please log in again.")

    # Business rule errors (tokens, permissions, rate limit, etc.)
    if detail:
        if "retry_after" in resp:
            retry_msg = _format_retry_after_minutes(resp["retry_after"])
            st.warning(
                f"{detail} Try again in {retry_msg}."
            )
        else:
            st.error(detail)
        st.stop()

    # Transport / client errors
    if "error" in resp:
        st.error(resp["error"])
        st.stop()
