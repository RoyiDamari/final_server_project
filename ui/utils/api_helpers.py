import streamlit as st


def logout_and_stop(msg: str) -> None:
    """
    Log out the current user and immediately stop the current Streamlit flow.

    This clears the current session state (removing auth tokens and any cached UI state),
    stores a one-time logout message in session_state, then triggers a rerun so the app
    returns to the login/register screen.

    Args:
        msg: Message to show to the user on the next run (e.g., "Session expired...").

    Returns:
        None.
    """

    st.session_state.clear()
    st.session_state["logout_message"] = msg
    st.rerun()


def _format_retry_after_minutes(seconds: int | float) -> str:
    """
    Convert a retry-after duration in seconds to a friendly minute string.

    Args:
        seconds: Retry duration in seconds (int/float).

    Returns:
        A human-readable string in minutes with 2 decimals (e.g., "1.50 minutes").
        If conversion fails, returns "a few minutes".
    """

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
    Handle a non-fatal API response for public (unauthenticated) flows.

    Intended usage: login / register pages where an error should be shown to the user
    but the app should NOT stop execution with st.stop().

    Behavior:
      - If response is not a dict -> show generic error, return False.
      - If response contains "detail" -> show error/warning (with optional retry-after), return False.
      - If response contains "error"  -> show error, return False.
      - Otherwise -> consider response OK, return True.

    Args:
        resp: API response dict returned from the client wrapper.

    Returns:
        True if the caller may continue processing the response.
        False if the caller should stop the current action (but not st.stop()).
    """

    if not isinstance(resp, dict):
        st.error("Unexpected server response.")
        return False

    if "detail" in resp:
        if "retry_after" in resp:
            retry_msg = _format_retry_after_minutes(resp["retry_after"])
            st.warning(
                f"{resp['detail']} Try again in {retry_msg}."
            )
        else:
            st.error(resp["detail"])
        return False

    if "error" in resp:
        st.error(resp["error"])
        return False

    return True


def handle_api_error(resp: dict) -> None:
    """
    Handle a fatal API response for authenticated (protected) fragments.

    Intended usage: pages that require a valid login token (train/predict/dashboards).
    Any error should either:
      - force logout and rerun (401),
      - or stop execution with st.stop() after showing a message.

    Behavior:
      - If response is not a dict -> show generic error and st.stop().
      - If status_code == 401 -> clear session, show logout message, rerun.
      - If "detail" exists -> show error/warning (with optional retry-after), st.stop().
      - If "error" exists -> show error, st.stop().
      - Otherwise -> returns normally (no error).

    Args:
        resp: API response dict returned from the client wrapper.

    Returns:
        None. (May call st.stop() or st.rerun() depending on error type.)
    """

    if not isinstance(resp, dict):
        st.error("Unexpected server response.")
        st.stop()

    status = resp.get("status_code")
    detail = resp.get("detail")

    if status == 401:
        logout_and_stop(detail or "Session expired. Please log in again.")

    if detail:
        if "retry_after" in resp:
            retry_msg = _format_retry_after_minutes(resp["retry_after"])
            st.warning(
                f"{detail} Try again in {retry_msg}."
            )
        else:
            st.error(detail)
        st.stop()

    if "error" in resp:
        st.error(resp["error"])
        st.stop()
