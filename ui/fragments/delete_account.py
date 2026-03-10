import streamlit as st
from ui.api.user import delete_user
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error


def delete_account_ui(token: str) -> None:
    """
    Render the Delete Account UI and delete the authenticated user's account on submit.

    Behavior:
        - Collects username and password confirmation.
        - If the user has a positive token balance, requires an extra confirmation checkbox
          acknowledging token forfeiture.
        - Calls the backend delete_user endpoint.
        - On success, clears session state and triggers a rerun to return user to login page
          with a one-time logout message.

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.header("🗑️ Delete My Account")

    token_balance = st.session_state.get("token_balance", 0)

    with st.form("delete_account_form"):
        username = st.text_input("Confirm your username")
        password = st.text_input("Confirm your password", type="password")

        confirm_balance = False
        if token_balance > 0:
            st.warning(
                f"You still have **{token_balance} tokens**.\n\n"
                "Deleting your account will permanently forfeit them."
            )
            confirm_balance = st.checkbox(
                "I understand and want to delete my account anyway"
            )

        submitted = st.form_submit_button("Delete Account")

    if submitted:
        if not username or not password:
            st.warning("Please enter both username and password.")
            return

        if token_balance > 0 and not confirm_balance:
            st.warning("You must confirm token forfeiture to proceed.")
            return

        with st.spinner("Deleting account..."):
            resp = delete_user(token, username, password, confirm_balance)

        handle_api_error(resp)

        message = resp.get("message")
        if not message:
            st.error("Unexpected server response. Please try again later.")
            return

        st.session_state.clear()
        st.session_state["logout_message"] = message
        st.rerun()


def main() -> None:
    """
    Fragment entry point for the Delete Account page.

    Behavior:
        - Ensures the user is authenticated.
        - Retrieves the current access token from session_state.
        - Renders the delete account form (username/password confirmation).
        - Requires explicit confirmation if token balance > 0.
        - On success, clears session state and returns the user to login screen.

    Returns:
        None.
    """

    ensure_authenticated()
    token = st.session_state["jwt_token"]
    delete_account_ui(token)
