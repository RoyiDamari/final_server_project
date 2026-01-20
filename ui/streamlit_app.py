import streamlit as st
from streamlit_option_menu import option_menu
from ui.utils.session_guard import ensure_token_fresh
from ui.utils.api_helpers import handle_api_response
from ui.utils.display_helpers import show_sidebar_balance
from ui.api.auth import login_user, logout_user
from ui.api.user import register_user
from ui.utils.validators import (
    validate_first_name, validate_last_name,
    validate_username, validate_password, validate_email
)
import fragments.home as home
import fragments.train_model as train_model
import fragments.prediction as make_prediction
import fragments.user_usage_dashboard as user_usage_dashboard
import fragments.user_tokens_dashboard as user_tokens_dashboard
import fragments.buy_tokens as buy_tokens
import fragments.delete_account as delete_account


st.set_page_config(page_title="ML App", layout="wide")


# --------------------------
# Login/Register Components
# --------------------------
def _show_warnings(warnings: list[str]):
    for msg in warnings:
        st.warning(msg)


def handle_login(username: str, password: str):
    warnings = []
    if not username or not password:
        warnings.append("Please fill in both username and password.")
    else:
        if msg := validate_username(username):
            warnings.append(msg)
        if msg := validate_password(password):
            warnings.append(msg)

    if warnings:
        _show_warnings(warnings)
        return

    with st.spinner("Logging in..."):
        resp = login_user(username.strip().lower(), password)

    if not handle_api_response(resp):
        return

    required_fields = ["access_token", "refresh_token", "expires_at"]

    missing = [k for k in required_fields if k not in resp]
    if missing:
        st.error("Unexpected server response. Please try again later.")
        return

    st.session_state["jwt_token"] = resp["access_token"]
    st.session_state["refresh_token"] = resp["refresh_token"]
    st.session_state["jwt_expires_at"] = resp["expires_at"]
    st.session_state["token_balance"] = resp["balance"]
    st.session_state["menu_choice"] = "🏠 Home"

    message = resp.get("message")
    if not message:
        st.error("Unexpected server response. Please try again later.")
        return

    st.success(message)
    st.rerun()


def handle_register(first, last, username, email, password):
    st.session_state["register_open"] = True

    warnings = []
    for validator, value in [
        (validate_first_name, first),
        (validate_last_name, last),
        (validate_username, username),
        (validate_email, email),
        (validate_password, password),
    ]:
        if msg := validator(value):
            warnings.append(msg)

    if warnings:
        _show_warnings(warnings)
        return

    with st.spinner("Registering..."):
        resp = register_user(
            first.strip(),
            last.strip(),
            username.strip().lower(),
            email.strip().lower(),
            password
        )

    if not handle_api_response(resp):
        return

    message = resp.get("message")
    if not message:
        st.error("Unexpected server response. Please try again later.")
        return

    st.session_state["register_success_message"] = message
    st.rerun()


def render_login_register():
    """
        Render the login / registration page.

        - Displays session-expiry or logout messages (if any)
        - Handles login and registration via forms
        - Designed to work with Streamlit rerun-based navigation
        """

    # --- Show logout / session-expired message (one-time) ---
    logout_msg = st.session_state.pop("logout_message", None)
    if logout_msg:
        st.warning(logout_msg)

    st.title("🔐 Welcome to the AI Platform")

    st.session_state.setdefault("register_open", False)

    with st.expander("Login", expanded=True):
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            submitted = st.form_submit_button("Login")

        if submitted:
            handle_login(username, password)

    with st.expander("Register", expanded=st.session_state["register_open"]):
        with st.form("register_form"):
            first = st.text_input("First Name")
            last = st.text_input("Last Name")
            username = st.text_input("Username", key="reg_user")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password", key="reg_pw")

            submitted = st.form_submit_button("Register")

        if submitted:
            handle_register(first, last, username, email, password)

        success_msg = st.session_state.pop("register_success_message", None)
        if success_msg:
            st.success(success_msg)


# --------------------------
# Sidebar Navigation
# --------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.image("ui/assets/ai_icon.jpg")
        st.write("## ML Dashboard")

        return option_menu(
            menu_title="Main Menu",
            options=[
                "🏠 Home",
                "💳 Buy Tokens",
                "🗑️ Delete Account",
                "📈 Train Model",
                "🔮 Make Prediction",
                "📊 User Usage Dashboard",
                "🪙 Tokens Dashboard",
                "🚪 Logout"
            ],
            icons=[
                "house",
                "credit-card",
                "trash",
                "bar-chart",
                "cpu",
                "graph-up",
                "wallet2",
                "box-arrow-right"
            ],
            default_index=0,
            key="menu_choice"
        )


# -----------------------------------
# Main Entry Logic
# -----------------------------------
def main():
    st.session_state.setdefault("active_fragment", None)
    token = st.session_state.get("jwt_token")

    if not token:
        render_login_register()
        return

    st.session_state.setdefault("token_balance", 0)

    ensure_token_fresh()

    show_sidebar_balance()

    choice = render_sidebar()

    # ---- fragment transition detection ----

    if st.session_state["active_fragment"] != choice:
        if st.session_state["active_fragment"] == "📈 Train Model":
            train_model.invalidate_model_context()
        st.session_state["active_fragment"] = choice

    match choice:
        case "🏠 Home":
            home.main()

        case "💳 Buy Tokens":
            buy_tokens.main()

        case "🗑️ Delete Account":
            delete_account.main()

        case "📈 Train Model":
            train_model.main()

        case "🔮 Make Prediction":
            make_prediction.main()

        case "📊 User Usage Dashboard":
            user_usage_dashboard.main()

        case "🪙 Tokens Dashboard":
            user_tokens_dashboard.main()

        case "🚪 Logout":
            logout_user(
                access_tok=st.session_state.get("jwt_token"),
                refresh_tok=st.session_state.get("refresh_token"),
            )
            st.session_state.clear()
            st.rerun()


main()
