import time
import streamlit as st
from streamlit_option_menu import option_menu
from ui.utils.session_guard import ensure_token_fresh
from ui.utils.api_helpers import handle_api_response
from ui.utils.display_helpers import init_sidebar_balance_slot, show_sidebar_balance
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
from ui.api.health import get_ready

st.set_page_config(page_title="ML App", layout="wide")


# --------------------------
# Login/Register Components
# --------------------------
def _show_warnings(warnings: list[str]) -> None:
    """
    Render a list of warning messages as Streamlit warning banners.

    Args:
        warnings: List of warning strings to display.

    Returns:
        None.
    """

    for msg in warnings:
        st.warning(msg)


def handle_login(username: str, password: str) -> None:
    """
    Validate login inputs, call the backend login endpoint, and initialize session state.

    Behavior:
        - Validates username/password and shows warnings if invalid.
        - Calls login_user() and uses handle_api_response() for non-fatal errors.
        - On success, stores access/refresh tokens, expiry, and token balance in session_state.
        - Redirects navigation to Home via menu_choice and triggers st.rerun().

    Args:
        username: Raw username input from the UI.
        password: Raw password input from the UI.

    Returns:
        None.
    """

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


def handle_register(first: str, last: str, username: str, email: str, password: str) -> None:
    """
    Validate registration inputs, call the backend register endpoint, and store a success message.

    Behavior:
        - Validates first/last/username/email/password and shows warnings if invalid.
        - Calls register_user() and uses handle_api_response() for non-fatal errors.
        - On success, stores a one-time success message in session_state and triggers st.rerun().

    Args:
        first: First name input.
        last: Last name input.
        username: Username input.
        email: Email input.
        password: Password input.

    Returns:
        None.
    """

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


def render_login_register() -> None:
    """
    Render the login/registration screen.

    Behavior:
        - Displays a one-time logout/session-expired message if present.
        - Renders Login and Register forms.
        - Delegates submit handling to handle_login() / handle_register().
        - Uses Streamlit rerun-based navigation.

    Returns:
        None.
    """

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
    """
    Render the sidebar navigation and token balance widget.

    Behavior:
        - Shows app icon and title.
        - Initializes and renders the token balance metric.
        - Renders the option_menu and returns the selected choice.

    Returns:
        The selected menu option string.
    """

    with st.sidebar:
        st.image("ui/assets/ai_icon.jpg")
        st.write("## ML Dashboard")

        init_sidebar_balance_slot()
        show_sidebar_balance()

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
def main() -> None:
    """
    Main Streamlit application entry point.

    Behavior:
        - Checks backend readiness (health endpoint) with exponential-ish backoff.
        - If not authenticated, renders login/register and exits.
        - Ensures token freshness (refresh if near expiry).
        - Renders sidebar, routes to selected fragment.
        - On fragment change, runs cleanup hooks (e.g., train_model.invalidate_model_context()).
        - On logout, calls backend logout and clears session.

    Returns:
        None.
    """

    st.session_state.setdefault("_ready_fail_count", 0)

    resp = get_ready()
    ready = bool(resp and resp.get("status_code") == 200)

    if not ready:
        st.session_state["_ready_fail_count"] += 1

        backoff_s = min(5, [1, 2, 3, 5][min(3, st.session_state["_ready_fail_count"] - 1)])

        st.warning(
            "Backend is starting or temporarily unavailable.\n\n"
            f"Retrying automatically in **{backoff_s}** seconds…"
        )
        time.sleep(backoff_s)
        st.rerun()

    st.session_state["_ready_fail_count"] = 0

    st.session_state.setdefault("active_fragment", None)
    token = st.session_state.get("jwt_token")

    if not token:
        render_login_register()
        return

    st.session_state.setdefault("token_balance", 0)

    ensure_token_fresh()

    choice = render_sidebar()

    # ---- fragment transition detection ----
    if st.session_state["active_fragment"] != choice:
        if st.session_state["active_fragment"] == "📈 Train Model":
            train_model.invalidate_model_context()
            train_model.clear_models_viewer_cache()
        if st.session_state["active_fragment"] == "🔮 Make Prediction":
            make_prediction.clear_predictions_viewer_cache()
        if st.session_state["active_fragment"] == "📊 User Usage Dashboard":
            user_tokens_dashboard.clear_tokens_view_cache()
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
