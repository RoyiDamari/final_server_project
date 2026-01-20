import streamlit as st
import uuid
import re
from ui.api.token_credit import buy_tokens
from ui.utils.validators import normalize_credit_card_number, validate_credit_card_number
from ui.config import TOKEN_PRICE
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error


def _format_cc_for_display(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    return "-".join(digits[i:i + 4] for i in range(0, len(digits), 4))


def _on_cc_change():
    raw = st.session_state.cc_input
    formatted = _format_cc_for_display(raw)
    st.session_state.cc_input = formatted


def buy_tokens_ui(token: str):
    st.header("💳 Buy Tokens")

    if "purchase_success_message" in st.session_state:
        st.success(st.session_state.pop("purchase_success_message"))

    if "purchase_key" not in st.session_state:
        st.session_state.purchase_key = None

    if "cc_input" not in st.session_state:
        st.session_state.cc_input = ""

    amount = st.slider(
        "Number of tokens to buy",
        min_value=1,
        max_value=100,
        value=10,
        key="buy_amount"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Price per token: ${TOKEN_PRICE:.2f}")
    with col2:
        st.caption(f"Total: ${amount * TOKEN_PRICE:.2f}")

    credit_card = st.text_input(
        "Credit Card (16 digits)",
        key="cc_input",
        on_change=_on_cc_change,
        placeholder="1234-5678-9012-3456"
    )

    with st.form("buy_tokens_form"):
        submitted = st.form_submit_button(f"Buy Tokens (${amount * TOKEN_PRICE:.2f})")

    if submitted:
        normalized_cc = normalize_credit_card_number(credit_card)
        card_error = validate_credit_card_number(normalized_cc)

        if card_error:
            st.warning(card_error)
            return

        if not st.session_state.purchase_key:
            st.session_state.purchase_key = str(uuid.uuid4())

        with st.spinner("Processing purchase..."):
            resp = buy_tokens(
                token,
                normalized_cc,
                amount,
                idempotency_key=st.session_state.purchase_key
            )

        handle_api_error(resp)

        message = resp.get("message")
        balance = resp.get("balance")

        if not message or balance is None:
            st.error("Unexpected server response. Please try again later.")
            return

        # Persist across rerun
        st.session_state["purchase_success_message"] = message
        st.session_state["token_balance"] = balance

        st.session_state.purchase_key = None
        st.rerun()


def main():
    """Thin wrapper to match all other fragment patterns."""
    ensure_authenticated()
    token = st.session_state["jwt_token"]
    buy_tokens_ui(token)
