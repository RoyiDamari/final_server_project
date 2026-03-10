import streamlit as st
import uuid
import re
from ui.api.token_credit import buy_tokens
from ui.utils.validators import normalize_credit_card_number, validate_credit_card_number
from ui.config import MAX_TOKENS, MAX_TOKENS_PER_PURCHASE, TOKEN_PRICE
from ui.utils.session_guard import ensure_authenticated
from ui.utils.api_helpers import handle_api_error


def _format_cc_for_display(raw: str) -> str:
    """
    Format a credit card string for display as groups of 4 digits separated by '-'

    Non-digit characters are removed before formatting.

    Args:
        raw: Raw credit card input (may include spaces/dashes/letters).

    Returns:
        A formatted string like "1234-5678-9012-3456" (or partial groups if shorter).
    """

    digits = re.sub(r"\D", "", raw or "")
    return "-".join(digits[i:i + 4] for i in range(0, len(digits), 4))


def _on_cc_change() -> None:
    """
    Streamlit on_change handler for the credit card input.

    Reads st.session_state.cc_input, formats it for display, and writes it back.

    Returns:
        None.
    """

    raw = st.session_state.cc_input
    formatted = _format_cc_for_display(raw)
    st.session_state.cc_input = formatted


def _ensure_purchase_key() -> str:
    """
    Ensures one stable idempotency key per *attempt*.
    Do NOT clear this until success is confirmed.
    """
    if not st.session_state.get("purchase_key"):
        st.session_state.purchase_key = str(uuid.uuid4())
    return st.session_state.purchase_key


def buy_tokens_ui(token: str) -> None:
    """
    Render the Buy Tokens UI and perform the purchase when the user submits.

    Behavior:
        - Computes how many tokens the user is allowed to buy (cap-aware).
        - Collects credit card input and validates it.
        - Uses st.session_state["buy_in_flight"] to disable inputs during the request.
        - Uses an idempotency key (purchase_key) so repeated submits do not double-charge.
        - On success, stores a one-time success message and updates token_balance.

    Args:
        token: User access token.

    Returns:
        None.
    """

    st.header("💳 Buy Tokens")

    st.session_state.setdefault("purchase_key", None)
    st.session_state.setdefault("buy_in_flight", False)
    st.session_state.setdefault("cc_input", "")
    st.session_state.setdefault("token_balance", 0)

    if "purchase_success_message" in st.session_state:
        st.success(st.session_state.pop("purchase_success_message"))

    balance = st.session_state.get("token_balance", 0) or 0
    remaining_capacity = max(0, MAX_TOKENS - balance)
    max_buy = min(MAX_TOKENS_PER_PURCHASE, remaining_capacity)

    if max_buy == 0:
        st.info(f"You're at the maximum balance ({MAX_TOKENS}). Use some tokens before buying more.")
        return

    if max_buy == 1:
        amount = 1
        st.info("You can buy **1** token (you’re almost at the cap).")
    else:
        default_value = min(st.session_state.get("buy_amount", 10), max_buy)

        amount = st.slider(
            "Number of tokens to buy",
            min_value=1,
            max_value=max_buy,
            value=default_value,
            key="buy_amount",
            disabled=st.session_state.buy_in_flight,
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
        placeholder="1234-5678-9012-3456",
        disabled=st.session_state.buy_in_flight
    )

    submitted = st.button(
        f"Buy Tokens (${amount * TOKEN_PRICE:.2f})",
        disabled=st.session_state.buy_in_flight,
    )

    if submitted:
        normalized_cc = normalize_credit_card_number(credit_card)
        card_error = validate_credit_card_number(normalized_cc)

        if card_error:
            st.warning(card_error)
            return

        st.session_state.buy_in_flight = True

        key = _ensure_purchase_key()

        try:
            with st.spinner("Processing purchase..."):
                resp = buy_tokens(
                    token,
                    normalized_cc,
                    amount,
                    idempotency_key=key
                )

            handle_api_error(resp)

            message = resp.get("message")
            balance = resp.get("balance")

            if not message or balance is None:
                st.error("Unexpected server response. Please try again later.")
                return

            st.session_state["purchase_success_message"] = message
            st.session_state["token_balance"] = balance

            st.session_state.purchase_key = None

        finally:
            st.session_state.buy_in_flight = False

        st.rerun()


def main():
    """
    Fragment entry point for the Buy Tokens page.

    Behavior:
        - Ensures the user is authenticated.
        - Retrieves the current access token from session_state.
        - Renders the buy tokens UI (amount, credit card input, submit purchase).
        - Updates session token balance and shows success message after purchase.

    Returns:
        None.
    """

    ensure_authenticated()
    token = st.session_state["jwt_token"]
    buy_tokens_ui(token)
