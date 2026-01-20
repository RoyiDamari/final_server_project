import streamlit as st


def _get_balance() -> int:
    """
    Single source of truth for token balance in UI.
    """
    return st.session_state.get("token_balance", 0)


def has_enough_tokens(min_tokens: int) -> bool:
    return _get_balance() >= min_tokens


def render_not_enough_tokens_warning(min_tokens: int):
    balance = _get_balance()

    if balance == 0:
        st.warning(
            f"🚫 You need **{min_tokens} tokens** to use this feature.\n"
            f"💰 You currently have **0 tokens**."
        )
    else:
        st.warning(
            f"🚫 You need **{min_tokens} tokens** to use this feature.\n"
            f"💰 You currently have **{balance} tokens**.\n\n"
            "ℹ️ You can buy more tokens **only after finishing your current balance**."
        )


def render_token_guarded_button(label: str, min_tokens: int) -> bool:
    """
    Renders a button guarded by token balance.
    Returns True only when action is allowed and clicked.
    """
    balance = _get_balance()

    # Enough tokens → normal button
    if balance >= min_tokens:
        return st.button(label, key=f"action_{label}")

    # Not enough tokens
    render_not_enough_tokens_warning(min_tokens)

    return False
