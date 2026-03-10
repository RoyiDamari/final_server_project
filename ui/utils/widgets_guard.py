import streamlit as st


def _get_balance() -> int:
    """
    Read the current token balance from Streamlit session_state.

    This is the UI "single source of truth" for balance across fragments.

    Returns:
        The current token balance (defaults to 0 if missing).
    """

    return st.session_state.get("token_balance", 0)


def has_enough_tokens(min_tokens: int) -> bool:
    """
    Check whether the current user has at least min_tokens available.

    Args:
        min_tokens: Required minimum token count.

    Returns:
        True if current balance >= min_tokens; otherwise False.
    """

    return _get_balance() >= min_tokens


def render_not_enough_tokens_warning(min_tokens: int) -> None:
    """
    Render a user-facing warning explaining that token balance is insufficient.

    The message varies slightly depending on whether balance is zero or positive,
    and includes guidance to purchase more tokens.

    Args:
        min_tokens: Required minimum token count for the attempted action.

    Returns:
        None.
    """

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
            "ℹ️ You can buy more tokens from the **Buy Tokens** page."
        )


def render_token_guarded_button(label: str, min_tokens: int, disabled: bool = False) -> bool:
    """
    Render a Streamlit button that is guarded by token balance.

    Behavior:
        - If balance >= min_tokens: render a button (optionally disabled).
        - If balance < min_tokens: show a warning and return False.

    Args:
        label: Button label displayed in the UI.
        min_tokens: Minimum token balance required to enable the action.
        disabled: If True, the button is rendered disabled even if balance is sufficient.

    Returns:
        True if the user has enough tokens AND clicked the button; otherwise False.
    """

    balance = _get_balance()

    if balance >= min_tokens:
        return st.button(label, key=f"action_{label}", disabled=disabled)

    render_not_enough_tokens_warning(min_tokens)
    return False
