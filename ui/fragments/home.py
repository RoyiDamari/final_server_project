import streamlit as st
from ui.utils.session_guard import ensure_authenticated


def main() -> None:
    ensure_authenticated()
    st.title("🏠 Welcome to the AI Prediction Platform")
    st.markdown("""
    Welcome to your personal ML dashboard.

    From here you can:
    - 📈 Train custom models
    - 🔮 Make predictions
    - 📊 View your model usage
    - 💳 Manage your tokens
    """)

    st.info("Need more tokens? Click the 'Buy Tokens' option from the sidebar.")

    st.markdown("### Get Started:")
