import re

# Simple length caps to mirror backend/DB (names & email only)
FIRST_LAST_MIN = 1
FIRST_LAST_MAX = 50
EMAIL_MAX      = 120

# Regex (same as backend)
USERNAME_REGEX = r"^[a-zA-Z0-9_-]{3,20}$"
PASSWORD_REGEX = r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@#$%^&+=!]{6,20}$"
EMAIL_REGEX    = r"^[^@]+@[^@]+\.[^@]+$"

def validate_first_name(value: str) -> str | None:
    v = (value or "").strip()
    if not (FIRST_LAST_MIN <= len(v) <= FIRST_LAST_MAX):
        return f"First name must be {FIRST_LAST_MIN}-{FIRST_LAST_MAX} characters."
    return None

def validate_last_name(value: str) -> str | None:
    v = (value or "").strip()
    if not (FIRST_LAST_MIN <= len(v) <= FIRST_LAST_MAX):
        return f"Last name must be {FIRST_LAST_MIN}-{FIRST_LAST_MAX} characters."
    return None

def validate_username(value: str) -> str | None:
    v = (value or "").strip().lower()
    if not re.match(USERNAME_REGEX, v):
        return "Username must be 3-20 chars and include only letters, digits, underscores, or hyphens."
    return None

def validate_email(value: str) -> str | None:
    v = (value or "").strip().lower()
    if len(v) > EMAIL_MAX:
        return f"Email must be at most {EMAIL_MAX} characters."
    if not re.match(EMAIL_REGEX, v):
        return "Please enter a valid email address."
    return None

def validate_password(value: str) -> str | None:
    p = value or ""
    if not re.match(PASSWORD_REGEX, p):
        return "Password must be 6-20 chars and include at least one letter and one number."
    return None

def normalize_credit_card_number(card: str) -> str:
    """Return only digits from any CC input (keystrokes may include spaces/dashes)."""
    return re.sub(r"\D", "", card or "")

def validate_credit_card_number(digits: str) -> str | None:
    if len(digits) != 16:
        return "Credit card must contain exactly 16 digits."
    if not digits.isdigit():
        return "Credit card must include digits only."
    return None
