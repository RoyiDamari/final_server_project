from passlib.context import CryptContext
from app.exceptions.user import PasswordFormatException

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Note:
        bcrypt only uses the first 72 bytes of the password, so we reject longer inputs
        to avoid surprising behavior.

    Args:
        password: Plain-text password.

    Returns:
        str: bcrypt hash string.

    Raises:
        PasswordFormatException: If the password exceeds bcrypt's 72-byte effective limit.
    """

    if len(password.encode("utf-8")) > 72:
        raise PasswordFormatException()

    return bcrypt_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a bcrypt hash.

    Args:
        plain_password: Password provided by the user.
        hashed_password: Stored bcrypt hash.

    Returns:
        bool: True if password matches, False otherwise.
    """

    return bcrypt_context.verify(plain_password, hashed_password)
