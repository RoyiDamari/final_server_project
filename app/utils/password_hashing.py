from passlib.context import CryptContext
from app.exceptions.user import PasswordFormatException


bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash a plain text password using bcrypt."""
    if len(password.encode("utf-8")) > 72:
        raise PasswordFormatException()

    return bcrypt_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain text password against its hashed version."""
    return bcrypt_context.verify(plain_password, hashed_password)
