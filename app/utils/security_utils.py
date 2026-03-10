import secrets
from hashlib import sha256


def generate_id() -> str:
    """
    Generate a URL-safe random identifier suitable for session IDs and tokens.

    Returns:
        str: Random URL-safe string (token_urlsafe(32)).
    """

    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """
    Generate a URL-safe random identifier suitable for session IDs and tokens.

    Returns:
        str: Random URL-safe string (token_urlsafe(32)).
    """

    return sha256(token.encode("utf-8")).hexdigest()


def stable_hash(text: str) -> str:
    """
    Compute a deterministic hash for cache keys, fingerprints, and deduplication.

    Args:
        text: Input string to hash.

    Returns:
        str: Hex-encoded SHA-256 digest.
    """

    return sha256(text.encode("utf-8")).hexdigest()
