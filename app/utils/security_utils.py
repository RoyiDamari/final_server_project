import secrets
from hashlib import sha256


def generate_id() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def stable_hash(text: str) -> str:
    """
    Deterministic hash for cache keys, fingerprints, etc.
    """
    return sha256(text.encode("utf-8")).hexdigest()