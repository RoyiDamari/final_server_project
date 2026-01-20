class CacheKeys:
    """Centralized Redis cache keys for consistency."""

    @staticmethod
    def rate_limit(identifier: str) -> str:
        return f"ratelimit:{identifier}"


