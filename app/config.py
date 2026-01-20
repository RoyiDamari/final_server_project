from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEBUG: bool = True

    # --- Database ---
    DATABASE_URL: str

    # --- Redis ---
    REDIS_URL: str
    REDIS_TTL: int = 86400

    MAX_TOKENS_PER_PURCHASE: int = 100
    TOKEN_PRICE: ClassVar[float] = 0.05

    # --- JWT Auth ---
    SECRET_KEY: str
    ALGORITHM: str
    TOKEN_EXPIRY_TIME: int = 30

    # Retry / Token Generation Settings
    MAX_TOKEN_GENERATION_RETRIES: int = 3

    # --- OpenAI/ChatGPT ---
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT: ClassVar[int] = 20
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"

    # Global rate limit for all actions
    RATE_LIMITS: ClassVar[dict[str, dict[str, int]]] = {
        "register": {"max_requests": 5, "window": 3600},
        "buy_tokens": {"max_requests": 3, "window": 60},
        "delete": {"max_requests": 2, "window": 300},
        "token_history": {"max_requests": 10, "window": 60},
        "all_users_tokens": {"max_requests": 5, "window": 60},
        "login": {"max_requests": 10, "window": 600},
        "refresh": {"max_requests": 10, "window": 600},
        "logout": {"max_requests": 10, "window": 60},
        "train": {"max_requests": 5, "window": 300},
        "user_models": {"max_requests": 20, "window": 60},
        "all_users_models": {"max_requests": 10, "window": 60},
        "predict": {"max_requests": 20, "window": 600},
        "user_predictions": {"max_requests": 20, "window": 60},
        "all_users_predictions": {"max_requests": 10, "window": 60},
        "explain": {"max_requests": 5, "window": 60},
        "model_type_distribution": {"max_requests": 30, "window": 60},
        "type_split": {"max_requests": 30, "window": 60},
        "label_distribution": {"max_requests": 20, "window": 60},
        "metric_distribution": {"max_requests": 10, "window": 60},
    }


config = Settings()
