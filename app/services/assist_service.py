from typing import Optional
from redis.asyncio.client import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from app.external_api.openai_client import OpenAIClient, OpenAINotConfigured
from app.exceptions.assist import (AssistInputException, AssistUnavailableException,
                                   OpenAIConfigException, OpenAIRequestException)
from app.models.orm_models.users import User
from app.models.enums import ActionType
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.utils.security_utils import stable_hash
from app.config import config


class AssistService:
    """
    Orchestrates assist features (parameter explanations).
    - Holds a single OpenAIClient
    - Converts client/SDK errors into BaseAppException subclasses
    - Caches identical requests
    """
    _client: Optional[OpenAIClient] = None
    _init_error: Optional[str] = None

    @classmethod
    def init(cls) -> None:
        """Initialize OpenAI client once."""
        if cls._client is not None or cls._init_error is not None:
            return
        try:
            cls._client = OpenAIClient()
        except OpenAINotConfigured as e:
            cls._client = None
            cls._init_error = str(e)

    @staticmethod
    def _norm(s: str | None) -> str:
        return (s or "").strip().lower()

    @classmethod
    async def explain_param(
        cls,
        db: AsyncSession,
        redis: Redis,
        user: User,
        action: ActionType,
        model_type: Optional[str],
        param_key: Optional[str],
        context: Optional[str],
    ) -> dict:
        """
        Backward-compatible endpoint behavior:
        - If param_key provided -> PARAM MODE (ignore context)
        - If param_key missing -> QUESTION MODE (context is the free-text question)
        """

        mt = cls._norm(model_type)
        pk = cls._norm(param_key) if param_key is not None else None
        ctx = cls._norm(context).strip() if context else None

        if cls._client is None:
            raise AssistUnavailableException(log_detail=cls._init_error or "OpenAI unavailable")

        # --------------------------------------------------
        # MODE C: Free-text question
        # --------------------------------------------------
        if ctx:
            q_key = f"assist:{user.id}:question:{stable_hash(ctx)}"

            cached = await CRepo.get_version(redis, q_key)
            if cached:
                return {"data": cached, "charged": False, "balance": user.tokens}

            try:
                text = cls._client.ask_question(question=ctx, model_type=mt)
            except OpenAINotConfigured as e:
                raise OpenAIConfigException(log_detail=str(e))
            except Exception as e:
                raise OpenAIRequestException(log_detail=str(e))

            balance = await URepo.update_tokens(db, user.id, action.cost)

            await CRepo.set_cache_entity(redis, q_key, text, config.REDIS_TTL)

            return {"data": text, "charged": True, "balance": balance}

        # --------------------------------------------------
        # MODE A: Model explanation
        # --------------------------------------------------
        if mt and pk is None:
            cache_key = f"assist:{user.id}:model:{mt}"

            cached = await CRepo.get_version(redis, cache_key)
            if cached:
                return {"data": cached, "charged": False, "balance": user.tokens}

            try:
                text = cls._client.explain(model_type=mt, param_key=None)
            except OpenAINotConfigured as e:
                raise OpenAIConfigException(log_detail=str(e))
            except Exception as e:
                raise OpenAIRequestException(log_detail=str(e))

            balance = await URepo.update_tokens(db, user.id, action.cost)

            await CRepo.set_cache_entity(redis, cache_key, text, config.REDIS_TTL)

            return {"data": text, "charged": True, "balance": balance}

        # --------------------------------------------------
        # MODE B: Preset / parameter explanation
        # --------------------------------------------------
        if mt and pk:
            cache_key = f"assist:{user.id}:param:{mt}:{pk}"

            cached = await CRepo.get_version(redis, cache_key)
            if cached:
                return {"data": cached, "charged": False, "balance": user.tokens}

            try:
                text = cls._client.explain(model_type=mt, param_key=pk)
            except OpenAINotConfigured as e:
                raise OpenAIConfigException(log_detail=str(e))
            except Exception as e:
                raise OpenAIRequestException(log_detail=str(e))

            balance = await URepo.update_tokens(db, user.id, action.cost)

            await CRepo.set_cache_entity(redis, cache_key, text, config.REDIS_TTL)

            return {"data": text, "charged": True, "balance": balance}

        # --------------------------------------------------
        # INVALID INPUT
        # --------------------------------------------------
        raise AssistInputException("Provide either model_type, model_type+param_key, or context")