import asyncio
import uuid
from typing import Optional, Any, Callable
from contextlib import suppress
from redis.asyncio.client import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession
from app.external_api.openai_client import OpenAIClient, OpenAINotConfigured
from app.exceptions.assist import (
    AssistUnavailableException,
    OpenAIConfigException,
    OpenAIRequestException,
    AssistInProgressException,
    AssistFailedException,
    AssistInputException,
)
from app.exceptions.base import BaseAppException
from app.models.orm_models.users import User
from app.models.enums import ActionType
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.cache_repository import CacheRepository as CRepo
from app.utils.security_utils import stable_hash
from app.config import config


class AssistService:
    """
    Assist orchestration service with Redis caching + concurrency protection.

    Goals:
      - Cache identical answers in Redis (TTL).
      - Charge tokens only when generating a new answer (cache miss).
      - Prevent duplicate OpenAI calls under concurrency:
            cache miss -> try lock
            if lock not acquired -> raise AssistInProgressException immediately
            else -> call OpenAI, charge, cache, return
    """

    _client: Optional[OpenAIClient] = None
    _init_error: Optional[str] = None

    @classmethod
    def init(cls) -> None:
        """
        Initialize OpenAI client once per process.

        Returns:
            None.
        """
        if cls._client is not None or cls._init_error is not None:
            return
        try:
            cls._client = OpenAIClient()
        except OpenAINotConfigured as e:
            cls._client = None
            cls._init_error = str(e)

    @staticmethod
    def _norm(s: str | None) -> str:
        """
        Normalize text input for cache-key stability.

        Args:
            s: Optional string.

        Returns:
            str: Lowercased, stripped string (empty if None).
        """
        return (s or "").strip().lower()

    @staticmethod
    def _lock_key(cache_key: str) -> str:
        """
        Derive lock key from cache key.

        Args:
            cache_key: Cache key storing the final answer.

        Returns:
            str: Lock key.
        """
        return f"{cache_key}:lock"

    @staticmethod
    async def _fresh_balance(db: AsyncSession, user_id: int) -> int:
        """
        Fetch a fresh token balance from DB (avoid stale ORM objects).

        Args:
            db: Async SQLAlchemy session.
            user_id: User id.

        Returns:
            int: Current token balance.
        """
        async with db.begin():
            return await URepo.get_tokens_by_id(db, user_id)

    @classmethod
    async def _serve_cached_or_compute(
        cls,
        db: AsyncSession,
        redis: Redis,
        user: User,
        action: ActionType,
        cache_key: str,
        compute_fn: Callable[[], str],
    ) -> dict[str, Any]:
        """
        Shared logic for all assist modes (question/model/param).

        Flow:
          1) If cache hit -> return cached, charged=False, fresh balance.
          2) Cache miss -> try to acquire lock (SET NX EX)
             - If not acquired -> raise AssistInProgressException immediately
          3) If acquired:
             - Call OpenAI (compute_fn)
             - Charge tokens (DB)
             - Cache answer (best-effort)
             - Return charged=True

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client.
            user: Authenticated user.
            action: ActionType.ASSIST (token cost).
            cache_key: Redis key for the answer.
            compute_fn: Callable that performs the OpenAI call and returns text.

        Returns:
            dict[str, Any]: {"data": str, "charged": bool, "balance": int}

        Raises:
            AssistInProgressException: If lock not acquired on cache miss.
            OpenAIConfigException / OpenAIRequestException: OpenAI failures.
            BaseAppException: Any known billing/domain errors from repositories.
            AssistFailedException: If assist fails, publishing fails, or DB apply fails.
        """

        cached = await CRepo.get_version(redis, cache_key)
        if cached:
            balance = await cls._fresh_balance(db, user.id)
            return {"data": cached, "charged": False, "balance": balance}

        lock_key = cls._lock_key(cache_key)
        lock_val = str(uuid.uuid4())
        ttl_s = config.ASSIST_LOCK_TTL_S

        got_lock = await CRepo.try_set_lock(redis, lock_key, lock_val, ttl_s=ttl_s)
        if not got_lock:
            raise AssistInProgressException()

        try:
            try:
                text = compute_fn()
            except OpenAINotConfigured as e:
                raise OpenAIConfigException(log_detail=str(e))
            except Exception as e:
                raise OpenAIRequestException(log_detail=str(e))

            try:
                async with db.begin():
                    balance = await URepo.update_tokens(db, user.id, action.cost)
            except BaseAppException:
                raise
            except Exception as e:
                raise AssistFailedException(log_detail=f"db apply failed: {e!r}") from e

            with suppress(RedisError, asyncio.TimeoutError):
                await CRepo.set_cache_entity(redis, cache_key, text, int(config.REDIS_TTL))

            return {"data": text, "charged": True, "balance": balance}

        finally:
            await CRepo.release_lock(redis, lock_key, lock_val)

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
    ) -> dict[str, Any]:
        """
        Generate or fetch a cached explanation.

        Modes:
          - QUESTION MODE: `context` provided.
          - MODEL MODE: `model_type` provided, `param_key` is None.
          - PARAM MODE: `model_type` and `param_key` provided.

        Concurrency:
          - Cache miss tries a Redis lock.
          - If lock is held, raises AssistInProgressException immediately
            (no waiting / no timing dependence).

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client.
            user: Authenticated user.
            action: ActionType.ASSIST.
            model_type: Model identifier or None.
            param_key: Preset/param key or None.
            context: Free-text question or None.

        Returns:
            dict[str, Any]: {"data": str, "charged": bool, "balance": int}

        Raises:
            AssistUnavailableException: OpenAI client unavailable.
            AssistInputException: Invalid mode inputs.
        """

        mt = cls._norm(model_type)
        pk = cls._norm(param_key) if param_key is not None else None
        ctx = cls._norm(context).strip() if context else None

        if cls._client is None:
            raise AssistUnavailableException(log_detail=cls._init_error or "OpenAI unavailable")

        # Mode C: free-text question
        if ctx:
            cache_key = f"assist:{user.id}:question:{stable_hash(ctx)}"

            def _compute() -> str:
                return cls._client.ask_question(question=ctx, model_type=mt)

            return await cls._serve_cached_or_compute(db, redis, user, action, cache_key, _compute)

        # Mode A: model explanation
        if mt and pk is None:
            cache_key = f"assist:{user.id}:model:{mt}"

            def _compute() -> str:
                return cls._client.explain(model_type=mt, param_key=None)

            return await cls._serve_cached_or_compute(db, redis, user, action, cache_key, _compute)

        # Mode B: param/preset explanation
        if mt and pk:
            cache_key = f"assist:{user.id}:param:{mt}:{pk}"

            def _compute() -> str:
                return cls._client.explain(model_type=mt, param_key=pk)

            return await cls._serve_cached_or_compute(db, redis, user, action, cache_key, _compute)

        raise AssistInputException("Provide either model_type, model_type+param_key, or context")