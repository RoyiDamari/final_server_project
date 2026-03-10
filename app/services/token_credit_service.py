import asyncio
from uuid import UUID
from contextlib import suppress
from typing import Mapping, Dict, Any
from redis.exceptions import RedisError
from redis.asyncio.client import Redis
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.cache_repository import CacheRepository as CRepo
from app.repositories.user_repository import UserRepository as URepo
from app.repositories.token_credit_repository import TokenCreditRepository as TCRepo
from app.repositories.seen_version_repository import SeenVersionRepository as SVRepo
from app.exceptions.base import BaseAppException
from app.exceptions.token_credit import (PurchaseInProgressException, PurchaseFailedException)
from app.models.orm_models.users import User
from app.models.pydantic_models.token_credit import BuyTokensResponse, TokenCreditResponse
from app.models.enums import ActionType, RowStatus
from app.core.logs import log_action, errors
from app.utils.cache_invalidation import invalidate_global_token_credits_cache


class TokenCreditService:
    @staticmethod
    async def buy_tokens(
            db: AsyncSession,
            user: User,
            redis: Redis,
            amount: int,
            key: str | UUID
    ) -> BuyTokensResponse:
        """
        Idempotent token purchase (credit) using a per-user idempotency key.

        Flow:
          1) Insert a TokenCredit row in 'pending' for (user_id, key) exactly once.
             - If inserted → this request owns the key and should apply the purchase.
             - If not inserted → a row already exists for this key (duplicate or in-progress).

          2) If inserted:
             - Credit the user only if policy allows (e.g. only when current tokens == 0).
             - Mark the TokenCredit row as 'applied' and store open_balance (the resulting tokens).

          3) If duplicate:
             - If existing row is 'applied' → return its recorded open_balance (idempotent replay).
             - Otherwise (pending) → raise PurchaseInProgressException.

        Cache / Redis:
          - After a successful apply, bump the global token-credits metadata cache version
            using the applied row's created_at timestamp (or other monotonic version).


        Args:
            db: Async SQLAlchemy session.
            user: Authenticated user ORM instance.
            redis: Redis client used for metadata cache invalidation / versioning.
            amount: Tokens to add (validated upstream).
            key: Idempotency key (unique per user).

        Returns:
            BuyTokensResponse: Confirmation + resulting balance (idempotent per key).

        Raises:
            BalanceMustBeZeroException: If policy requires tokens==0 and user had > 0.
            PurchaseInProgressException: If a pending row exists for this key.
        """

        key = str(key)

        async with db.begin():
            row_id = await TCRepo.try_insert_pending(db, user.id, key)

            if row_id is None:
                existing = await TCRepo.get_by_key_status_amount_balance_after(db, user.id, key)

                if not existing or existing["status"] == RowStatus.pending:
                    raise PurchaseInProgressException()

                result_balance = existing["balance_after"]
                return BuyTokensResponse(
                    message=f"{existing['amount']} tokens has been added.",
                    balance=result_balance,
                )

        try:
            async with db.begin():
                new_balance = await URepo.add_tokens(db, user.id, amount)
                applied = await TCRepo.mark_applied(db, row_id, amount, new_balance)

                if not applied:
                    raise PurchaseInProgressException()

        except BaseAppException:
            await TokenCreditService._mark_failed(db, row_id)
            raise
        except Exception as e:
            await TokenCreditService._mark_failed(db, row_id)
            raise PurchaseFailedException(log_detail=f"apply step failed: {e!r}") from e

        result_balance = applied["balance_after"]

        with suppress(RedisError, asyncio.TimeoutError):
            ts = applied["created_at"].isoformat()
            await invalidate_global_token_credits_cache(redis, ts)

        try:
            log_action(
                "tokens_purchased",
                user_id=user.id,
                username=user.username,
                credited=amount,
                balance_after=result_balance,
            )
        except Exception as e:
            errors.exception("log_action failed in buy_tokens: %r", e)

        return BuyTokensResponse(
            message=f"{applied['amount']} tokens has been added.",
            balance=result_balance,
        )

    @staticmethod
    async def get_user_tokens(db: AsyncSession, user: User) -> list[Mapping[str, Any]]:
        """
        Fetch the authenticated user's token credit history.

        This endpoint is read-only:
          - No token charge
          - No writes / side effects
          - Returns an empty list if the user has no purchase history

        Args:
        db: Async SQLAlchemy session.
        user: Authenticated user ORM instance.

        Returns:
            list[Mapping[str, Any]]: Rows returned by TokenCreditRepository.get_user_tokens().
                Each row typically includes username, amount, balance_after, current_tokens,
                status, and created_at.

        Raises:
            Exception: Not raised intentionally; unexpected errors propagate (DB errors, etc.).
        """

        user_tokens = await TCRepo.get_user_tokens(db, user.id)

        try:
            log_action(
                event="user_viewed_his_tokens_history",
                user_id=user.id,
                username=user.username,
                charged=False,
            )
        except Exception as e:
            errors.exception("log_action failed in get_user_tokens: %r", e)

        return user_tokens

    @staticmethod
    async def get_all_users_tokens(
            db: AsyncSession,
            redis: Redis,
            user: User,
            action: ActionType,
    ) -> Dict[str, Any]:
        """
        Return all users' token-credit history (ACTIVE users) with per-user-per-version billing.

        Billing correctness:
            - Version source of truth is Postgres: db_ver := max(token_credits.created_at) for applied rows across active users.
            - A user is charged at most once per version, tracked durably in Postgres via user_seen_versions.

        Performance:
            - Payload is cached in Redis keyed by version (best-effort).
            - If Redis is down, fall back to DB and still keep billing correct.

        Args:
            db: Async SQLAlchemy session.
            redis: Redis client (optional cache).
            user: Authenticated viewer user.
            action: ActionType that defines token cost (metadata charge).

        Returns:
            dict[str, Any]:
                {
                  "data": list[dict],   # list of TokenCreditResponse dicts
                  "charged": bool,
                  "balance": int
                }
        """

        resource = "tokens:all"
        list_key = "tokens:all:list"
        ver_key = "tokens:all:version"

        async with db.begin():
            db_ver_dt = await TCRepo.get_latest_created_at_all_users(db)
            if db_ver_dt is None:
                return {"data": [], "charged": False, "balance": user.tokens}

            db_ver = db_ver_dt.isoformat()

            first_time = await SVRepo.insert_seen(db, user_id=user.id, resource=resource, version=db_ver)
            if first_time:
                balance = await URepo.update_tokens(db, user.id, action.cost)
                charged = True
            else:
                balance = await URepo.get_tokens_by_id(db, user.id)
                charged = False

        data: list[dict] | None = None

        with suppress(RedisError, asyncio.TimeoutError, Exception):
            redis_ver = await CRepo.get_version(redis, ver_key)
            if redis_ver == db_ver:
                cached = await CRepo.get_list(redis, list_key)
                if cached is not None:
                    data = cached

        if data is None:
            rows = await TCRepo.get_all_users_tokens(db)
            data = [TokenCreditResponse.model_validate(dict(r)).model_dump(mode="json") for r in rows]

            with suppress(RedisError, asyncio.TimeoutError, Exception):
                await CRepo.set_list(redis, list_key, data)
                await CRepo.set_version(redis, ver_key, db_ver)

        try:
            log_action(
                event="user_viewed_all_users_token_history",
                user_id=user.id,
                username=user.username,
                action=action,
                charged=(action.cost if charged else 0),
                balance_after=balance,
            )
        except Exception as e:
            errors.exception("log_action failed: %r", e)

        return {"data": data, "charged": charged, "balance": balance}

    @staticmethod
    async def _mark_failed(db: AsyncSession, row_id: int) -> None:
        """
        Best-effort: mark the prediction row as failed in a short transaction.
        Never raises.
        """

        with suppress(SQLAlchemyError):
            async with db.begin():
                await TCRepo.mark_failed(db, row_id)
