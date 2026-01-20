from typing import Mapping, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.token_credit_repository import TokenCreditRepository as TCRepo
from app.exceptions.token_credit import PurchaseInProgressException, BalanceMustBeZeroException
from app.repositories.user_repository import UserRepository as UserRepo
from app.models.orm_models.users import User
from app.models.pydantic_models.token_credit import BuyTokensResponse
from app.models.enums import RowStatus
from app.core.logs import log_action


class TokenCreditService:
    @staticmethod
    async def buy_tokens(db: AsyncSession, user: User, amount: int, key: str | UUID) -> BuyTokensResponse:
        """
        Idempotent token *purchase* (credit).

        Flow:
          1) Try to insert a 'pending' credit row for (user_id, key) exactly once.
             - If inserted: this is the first attempt for this key → proceed to credit.
             - If not inserted: it’s either a duplicate key or another credit is pending.
          2) On first-time application:
             - Credit the user *only if* current tokens == 0.
             - Mark the credit 'applied' with the resulting balance.
          3) On duplicate:
             - If that key is already 'applied', return its recorded balance (same key → same response).
             - If 'failed', raise BalanceMustBeZeroException (enforce zero-balance policy).
             - Otherwise, raise PurchaseInProgressException.

        Transactions & logging:
          - All DB writes occur inside a single transaction.
          - After commit, if we actually applied the purchase now, logs "tokens_purchased" event with:
            'credited=amount', 'balance_after=result_balance'

        Args:
            db: Async SQLAlchemy session.
            user: Authenticated user ORM instance (caller/buyer).
            amount: Number of tokens to credit (already validated upstream).
            key: Idempotency key for this purchase attempt.

        Returns:
            int: The resulting token balance for the user (idempotent per key).

        Raises:
            BalanceMustBeZeroException: Current tokens were not 0 on first-time application.
            PurchaseInProgressException: Another purchase is in progress (pending) for this user/key.
        """
        key = str(key)
        applied_now = False
        result_balance: int


        inserted = await TCRepo.try_insert_pending(db, user.id, key)

        if inserted:
            new_balance = await UserRepo.add_tokens(db, user.id, amount)
            if new_balance is None:
                await TCRepo.mark_failed(db, user.id, key)
                raise BalanceMustBeZeroException()

            await TCRepo.mark_applied(db, user.id, key, new_balance)
            result_balance = new_balance
            applied_now = True

        else:
            info = await TCRepo.get_by_key_status_open_balance(db, user.id, key)
            if info and info["status"] == RowStatus.applied and info["open_balance"] is not None:
                result_balance = info["open_balance"]
            elif info and info["status"] == RowStatus.failed:
                raise BalanceMustBeZeroException()
            else:
                raise PurchaseInProgressException()

        if applied_now and result_balance is not None:
            log_action(
                "tokens_purchased",
                user_id=user.id,
                username=user.username,
                credited=amount,
                balance_after=result_balance,
            )

        return BuyTokensResponse(
            message=f"{result_balance} tokens has been added.",
            balance=result_balance,
        )

    @staticmethod
    async def get_user_token_history(db: AsyncSession, user: User) -> list[Mapping[str, Any]]:
        """
        Return the authenticated user's token credit history.

        - No token charge
        - No side effects
        - Empty list if no history exists
        """

        user_token_history = await TCRepo.get_user_token_history(db, user.id)

        log_action(
            event="user_viewed_his_tokens_history",
            user_id=user.id,
            username=user.username,
            charged=False,
        )

        return user_token_history

