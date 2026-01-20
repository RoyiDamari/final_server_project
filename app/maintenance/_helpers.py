import os
from contextlib import suppress
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logs import errors
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.utils.files import move_temp_to_final, safe_unlink
from app.exceptions.train_model import ArtifactWriteException


async def mark_failed_safely(db: AsyncSession, tm_id: int, reason: str) -> None:
    """Best-effort: flip row to failed in its own short tx, with a small log."""
    with suppress(Exception):
        async with db.begin():
            await TMRepo.mark_failed(db, trained_model_id=tm_id)
    errors.info("[reconciler] marked_failed tm_id=%s reason=%s", tm_id, reason)


def _inspect_paths(final_path: Optional[str]) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """
    Return (final_path, tmp_path, final_exists, tmp_exists).

    If final_path is falsy/None, tmp_path is None too and both exists flags are False.
    """
    if not final_path:
        return None, None, False, False

    final_exists = os.path.exists(final_path)
    tmp_path = f"{final_path}.tmp"
    tmp_exists = os.path.exists(tmp_path)
    return final_path, tmp_path, final_exists, tmp_exists


async def finish_publish_or_fail(db: AsyncSession, tm_id: int, final_path: Optional[str]) -> None:
    """
    For rows already marked 'applied':
      - If final exists → nothing to do.
      - If final missing but tmp exists → try to atomically move tmp→final; on error, mark failed + cleanup.
      - If both missing or path is None → mark failed.
    """
    final_path, tmp_path, final_exists, tmp_exists = _inspect_paths(final_path)

    if not final_path:
        await mark_failed_safely(db, tm_id, reason="no_final_path")
        return

    if final_exists:
        errors.info("[reconciler] final_exists tm_id=%s path=%s", tm_id, final_path)
        return

    if tmp_exists and tmp_path:
        try:
            move_temp_to_final(tmp_path, final_path)
            errors.info("[reconciler] publish_completed tm_id=%s tmp=%s final=%s", tm_id, tmp_path, final_path)
        except ArtifactWriteException as e:
            await mark_failed_safely(db, tm_id, reason="publish_move_failed")
            with suppress(Exception):
                safe_unlink(tmp_path)
                safe_unlink(final_path)
            errors.warning("[reconciler] publish_cleanup tm_id=%s tmp=%s final=%s err=%s",
                           tm_id, tmp_path, final_path, e)
    else:
        await mark_failed_safely(db, tm_id, reason="artifact_missing")
        errors.warning("[reconciler] artifact_missing tm_id=%s final=%s tmp=%s",
                       tm_id, final_path, (tmp_path or "<none>"))


async def fail_pending_and_clean_tmp(db: AsyncSession, tm_id: int, final_path: Optional[str]) -> None:
    """Flip any pending to failed and delete leftover .tmp best-effort, with logs."""
    await mark_failed_safely(db, tm_id, reason="pending_at_startup")
    tmp = f"{final_path}.tmp" if final_path else None
    if tmp:
        with suppress(Exception):
            safe_unlink(tmp)
        errors.info("[reconciler] tmp_cleanup tm_id=%s tmp=%s", tm_id, tmp)
