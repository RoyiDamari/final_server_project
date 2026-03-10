import os
from contextlib import suppress
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.repositories.prediction_repository import PredictionRepository as PRepo
from app.repositories.token_credit_repository import TokenCreditRepository as TCRepo
from app.utils.files import safe_unlink, move_temp_to_final
from app.core.logging_config import activity, errors
from app.models.enums import RowStatus


# -----------------------------
# Path inspection (no dataclass)
# -----------------------------
def inspect_paths(final_path: Optional[str]) -> tuple[Optional[str], Optional[str], bool, bool]:
    """
    Args:
        final_path: Final artifact path stored in DB (maybe None/empty).

    Returns:
        tuple:
            (final_path, tmp_abs, final_exists, tmp_exists)
    """

    if not final_path:
        return None, None, False, False

    tmp_abs = f"{final_path}.tmp"

    final_exists = os.path.exists(final_path)
    tmp_exists = os.path.exists(tmp_abs)

    return final_path, tmp_abs, final_exists, tmp_exists


# -----------------------------
# Unified "mark failed" helper
# -----------------------------
async def mark_failed_best_effort(
        db: AsyncSession,
        kind: str,
        row_id: int,
        reason: str,
) -> None:
    """
    Best-effort mark a row as FAILED.

    Args:
        db: Async SQLAlchemy session.
        kind: "model" | "prediction" | "token_credit".
        row_id: Primary key of the row to mark failed.
        reason: Reason for logging.

    Returns:
        None. Never raises (DB errors are suppressed).
    """

    with suppress(SQLAlchemyError):
        async with db.begin():
            if kind == "model":
                await TMRepo.mark_failed(db, trained_model_id=row_id)
            elif kind == "prediction":
                await PRepo.mark_failed(db, pred_id=row_id)
            elif kind == "token_credit":
                await TCRepo.mark_failed(db, token_credit_id=row_id)
            else:
                return

    activity.warning("[reconciler] %s_mark_failed id=%s reason=%s", kind, row_id, reason)


# -----------------------------
# Pending -> failed (generic)
# -----------------------------
async def fail_all_pending(db: AsyncSession, kind: str) -> int:
    """
    Mark all PENDING rows of a given kind as FAILED.

    Args:
        db: Async SQLAlchemy session.
        kind: "prediction" | "token_credit".

    Returns:
        int: Number of rows found in PENDING and attempted to mark FAILED.
    """

    if kind == "prediction":
        pending_ids = await PRepo.list_ids_by_status(db, RowStatus.pending)
    elif kind == "token_credit":
        pending_ids = await TCRepo.list_ids_by_status(db, RowStatus.pending)
    else:
        return 0

    for rid in pending_ids:
        await mark_failed_best_effort(db, kind=kind, row_id=rid, reason="pending_at_startup")

    if pending_ids:
        activity.warning("[reconciler] %s_pending_failed count=%s", kind, len(pending_ids))
    else:
        activity.info("[reconciler] %s_pending_failed count=0", kind)

    return len(pending_ids)


# -----------------------------
# Trained model reconciliation helpers
# -----------------------------
async def finish_publish_or_fail(db: AsyncSession, tm_id: int, final_path: Optional[str]) -> None:
    """
    For APPLIED trained_models rows:
      - If final exists -> OK.
      - If final missing but tmp exists -> move tmp -> final.
      - If move fails OR both missing OR path missing -> mark FAILED and cleanup.

    Args:
        db: Async SQLAlchemy session.
        tm_id: TrainedModel.id
        final_path: TrainedModel.model_path (maybe None)

    Returns:
        None
    """

    final_path, tmp_path, final_exists, tmp_exists = inspect_paths(final_path)

    if not final_path:
        await mark_failed_best_effort(db, kind="model", row_id=tm_id, reason="no_model_path")
        return

    if final_exists:
        activity.info("[reconciler] model_applied_ok id=%s final=%s", tm_id, final_path)
        return

    if tmp_exists and tmp_path:
        try:
            move_temp_to_final(tmp_path, final_path)
            activity.warning(
                "[reconciler] model_publish_completed id=%s tmp=%s final=%s",
                tm_id, tmp_path, final_path
            )
            return
        except Exception as e:
            errors.exception("[reconciler] model_publish_move_failed id=%s err=%r", tm_id, e)
            await mark_failed_best_effort(db, kind="model", row_id=tm_id, reason="publish_move_failed")
            with suppress(Exception):
                safe_unlink(tmp_path)
                safe_unlink(final_path)
            return

    await mark_failed_best_effort(db, kind="model", row_id=tm_id, reason="artifact_missing")
    activity.warning("[reconciler] model_artifact_missing id=%s final=%s tmp=%s", tm_id, final_path, tmp_path)


async def fail_pending_model_and_clean_tmp(db: AsyncSession, tm_id: int, final_path: Optional[str]) -> None:
    """
    For PENDING trained_models rows at startup:
      - mark FAILED
      - delete tmp file best-effort

    Args:
        db: Async SQLAlchemy session.
        tm_id: TrainedModel.id
        final_path: TrainedModel.model_path (may be None)

    Returns:
        None
    """

    await mark_failed_best_effort(db, kind="model", row_id=tm_id, reason="pending_at_startup")

    final_path, tmp_path, _final_exists, _tmp_exists = inspect_paths(final_path)
    if tmp_path:
        with suppress(Exception):
            safe_unlink(tmp_path)
        activity.warning("[reconciler] model_tmp_deleted id=%s tmp=%s", tm_id, tmp_path)


# -----------------------------
# Trained model reconciliation helpers
# -----------------------------
def sweep_saved_models_tmp_files(base_dir: str) -> int:
    """
    Delete all '*.tmp' files under `base_dir` recursively.

    Assumes `base_dir` is an absolute path (recommended), typically passed from
    reconciler.py after normalization.

    Args:
        base_dir: Root directory containing model artifacts.

    Returns:
        int: Number of tmp files deleted.
    """
    if not os.path.isdir(base_dir):
        return 0

    deleted = 0
    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".tmp"):
                continue
            p = os.path.join(root, name)  # ✅ CHANGED (removed abspath)
            with suppress(Exception):
                safe_unlink(p)
                deleted += 1
    return deleted


def sweep_orphan_final_model_files(
    referenced_final_paths: set[str],
    base_dir: str,
) -> int:
    """
    Delete '*.pkl' files under `base_dir` that are NOT referenced by DB applied rows.

    Assumes:
        - DB stores model_path as relative paths like "saved_models/<uid>/<fp>.pkl"
        - reconciler passes absolute base_dir like "/app/saved_models"
        - we normalize referenced paths using os.path.abspath(p) which will use cwd=/app

    Args:
        referenced_final_paths: Set of model_path strings from DB.
        base_dir: Root directory containing model artifacts.

    Returns:
        int: Number of orphan final files deleted.
    """
    if not os.path.isdir(base_dir):
        return 0

    referenced_abs = {p for p in referenced_final_paths if p}

    deleted = 0
    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".pkl"):
                continue

            full_path = os.path.join(root, name)

            if full_path.endswith(".tmp"):
                continue

            if full_path not in referenced_abs:
                with suppress(Exception):
                    safe_unlink(full_path)
                    deleted += 1

    return deleted


def sweep_upload_tmp_dir(base_dir: str) -> int:
    """
    Delete all files directly under uploads temp directory.

    Args:
        base_dir: Upload temp directory (usually 'uploads/_tmp').

    Returns:
        int: Number of files deleted.
    """

    if not os.path.isdir(base_dir):
        return 0

    deleted = 0
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            with suppress(Exception):
                os.remove(p)
                deleted += 1
    return deleted
