import os
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.maintenance._helpers import (finish_publish_or_fail, fail_pending_model_and_clean_tmp,
                                      fail_all_pending, sweep_orphan_final_model_files,
                                      sweep_saved_models_tmp_files, sweep_upload_tmp_dir)
from app.models.enums import RowStatus
from app.core.logs import activity
from app.config import config


async def reconcile_files_on_startup(db: AsyncSession) -> None:
    """
    Startup reconciliation to restore invariants after crashes/reloads.

    Steps:
      1) Trained models:
         - For APPLIED rows: ensure final artifact exists; if only tmp exists, finalize publish.
         - For PENDING rows: mark FAILED and remove tmp.

      2) Predictions:
         - Mark all PENDING predictions as FAILED.

      3) Token credits:
         - Mark all PENDING token_credits as FAILED.

    Args:
        db: Async SQLAlchemy session.

    Returns:
        None
    """

    # -----------------------------
    # 1) trained_models reconciliation
    # -----------------------------
    applied_models = await TMRepo.list_ids_and_paths_by_status(db, RowStatus.applied)
    for tm_id, path in applied_models:
        await finish_publish_or_fail(db, tm_id=tm_id, final_path=path)

    pending_models = await TMRepo.list_ids_and_paths_by_status(db, RowStatus.pending)
    for tm_id, path in pending_models:
        await fail_pending_model_and_clean_tmp(db, tm_id=tm_id, final_path=path)

    # -----------------------------
    # 2) predictions pending -> failed
    # -----------------------------
    pred_failed = await fail_all_pending(db, kind="prediction")

    # -----------------------------
    # 3) token_credits pending -> failed
    # -----------------------------
    tc_failed = await fail_all_pending(db, kind="token_credit")

    # -----------------------------
    # 4) referenced set (for orphan sweep)
    # IMPORTANT: uses your existing repo function which filters ACTIVE + APPLIED.
    # -----------------------------
    referenced_paths = await TMRepo.list_model_paths_applied(db)

    # -----------------------------
    # 5) filesystem sweeps
    # -----------------------------
    base_models_dir = os.path.abspath(config.MODEL_BASE_DIR)
    base_csv_dir = os.path.abspath(config.UPLOAD_TMP_DIR)

    deleted_model_tmps = sweep_saved_models_tmp_files(base_dir=base_models_dir)
    deleted_orphan_finals = sweep_orphan_final_model_files(referenced_paths, base_dir=base_models_dir)
    deleted_upload_tmps = sweep_upload_tmp_dir(base_dir=base_csv_dir)

    activity.warning(
        "[reconciler] applied_checked=%s pending_failed=%s pred_failed=%s tc_failed=%s "
        "model_tmp_deleted=%s orphan_finals_deleted=%s upload_tmp_deleted=%s",
        len(applied_models),
        len(pending_models),
        pred_failed,
        tc_failed,
        deleted_model_tmps,
        deleted_orphan_finals,
        deleted_upload_tmps,
    )