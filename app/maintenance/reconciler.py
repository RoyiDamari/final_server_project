from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logs import errors
from app.repositories.train_model_repository import TrainModelRepository as TMRepo
from app.maintenance._helpers import (
    sweep_tmp_files,
    sweep_orphan_final_files,
    sweep_tmp_dir,
)

async def reconcile_files_on_startup(db: AsyncSession) -> None:
    # 1) DB snapshot (read-only) inside a tx for consistent view
    async with db.begin():
        referenced = await TMRepo.list_model_paths_applied(db)

    # 2) Filesystem sweeps (no DB)
    deleted_model_tmps = sweep_tmp_files(base_dir="saved_models")
    deleted_orphan_finals = sweep_orphan_final_files(referenced, base_dir="saved_models")

    deleted_upload_tmps = sweep_tmp_dir(base_dir="uploads/_tmp")

    errors.info(
        "[reconciler] model_tmp_deleted=%s orphan_finals_deleted=%s upload_tmp_deleted=%s",
        deleted_model_tmps,
        deleted_orphan_finals,
        deleted_upload_tmps,
    )
