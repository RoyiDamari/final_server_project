import os
import tempfile
import joblib
from fastapi import UploadFile
from typing import Any
from contextlib import suppress
from app.models.orm_models.users import User
from app.exceptions.artifact import ArtifactWriteException
from app.config import config


def save_upload_to_temp_csv(upload: UploadFile, suffix: str) -> str:
    """
    Persist an UploadFile to a temporary file on disk and return its path.

    Args:
        upload: FastAPI UploadFile object.
        suffix: Desired filename suffix for the temp file (default ".csv").

    Returns:
        Absolute path to the temp file (caller is responsible for deletion).
    """

    base_abs = os.path.abspath(config.UPLOAD_TMP_DIR)
    os.makedirs(base_abs, exist_ok=True)

    fd, path = tempfile.mkstemp(dir=base_abs, suffix=suffix)

    try:
        with os.fdopen(fd, "wb") as f:
            upload.file.seek(0)
            while True:
                chunk = upload.file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
        return path
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def unique_model_path(user: User, fp: str) -> str:
    """
    Build a unique model artifact path by normalizing the name and appending a timestamp.

    Example: saved_models/price_model_1723650000.pkl

    Args:
        user: Authenticated user.
        fp: fingerprint of the file

    Returns:
        Full path to a unique .pkl file under dirpath.
    """

    base_dir = os.path.abspath(config.MODEL_BASE_DIR)
    user_dir = os.path.join(base_dir, str(user.id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f"{fp}.pkl")


def temp_path_for(final_path: str) -> str:
    """
    Compute the temporary artifact path for a final model artifact path.

    Args:
        final_path: Path where the final artifact should live (e.g. ".../fp.pkl").

    Returns:
        str: Temporary path used during training (e.g. ".../fp.pkl.tmp").
    """

    return f"{final_path}.tmp"


def move_temp_to_final(tmp_path: str, final_path: str) -> None:
    """
    Atomically replace the final artifact with the tmp artifact.

    Uses os.replace(), which is atomic on POSIX when source/target are on the same filesystem.
    This prevents partially-written final files because the rename/replace happens as a single
    filesystem operation.

    Args:
        tmp_path: Path to the temporary artifact file.
        final_path: Path to the final artifact file.

    Returns:
        None

    Raises:
        ArtifactWriteException: If the OS rename/replace fails (permissions, missing file, etc.).
    """

    try:
        os.replace(tmp_path, final_path)
    except OSError as e:
        raise ArtifactWriteException(
            log_detail=f"move failed tmp={tmp_path!r} final={final_path!r} errno={getattr(e, 'errno', None)} msg={e}"
        ) from e


def safe_unlink(path: str | None) -> None:
    """
    Best-effort file deletion.

    Deletes the file at `path` if it exists. Never raises for common cleanup failures.

    Args:
        path: File path to delete. If None/empty, does nothing.

    Returns:
        None
    """

    if not path:
        return
    with suppress(FileNotFoundError, IsADirectoryError, PermissionError):
        os.remove(path)


def load_joblib_model(path: str) -> Any:
    """
    Load a joblib-serialized object from `path`.

    Args:
        path: Absolute or relative filesystem path to a .pkl file.

    Returns:
        The deserialized object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)
