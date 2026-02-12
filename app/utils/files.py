import os
import tempfile
import joblib
from fastapi import UploadFile
from typing import Any
from contextlib import suppress
from app.models.orm_models.users import User
from app.exceptions.artifact import ArtifactWriteException



def save_upload_to_temp_csv(upload: UploadFile, suffix) -> str:
    """
    Persist an UploadFile to a temporary file on disk and return its path.

    Args:
        upload: FastAPI UploadFile object.
        suffix: Desired filename suffix for the temp file (default ".csv").

    Returns:
        Absolute path to the temp file (caller is responsible for deletion).
    """
    base = os.path.join("uploads", "_tmp")
    os.makedirs(base, exist_ok=True)

    fd, path = tempfile.mkstemp(dir=base, suffix=suffix)

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


def unique_model_path(user: User, fp, dirpath: str = "saved_models") -> str:
    """
    Build a unique model artifact path by normalizing the name and appending a timestamp.

    Example: saved_models/price_model_1723650000.pkl

    Args:
        user: Authenticated user.
        fp: fingerprint of the file
        dirpath: Target directory for artifacts (created if missing).

    Returns:
        Full path to a unique .pkl file under dirpath.
    """
    base = os.path.join(dirpath, str(user.id))
    os.makedirs(base, exist_ok=True)  # <-- ensure folder exists
    return os.path.join(base, f"{fp}.pkl")


def temp_path_for(final_path: str) -> str:
    return f"{final_path}.tmp"


def move_temp_to_final(tmp_path: str, final_path: str) -> None:
    try:
        os.replace(tmp_path, final_path)
    except OSError as e:
        raise ArtifactWriteException(
            log_detail=f"move failed tmp={tmp_path!r} final={final_path!r} errno={getattr(e, 'errno', None)} msg={e}"
        ) from e


def safe_unlink(path: str | None) -> None:
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
        Exception: Any joblib/pickle errors during load.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)