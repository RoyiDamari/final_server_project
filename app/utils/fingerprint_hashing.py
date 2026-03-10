import hashlib
import inspect
import importlib
import json
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any


# ---------- helpers ----------
def file_sha256(path: str) -> str:
    """
    Compute a streaming SHA-256 hash of a file's bytes.

    Args:
        path: Filesystem path to the file.

    Returns:
        str: Hex-encoded SHA-256 digest of the file contents.

    Raises:
        OSError: If the file cannot be opened/read.
    """

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    """
    Serialize an object to deterministic JSON (sorted keys, compact separators).

    Args:
        obj: JSON-serializable object.

    Returns:
        str: Deterministic JSON string.

    Raises:
        TypeError: If obj is not JSON-serializable.
    """

    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------- code & lockfile hashes (cached once per process) ----------
@lru_cache(maxsize=1)
def model_code_hash() -> str:
    """
    Compute a hash of the concrete strategy source code (cached per process).

    The hash changes when the source code of the concrete strategies changes.
    If the module or source cannot be loaded, returns a sentinel value.

    Returns:
        str: Hex-encoded SHA-256 digest of normalized source code, or "no-src".
    """

    if find_spec("app.models.ml_models.concrete_strategy_classes") is None:
        return "no-src"

    try:
        mod = importlib.import_module("app.models.ml_models.concrete_strategy_classes")
    except (ImportError, RuntimeError, ValueError):
        return "no-src"

    try:
        src = inspect.getsource(mod)
    except (OSError, TypeError):
        return "no-src"

    norm = "\n".join(line.rstrip() for line in src.splitlines())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def lockfile_sha(path: str = "requirements.txt") -> str:
    """
    Compute a hash of the dependency lockfile (cached per process).

    Args:
        path: Filesystem path to the requirements/lock file.

    Returns:
        str: Hex-encoded SHA-256 digest of the file, or "no-lock" if missing.

    Raises:
        OSError: If the file exists but cannot be read.
    """

    p = Path(path)
    if not p.exists():
        return "no-lock"
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------- main fingerprint ----------
def compute_training_fingerprint(
        csv_file_path: str,
        sorted_features_clean: list[str],
        label_clean: str,
        model_type_clean: str,
        params_norm: dict,
        requirements_file_path: str = "requirements.txt",
) -> str:
    """
    Compute a stable fingerprint for a training request.

    Fingerprint inputs:
      - CSV file bytes hash
      - normalized features/label/model_type/params
      - pipeline version (code hash + requirements hash)

    Args:
        csv_file_path: Path to the validated CSV file.
        sorted_features_clean: Sorted list of feature column names.
        label_clean: Normalized label column name.
        model_type_clean: Normalized model type name.
        params_norm: Normalized hyperparameters used in the pipeline.
        requirements_file_path: Path to dependency lockfile.

    Returns:
        str: Hex-encoded SHA-256 fingerprint for idempotency.

    Raises:
        OSError: If the CSV file cannot be read for hashing.
        TypeError: If params_norm is not JSON-serializable.
    """

    pipeline_version = f"{model_code_hash()}|lock={lockfile_sha(requirements_file_path)}"

    parts = {
        "data_sha256": file_sha256(csv_file_path),
        "features": sorted_features_clean,
        "label": label_clean,
        "model_type": model_type_clean,
        "params": params_norm,
        "pipeline_version": pipeline_version,
    }
    return hashlib.sha256(stable_json(parts).encode("utf-8")).hexdigest()


def compute_prediction_fingerprint(
        model_id: int,
        feature_values: dict[str, Any],
) -> str:
    """
    Compute a stable fingerprint for a prediction request.

    Args:
        model_id: Trained model id used for prediction.
        feature_values: Feature-name -> value mapping for the request.

    Returns:
        str: Hex-encoded SHA-256 fingerprint for idempotency.

    Raises:
        TypeError: If feature_values contains non-JSON-serializable values.
    """

    canonical = {
        "model_id": model_id,
        "features": sorted(feature_values.items()),
    }
    payload = stable_json(canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
