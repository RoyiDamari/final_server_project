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
    """Streaming SHA-256 over file contents (1MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    """Deterministic JSON string (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------- code & lockfile hashes (cached once per process) ----------
@lru_cache(maxsize=1)
def model_code_hash() -> str:
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
    Short hash of the dependency lockfile. Guarantees that environment
    changes (versions) are captured in pipeline_version.
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
    Build a stable fingerprint over:
      - CSV file contents (sha256 of bytes)
      - normalized features/label/model_type/params
      - pipeline_version := {model_code_hash()}|lock={requirements.txt hash}

    Any change to data, metadata, concrete strategy code, or requirements.txt
    changes the fingerprint.
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
    canonical = {
        "model_id": model_id,
        "features": sorted(feature_values.items()),
    }
    payload = stable_json(canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
