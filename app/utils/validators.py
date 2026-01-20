import pandas as pd
from typing import Dict, Set, Any
from app.exceptions.train_model import (
    InvalidFormatException,
    MissingDataException,
    InvalidFeatureException,
    InvalidLabelException,
    InvalidParamException,
    UnsupportedModelTypeException,
)
from app.models.ml_models.model_strategy_factory import MODEL_FACTORY


def normalize_features(features: list[str]) -> list[str]:
    """
    Strip whitespace, drop empties, and de-duplicate while preserving order.
    Assumes `features` is already a list of strings (validated at parsing).
    """
    normalize = [s for s in (f.strip() for f in features) if s]
    cleaned = list(dict.fromkeys(normalize))
    return cleaned


def normalize_params(params: dict | None) -> dict:
    """None → {}; trim keys; forbid empty keys. Values kept as-is."""
    if params is None:
        return {}

    out: Dict[str, Any] = {}
    for k, v in params.items():
        key = str(k).strip()
        if not key:
            raise MissingDataException("Empty hyperparameter name is not allowed")
        out[key] = v
    return out


def normalize_meta_for_fingerprint(model_type: str, params_norm: dict, params_n: dict) -> dict:
    out = dict(params_norm)
    if model_type == "linear":
        out["kind"] = str(params_n.get("kind", "ols")).strip().lower()
    elif model_type == "random_forest":
        out["task"] = str(params_n.get("task", "auto")).strip().lower()
    return out


def ensure_csv_valid(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        raise InvalidFormatException("Uploaded file is not a valid CSV")
    if df is None or df.empty:
        raise MissingDataException("Uploaded CSV is empty")
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        raise InvalidFormatException("CSV has duplicate column names")
    return df


def ensure_model_type_valid(model_type: str) -> str:
    mt = (model_type or "").strip().lower()
    if not mt:
        raise MissingDataException("model_type must be provided.")
    if mt not in MODEL_FACTORY:
        raise UnsupportedModelTypeException(f"Unsupported model_type: {mt}")
    return mt


def ensure_label_valid(df: pd.DataFrame, label: str) -> str:
    lab = (label or "").strip()
    if not lab:
        raise MissingDataException("You must select Label")
    if lab not in df.columns:
        raise InvalidLabelException(f"Label column '{lab}' not found in dataset")
    return lab


def ensure_features_valid(df: pd.DataFrame, features: list[str], label: str) -> list[str]:
    """
    Normalize + validate feature names against the dataframe columns.
    - require at least one feature
    - forbid the label from appearing in features
    - ensure every feature exists in df.columns
    Returns the normalized, de-duplicated (order-preserving) feature list.
    """
    cleaned = normalize_features(features)

    if not cleaned:
        raise MissingDataException("You must select at least one feature")

    label_clean = (label or "").strip()
    if label_clean in cleaned:
        raise InvalidFeatureException("Label must not appear in the feature list")

    missing = [f for f in cleaned if f not in df.columns]
    if missing:
        if len(missing) == 1:
            raise InvalidFeatureException(f"Feature '{missing[0]}' not found in dataset")
        raise InvalidFeatureException("Features not found in dataset: " + ", ".join(missing))

    return cleaned


def ensure_params_valid(strategy, params: dict, df=None) -> dict:
    """
    Normalize, then validate against the estimator behind the strategy.
    Allows empty dict (use estimator defaults).
    """
    submitted = dict(params)
    for k in getattr(strategy, "META_KEYS", ()):
        submitted.pop(k, None)

    pipe = strategy.build_pipeline(df)
    try:
        est = pipe.named_steps["model"]
    except KeyError:
        raise InvalidParamException("Pipeline is missing a 'model' step")

    allowed: Set[str] = set(est.get_params().keys())
    unknown = set(submitted.keys()) - allowed
    if unknown:
        raise InvalidParamException("Unknown hyperparameters: " + ", ".join(sorted(unknown)))

    return submitted


def is_positive(v):
    return isinstance(v, (int, float)) and v > 0


def is_non_negative(v):
    return isinstance(v, (int, float)) and v >= 0


def in_range_0_1(v):
    return isinstance(v, (int, float)) and 0 <= v <= 1


def is_bool(v):
    return isinstance(v, bool)


def one_of(*values):
    return lambda v: v in values


PARAM_RULES = {
    "logistic": {
        "C": is_positive,
        "l1_ratio": in_range_0_1,
        "max_iter": lambda v: isinstance(v, int) and v > 0,
        "solver": one_of("lbfgs", "liblinear", "saga", "newton-cg"),
        "penalty": one_of("l1", "l2", "elasticnet", "none"),
        "fit_intercept": is_bool,
    },

    "linear": {
        "alpha": is_non_negative,
        "l1_ratio": in_range_0_1,
        "fit_intercept": is_bool,
    },

    "random_forest": {
        "n_estimators": lambda v: isinstance(v, int) and v > 0,
        "max_depth": lambda v: v is None or (isinstance(v, int) and v > 0),
        "n_jobs": lambda v: isinstance(v, int),
        "random_state": lambda v: isinstance(v, int),
    },
}

_VALID_SOLVERS = {
    "l2": {"lbfgs", "newton-cg", "saga", "liblinear"},
    "l1": {"liblinear", "saga"},
    "elasticnet": {"saga"},
    "none": {"lbfgs", "newton-cg", "saga"},
}


def _validate_logistic_semantics(params: dict[str, Any]) -> None:
    penalty = params.get("penalty", "l2")
    solver = params.get("solver", "lbfgs")

    allowed = _VALID_SOLVERS.get(penalty)
    if allowed and solver not in allowed:
        raise InvalidParamException(
            f"solver='{solver}' is not compatible with penalty='{penalty}'"
        )


def validate_param_values(model_type: str, params: dict[str, Any]) -> None:
    """
    Centralized parameter validation entrypoint.

    - Validates parameter values
    - Dispatches semantic validation per model
    - Raises InvalidParamException on first error
    """

    # ---------- Value validation ----------
    rules = PARAM_RULES.get(model_type, {})
    for key, value in params.items():
        rule = rules.get(key)
        if rule and not rule(value):
            raise InvalidParamException(
                f"Invalid value for '{key}': {value}"
            )

    # ---------- Semantic validation ----------
    if model_type == "logistic":
        _validate_logistic_semantics(params)
