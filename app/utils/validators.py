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
    Normalize a feature list by stripping whitespace, dropping empty strings,
    and removing duplicates while preserving the original order.

    Args:
        features: Raw list of feature names.

    Returns:
        list[str]: Cleaned, order-preserving list of unique feature names.
    """

    normalize = [s for s in (f.strip() for f in features) if s]
    cleaned = list(dict.fromkeys(normalize))
    return cleaned


def normalize_params(params: dict | None) -> dict:
    """
    Normalize a hyperparameter dict:
      - None -> {}
      - strip whitespace from keys
      - forbid empty/blank keys

    Args:
        params: Raw hyperparameters dict or None.

    Returns:
        dict: Normalized hyperparameters dict.

    Raises:
        MissingDataException: If a hyperparameter key is empty after stripping.
    """

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
    """
    Build the canonical "meta" params used in fingerprinting.

    This injects strategy-level meta fields (like "kind" or "task") based on the
    submitted params, to ensure the fingerprint changes when those semantics change.

    Args:
        model_type: Normalized model type (e.g., "linear", "random_forest").
        params_norm: Validated estimator params (what will be applied to the estimator).
        params_n: Normalized submitted params (may include meta keys like "kind"/"task").

    Returns:
        dict: A dict used as part of the fingerprint payload.
    """

    out = dict(params_norm)
    if model_type == "linear":
        out["kind"] = str(params_n.get("kind", "ols")).strip().lower()
    elif model_type == "random_forest":
        out["task"] = str(params_n.get("task", "auto")).strip().lower()
    return out


def ensure_csv_valid(csv_path: str) -> pd.DataFrame:
    """
    Load and validate an uploaded CSV file.

    Validation rules:
        - must be readable by pandas
        - must not be empty
        - must not contain duplicate column names

    Args:
        csv_path: Filesystem path to the uploaded CSV.

    Returns:
        pandas.DataFrame: Loaded dataframe.

    Raises:
        InvalidFormatException: If file is not a valid CSV or has duplicate columns.
        MissingDataException: If the CSV loads but is empty.
    """

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
    """
    Normalize and validate the requested model type.

    Args:
        model_type: Raw model type string from user input.

    Returns:
        str: Normalized model type (lowercased, stripped).

    Raises:
        MissingDataException: If model_type is blank.
        UnsupportedModelTypeException: If model_type is not supported by MODEL_FACTORY.
    """

    mt = (model_type or "").strip().lower()
    if not mt:
        raise MissingDataException("model_type must be provided.")
    if mt not in MODEL_FACTORY:
        raise UnsupportedModelTypeException(f"Unsupported model_type: {mt}")
    return mt


def ensure_label_valid(df: pd.DataFrame, label: str) -> str:
    """
    Validate the label column name.

    Args:
        df: Training dataframe.
        label: Raw label name from user input.

    Returns:
        str: Cleaned label name.

    Raises:
        MissingDataException: If label is blank.
        InvalidLabelException: If label column does not exist in df.
    """

    lab = (label or "").strip()
    if not lab:
        raise MissingDataException("You must select Label")
    if lab not in df.columns:
        raise InvalidLabelException(f"Label column '{lab}' not found in dataset")
    return lab


def ensure_features_valid(df: pd.DataFrame, features: list[str], label: str) -> list[str]:
    """
    Normalize + validate feature names against the dataframe columns.

    Rules:
      - require at least one feature
      - forbid the label from appearing in features
      - require all features to exist in df.columns

    Args:
        df: Training dataframe.
        features: Raw list of feature names from user input.
        label: Label column name (raw or cleaned).

    Returns:
        list[str]: Normalized, de-duplicated feature list (order-preserving).

    Raises:
        MissingDataException: If no features remain after normalization.
        InvalidFeatureException: If label appears in features or a feature is missing from df.
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
    Validate submitted hyperparameters against the estimator behind a strategy.

    - Removes any strategy META_KEYS from the submitted dict.
    - Builds the pipeline and extracts the estimator from pipe.named_steps["model"].
    - Rejects unknown hyperparameter names.

    Args:
        strategy: Strategy instance that provides build_pipeline(df) and optionally META_KEYS.
        params: Submitted hyperparameter dict.
        df: Optional dataframe used by build_pipeline (for pipelines that depend on data).

    Returns:
        dict: Filtered dict containing only estimator parameters.

    Raises:
        InvalidParamException: If the pipeline lacks a "model" step or unknown params are found.
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
    """
    Predicate: value is a positive int/float.

    Args:
        v: Any value.

    Returns:
        bool: True if v is int/float and > 0.
    """

    return isinstance(v, (int, float)) and v > 0


def is_non_negative(v):
    """
    Predicate: value is a positive int/float.

    Args:
        v: Any value.

    Returns:
        bool: True if v is int/float and > 0.
    """

    return isinstance(v, (int, float)) and v >= 0


def in_range_0_1(v):
    """
    Predicate: value is an int/float in [0, 1].

    Args:
        v: Any value.

    Returns:
        bool: True if v is int/float and between 0 and 1 inclusive.
    """

    return isinstance(v, (int, float)) and 0 <= v <= 1


def is_bool(v):
    """
    Predicate: value is an int/float in [0, 1].

    Args:
        v: Any value.

    Returns:
        bool: True if v is int/float and between 0 and 1 inclusive.
    """

    return isinstance(v, bool)


def one_of(*values):
    """
    Build a predicate that checks membership in a fixed set of allowed values.

    Args:
        *values: Allowed values.

    Returns:
        Callable[[Any], bool]: Function that returns True if input is in values.
    """

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
    """
    Enforce semantic constraints for logistic regression hyperparameters.

    Validates solver compatibility with penalty type.

    Args:
        params: Estimator params dict.

    Raises:
        InvalidParamException: If solver is incompatible with penalty.
    """

    penalty = params.get("penalty", "l2")
    solver = params.get("solver", "lbfgs")

    allowed = _VALID_SOLVERS.get(penalty)
    if allowed and solver not in allowed:
        raise InvalidParamException(
            f"solver='{solver}' is not compatible with penalty='{penalty}'"
        )


def validate_param_values(model_type: str, params: dict[str, Any]) -> None:
    """
    Centralized parameter value validation.

    Performs:
      1) Value validation using PARAM_RULES (type/range checks).
      2) Semantic validation for certain models (e.g., logistic solver/penalty compatibility).

    Args:
        model_type: Normalized model type.
        params: Estimator params dict (already filtered to estimator params).

    Returns:
        None

    Raises:
        InvalidParamException: On the first invalid parameter value or semantic mismatch.
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
