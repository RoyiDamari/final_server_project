from typing import Any
from app.models.ml_models.base_model_strategy import BaseModelStrategy
from app.models.ml_models.concrete_strategy_classes import (
    LinearRegressionStrategy,
    LogisticRegressionStrategy,
    RandomForestStrategy,
)

MODEL_FACTORY = {
    "linear": LinearRegressionStrategy,
    "logistic": LogisticRegressionStrategy,
    "random_forest": RandomForestStrategy,
}


def get_model_strategy(
        model_type: str,
        features: list[str],
        label: str,
        model_params: dict[str, Any] | None,
) -> BaseModelStrategy:
    """
    Instantiate the appropriate model strategy by model_type.

    Args:
        model_type: Strategy key ("linear", "logistic", "random_forest").
        features: List of feature column names.
        label: Target column name.
        model_params: Hyperparameters dict (may include META keys like "kind"/"task").

    Returns:
        BaseModelStrategy: Concrete strategy instance.

    Raises:
        KeyError: If model_type is not present in MODEL_FACTORY.
    """

    return MODEL_FACTORY[model_type](model_type, features, label, model_params)
