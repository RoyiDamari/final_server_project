from .concrete_strategy_classes import (
    LinearRegressionStrategy,
    LogisticRegressionStrategy,
    RandomForestStrategy,
)


MODEL_FACTORY = {
    "linear": LinearRegressionStrategy,
    "logistic": LogisticRegressionStrategy,
    "random_forest": RandomForestStrategy,
}

def get_model_strategy(model_type, features, label, model_params):
    return MODEL_FACTORY[model_type](model_type, features, label, model_params)
