from ui.api.base import api_call

def get_model_type_distribution(token: str):
    return api_call(
        "/usage/model_type_distribution",
        method="GET",
        token=token,
    )

def get_regression_vs_classification_split(token: str):
    return api_call(
        "/usage/type_split",
        method="GET",
        token=token,
    )

def get_label_distribution(token: str):
    return api_call(
        "/usage/label_distribution",
        method="POST",
        token=token,
    )

def get_metric_distribution(token: str):
    return api_call(
        "/usage/metric_distribution",
        method="POST",
        token=token,
    )
