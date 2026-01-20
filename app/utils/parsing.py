from app.exceptions.train_model import InvalidFormatException
from typing import Any
import json


def parse_json_object_strict(s: str, field: str = "model_params") -> dict[str, Any]:
    """
    Parse a JSON object string like '{"alpha":0.1}' into a dict.
    Raises InvalidFormatException on bad JSON or non-object JSON.
    """
    try:
        v = json.loads(s)
    except Exception:
        raise InvalidFormatException(f"{field} must be a JSON object string like {{\"alpha\":0.1}}")
    if not isinstance(v, dict):
        raise InvalidFormatException(f"{field} must be a JSON object")
    return v

def parse_json_list_strict(s: str, *, field: str = "features") -> list[str]:
    """
    Parse a JSON array of strings like '["age","price"]' into a list[str].
    Raises InvalidFormatException on bad JSON or wrong element types.
    """
    try:
        v = json.loads(s)
    except Exception:
        raise InvalidFormatException(f"{field} must be a JSON array string like [\"age\",\"price\"]")
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise InvalidFormatException(f"{field} must be a JSON array of strings")
    return v