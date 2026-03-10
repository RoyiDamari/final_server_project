from app.exceptions.train_model import InvalidFormatException
from typing import Any
import json


def parse_json_object_strict(s: str, field: str = "model_params") -> dict[str, Any]:
    """
    Parse a JSON object string into a Python dict.

    Args:
        s: Raw JSON string (e.g. '{"alpha": 0.1}').
        field: Field name for error messages.

    Returns:
        dict[str, Any]: Parsed JSON object.

    Raises:
        InvalidFormatException: If JSON is invalid or the parsed value is not a dict.
    """

    try:
        v = json.loads(s)
    except Exception:
        raise InvalidFormatException(f"{field} must be a JSON object string like {{\"alpha\":0.1}}")
    if not isinstance(v, dict):
        raise InvalidFormatException(f"{field} must be a JSON object")
    return v


def parse_json_list_strict(s: str, field: str = "features") -> list[str]:
    """
    Parse a JSON array-of-strings into a Python list[str].

    Args:
        s: Raw JSON string (e.g. '["age", "price"]').
        field: Field name for error messages.

    Returns:
        list[str]: Parsed list of strings.

    Raises:
        InvalidFormatException: If JSON is invalid, not a list, or contains non-strings.
    """

    try:
        v = json.loads(s)
    except Exception:
        raise InvalidFormatException(f"{field} must be a JSON array string like [\"age\",\"price\"]")
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise InvalidFormatException(f"{field} must be a JSON array of strings")
    return v
