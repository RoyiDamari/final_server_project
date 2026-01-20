from app.core.logging_config import activity, errors
from typing import Any


def level_for_status(code: int):
    """
    Map HTTP status → logging function on `errors`.
    - 5xx:   errors.error
    - 404/405: errors.info (common, not an 'error')
    - 429/409: errors.warning (throttling/conflicts to watch)
    - else:  errors.info
    """
    if code >= 500:       return errors.error
    if code in (404,405): return errors.info
    if code in (429,409): return errors.warning
    return errors.info

def _format_kv(fields: dict[str, Any]) -> str:
    parts = []
    for k, v in fields.items():
        if isinstance(v, str):
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)

def log_action(event: str, user_id: int, username: str, **fields: Any) -> None:
    """
    Activity log: bake fields into the message so they appear with the plain formatter.
    (You chose not to render `extra` via the formatter, so we keep it human-friendly here.)
    """
    base = {"user_id": user_id, "username": username}
    base.update(fields)
    activity.info("%s %s", event, _format_kv(base))


