import requests
import streamlit as st
from typing import Optional, Dict, Any
from requests.exceptions import RequestException, Timeout
from ui.utils.api_helpers import logout_and_stop
from ui.config import API_BASE_URL


def get_auth_headers(token: Optional[str]) -> Dict[str, Any]:
    """
    Build Authorization headers for API calls.

    Args:
        token: Bearer token string, or None.

    Returns:
        A headers dict containing Authorization if token is provided; otherwise an empty dict.
    """

    return {"Authorization": f"Bearer {token}"} if token else {}


def api_call(
        endpoint: str,
        method: str = "GET",
        token: str | None = None,
        allow_refresh: bool = True,
        has_retried: bool = False,
        **kwargs
) -> dict[str, Any] | None:
    """
    Unified HTTP request wrapper for all backend endpoints.

    Responsibilities:
        - Resolve access token from st.session_state if not explicitly provided.
        - Inject Authorization header (Bearer token) into the request.
        - Choose request timeout:
            * unlimited for long-running endpoints (training/explain)
            * default 10s for others
        - Handle network exceptions (Timeout/RequestException) by returning None.
        - Handle 429 rate limiting by returning payload + optional retry_after.
        - Handle 401 once by attempting refresh (if allow_refresh=True and not has_retried):
            * calls ui.api.auth.refresh_token (NOT api_call to avoid recursion)
            * updates session_state tokens on success
            * retries the original request exactly once
            * logs out on failure
        - Decode JSON response and normalize it into a consistent wrapper:
            * list payload -> {"status_code": ..., "data": [...]}
            * dict payload -> {"status_code": ..., **data}

    Args:
        endpoint: API path starting with "/" (e.g., "/train_model/train").
        method: HTTP method ("GET", "POST", "DELETE", etc.).
        token: Access token. If None, uses st.session_state["jwt_token"].
        allow_refresh: If True, allows one refresh attempt on 401.
        has_retried: Internal recursion guard to prevent infinite refresh loops.
        **kwargs: Passed directly to requests.request (json, data, files, params, headers, etc.).

    Returns:
        A normalized response dict on success/error, or None on network failure.
    """

    # Resolve token from session if not passed
    if token is None:
        token = st.session_state.get("jwt_token")

    headers = kwargs.pop("headers", {})
    headers.update(get_auth_headers(token))

    url = f"{API_BASE_URL}{endpoint}"

    if endpoint.startswith("/train_model/train") or endpoint.startswith("/assist/explain"):
        timeout_value = None  # unlimited wait
    else:
        timeout_value = 10

    # --- ACTUAL REQUEST ---
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            timeout=timeout_value,
            **kwargs,
        )
    except (Timeout, RequestException):
        return

    # --- RATE LIMIT RETURN ---
    if response.status_code == 429:
        payload = response.json() if response.content else {}
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            payload["retry_after"] = retry_after
        return payload

    # --- ACCESS TOKEN EXPIRED → TRY REFRESH ONCE ---
    if response.status_code == 401 and allow_refresh and not has_retried:
        refresh_tok = st.session_state.get("refresh_token")
        if not refresh_tok:
            logout_and_stop("Session expired. Please log in again.")
            return

        from ui.api.auth import refresh_token
        new_tokens = refresh_token(refresh_tok)

        if isinstance(new_tokens, dict) and "access_token" in new_tokens:
            st.session_state["jwt_token"] = new_tokens["access_token"]
            st.session_state["refresh_token"] = new_tokens["refresh_token"]
            st.session_state["jwt_expires_at"] = new_tokens["expires_at"]

            # retry one time with new token
            return api_call(
                endpoint,
                method,
                token=new_tokens["access_token"],
                allow_refresh=False,
                has_retried=True,
                **kwargs,
            )

        # refresh failed → log out cleanly
        logout_and_stop("Session expired. Please log in again.")
        return

    # --- DECODE JSON RESPONSE ---
    try:
        data = response.json()
    except ValueError:
        return {
            "status_code": response.status_code,
            "error": f"Invalid response: {response.status_code}",
        }

    # unify list/dict format
    if isinstance(data, list):
        return {"status_code": response.status_code, "data": data}

    return {"status_code": response.status_code, **data}
