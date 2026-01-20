import streamlit as st
from typing import Any
from ui.utils.params.presets import PARAM_HELP


def render_linear_params_ui(model_type: str, base_params: dict, token: str, explain_fn, ask_btn):
    p: dict[str, Any] = dict(base_params or {})
    kind = p.get("kind", "ols")
    p["kind"] = kind

    fit_intercept = st.checkbox(
        "fit_intercept",
        value=bool(p.get("fit_intercept", True)),
        help=PARAM_HELP["fit_intercept"],
        key=f"linear_fit_intercept_{model_type}"
    )

    ask_btn(
        label="fit_intercept",
        model_type="linear",
        token=token,
        explain_fn=explain_fn,
        param_key="fit_intercept",
    )

    p["fit_intercept"] = bool(fit_intercept)

    if kind in ("ridge", "lasso"):
        alpha = st.number_input(
            "alpha",
            min_value=1e-6,
            max_value=1e6,
            value=float(p.get("alpha", 1.0)),
            help=PARAM_HELP["alpha"],
            key=f"linear_alpha_{model_type}"
        )

        ask_btn(
            label="alpha",
            model_type="linear",
            token=token,
            explain_fn=explain_fn,
            param_key="alpha",
        )

        p["alpha"] = float(alpha)

    elif kind == "elasticnet":
        alpha = st.number_input(
            "alpha",
            min_value=1e-6,
            max_value=1e6,
            value=float(p.get("alpha", 1.0)),
            help=PARAM_HELP["alpha"],
            key=f"linear_alpha_{model_type}"
        )

        ask_btn(
            label="alpha",
            model_type="linear",
            token=token,
            explain_fn=explain_fn,
            param_key="alpha",
        )

        l1_ratio = st.number_input(
            "l1_ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(p.get("l1_ratio", 0.5)),
            help=PARAM_HELP["l1_ratio"],
            key=f"linear_l1_ratio_{model_type}"
        )

        ask_btn(
            label="l1_ratio",
            model_type="linear",
            token=token,
            explain_fn=explain_fn,
            param_key="l1_ratio",
        )

        p.update({"alpha": float(alpha), "l1_ratio": float(l1_ratio)})

    return p
