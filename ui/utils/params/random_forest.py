import streamlit as st
from typing import Any
from ui.utils.params.presets import PARAM_HELP


def render_rf_params_ui(model_type: str, base_params: dict, token: str, explain_fn, ask_btn):
    p: dict[str, Any] = dict(base_params or {})

    n_estimators = st.number_input(
        "n_estimators",
        min_value=10,
        max_value=2000,
        value=int(p.get("n_estimators", 100)),
        step=10,
        help=PARAM_HELP["n_estimators"],
        key=f"rf_n_estimators_{model_type}"
    )

    ask_btn(
        label="n_estimators",
        model_type="random_forest",
        token=token,
        explain_fn=explain_fn,
        param_key="n_estimators",
    )

    raw_depth = p.get("max_depth")

    max_depth = st.number_input(
        "max_depth (0=None)",
        min_value=0, max_value=200,
        value=0 if raw_depth is None else int(raw_depth),
        step=1,
        help=PARAM_HELP["max_depth"],
        key=f"rf_max_depth_{model_type}"
    )

    ask_btn(
        label="max_depth",
        model_type="random_forest",
        token=token,
        explain_fn=explain_fn,
        param_key="max_depth",
    )

    random_state = st.number_input(
        "random_state",
        min_value=0,
        max_value=10000,
        value=int(p.get("random_state", 42)),
        step=1,
        help=PARAM_HELP["random_state"],
        key=f"rf_random_state_{model_type}"
    )

    ask_btn(
        label="random_state",
        model_type="random_forest",
        token=token,
        explain_fn=explain_fn,
        param_key="random_state",
    )

    n_jobs = st.number_input(
        "n_jobs (-1=all cores)",
        min_value=-1,
        max_value=64,
        value=int(p.get("n_jobs", -1)),
        step=1,
        help=PARAM_HELP["n_jobs"],
        key=f"rf_n_jobs_{model_type}"
    )

    ask_btn(
        label="n_jobs",
        model_type="random_forest",
        token=token,
        explain_fn=explain_fn,
        param_key="n_jobs",
    )

    p.update({
        "n_estimators": int(n_estimators),
        "max_depth": (None if int(max_depth) == 0 else int(max_depth)),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    })

    return p
