import streamlit as st
from typing import Any
from ui.utils.params.presets import PARAM_HELP
from ui.utils.params.presets import _VALID_SOLVERS


def render_logistic_params_ui(model_type: str, base_params: dict, token: str, explain_fn, ask_btn):
    p: dict[str, Any] = dict(base_params or {})
    penalty = p.get("penalty", "l2")
    p["penalty"] = penalty

    solver_options = _VALID_SOLVERS[penalty]
    default_solver = p.get("solver", solver_options[0])

    if default_solver not in solver_options:
        default_solver = solver_options[0]

    solver = st.selectbox(
        "solver",
        solver_options,
        index=solver_options.index(default_solver),
        help=PARAM_HELP["solver"],
        key=f"logistic_solver_{model_type}"
    )

    ask_btn(
        label="solver",
        model_type="logistic",
        token=token,
        explain_fn=explain_fn,
        param_key="solver",
    )

    p["solver"] = solver

    c = st.number_input(
        "C",
        min_value=1e-6,
        max_value=1e6,
        value=float(p.get("C", 1.0)),
        help=PARAM_HELP["C"],
        key=f"logistic_C_{model_type}"
    )

    ask_btn(
        label="C",
        model_type="logistic",
        token=token,
        explain_fn=explain_fn,
        param_key="C",
    )

    max_iter = st.number_input(
        "max_iter",
        min_value=50,
        max_value=10000,
        value=int(p.get("max_iter", 200)),
        step=50,
        help=PARAM_HELP["max_iter"],
        key=f"logistic_max_iter_{model_type}"
    )

    ask_btn(
        label="max_iter",
        model_type="logistic",
        token=token,
        explain_fn=explain_fn,
        param_key="max_iter",
    )

    p.update({"C": float(c), "max_iter": int(max_iter)})

    if penalty == "elasticnet":
        l1_ratio = st.number_input(
            "l1_ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(p.get("l1_ratio", 0.5)),
            help=PARAM_HELP["l1_ratio"],
            key=f"logistic_l1_ratio_{model_type}"
        )

        ask_btn(
            label="l1_ratio",
            model_type="logistic",
            token=token,
            explain_fn=explain_fn,
            param_key="l1_ratio",
        )

        p["l1_ratio"] = float(l1_ratio)

    else:
        p.pop("l1_ratio", None)

    return p
