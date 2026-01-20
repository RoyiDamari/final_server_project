MODEL_PRESETS = {
    "linear": {
        "Default (OLS)": {},
        "Ridge (L2)": {"kind": "ridge", "alpha": 1.0, "fit_intercept": True},
        "Lasso (L1)": {"kind": "lasso", "alpha": 1.0, "fit_intercept": True},
        "ElasticNet (L1+L2)": {"kind": "elasticnet", "alpha": 1.0, "l1_ratio": 0.5, "fit_intercept": True},
    },
    "logistic": {
        "Default": {},
        "Ridge (L2)": {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 200},
        "Lasso (L1)": {"penalty": "l1", "C": 1.0, "solver": "liblinear", "max_iter": 200},
        "ElasticNet (L1+L2)": {"penalty": "elasticnet", "l1_ratio": 0.5, "C": 1.0, "solver": "saga", "max_iter": 200},
    },
    "random_forest": {
        "Default": {},
        "Fast (small)": {"n_estimators": 50, "max_depth": 10, "random_state": 42, "n_jobs": -1},
        "Accurate (bigger)": {"n_estimators": 300, "max_depth": None, "random_state": 42, "n_jobs": -1},
    },
}


PARAM_HELP = {
    "linear": "best for predicting a number; fast, interpretable baseline;\n"
              "works well when relationships are roughly linear.",
    "logistic": "best for binary outcomes (yes/no, 0/1);\n"
                "outputs probabilities; not for continuous labels like house prices.",
    "random_forest": "strong general-purpose model for non-linear patterns and mixed features;\n"
                     "often higher accuracy on tabular data; slower and less interpretable.",

    # Parameter-level help
    "Lasso (L1)": "Penalizes by forcing some coefficients to exactly zero.\n"
                  "Useful for feature selection when many features may be irrelevant.",
    "Ridge (L2)": "Penalizes large coefficients smoothly.\n"
                  "Best when many features contribute a little.\n"
                  "Keeps all features, shrinks their influence.",
    "ElasticNet (L1+L2)": "Combines Ridge and Lasso.\n"
                          "Useful when features are correlated and you still want sparsity.",
    "Fast (small)": "Fewer trees / iterations. Lower memory and CPU. Good for exploration",
    "Accurate (bigger)": "More capacity. Better generalization. Slower training.",
    "fit_intercept": "Include a bias term (recommended unless data is centered).",
    "alpha": "Regularization strength. Higher values increase shrinkage.",
    "l1_ratio": "ElasticNet mixing parameter (0 = Ridge, 1 = Lasso).",
    "C": "Inverse regularization strength (smaller = stronger regularization).",
    "solver": "Optimization algorithm used during training.",
    "max_iter": "Maximum number of optimization iterations.",
    "n_estimators": "Number of trees in the forest.",
    "max_depth": "Maximum tree depth. None means unlimited growth.",
    "random_state": "Seed for reproducibility.",
    "n_jobs": "Number of CPU cores used (-1 = all cores).",
}


_VALID_SOLVERS = {
    "l2": ["lbfgs", "newton-cg", "saga", "liblinear"],
    "l1": ["liblinear", "saga"],
    "elasticnet": ["saga"],
}