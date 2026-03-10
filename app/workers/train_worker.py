import sys
import json
import argparse
import pandas as pd
import joblib
from app.models.ml_models.model_strategy_factory import get_model_strategy


def main() -> int:
    """
    CLI entry point for the model training worker.

    This worker is intentionally "thin":
        - It assumes the parent process already validated inputs.
        - It reads the CSV from disk, constructs the correct ModelStrategy,
          trains the model, computes metrics, and writes the model artifact.
        - Metrics are written to stdout as JSON.
        - Errors are written to stderr with a "worker_error:" prefix.

    Exit codes:
        0 -> success
        2 -> worker-level exception

    Returns:
        Process exit code integer (0 on success, 2 on failure).
    """

    p = argparse.ArgumentParser(description="Train model worker (no validation).")
    p.add_argument("--csv", required=True, help="Path to training CSV (already validated).")
    p.add_argument("--tmp", required=True, help="Temp path to write the model (will be moved by parent).")
    p.add_argument("--model-type", required=True, help="Model family: linear|logistic|random_forest")
    p.add_argument("--features", required=True, help="JSON list of feature column names.")
    p.add_argument("--label", required=True, help="Target column name in the CSV")
    p.add_argument("--params", required=True, help="JSON dict of model params (already normalized).")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)

        features = json.loads(args.features)
        params = json.loads(args.params)
        label = args.label
        mt = args.model_type

        strat = get_model_strategy(mt, features, label, dict(params))

        model, metrics = strat.train_and_evaluate(df, debug=True)

        joblib.dump(model, args.tmp)

        sys.stdout.write(json.dumps(metrics or {}))
        sys.stdout.flush()
        return 0

    except Exception as e:
        sys.stderr.write(f"worker_error: {e}\n")
        sys.stderr.flush()
        return 2


if __name__ == "__main__":
    sys.exit(main())
