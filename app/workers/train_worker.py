import sys
import json
import argparse
import pandas as pd
import joblib
from app.models.ml_models.model_strategy_factory import get_model_strategy


def main() -> int:
    p = argparse.ArgumentParser(description="Train model worker (no validation).")
    p.add_argument("--csv", required=True, help="Path to training CSV (already validated).")
    p.add_argument("--tmp", required=True, help="Temp path to write the model (will be moved by parent).")
    p.add_argument("--model-type", required=True)
    p.add_argument("--features", required=True, help="JSON list of feature column names.")
    p.add_argument("--label", required=True)
    p.add_argument("--params", required=True, help="JSON dict of model params (already normalized).")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)

        features = json.loads(args.features)
        params   = json.loads(args.params)
        label    = args.label
        mt       = args.model_type

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
