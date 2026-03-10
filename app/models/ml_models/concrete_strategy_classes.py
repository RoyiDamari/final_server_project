from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from app.models.ml_models.base_model_strategy import BaseModelStrategy


class LinearRegressionStrategy(BaseModelStrategy):
    META_KEYS = {"kind"}

    def build_pipeline(self, df=None) -> Pipeline:
        """
        Build a regression pipeline with preprocessing + a linear-family estimator.

        Uses model_params["kind"] (META key) to choose:
          - "ols" => LinearRegression
          - "ridge" => Ridge
          - "lasso" => Lasso
          - "elasticnet" => ElasticNet

        Args:
            df: Optional dataframe used to build the preprocessor.

        Returns:
            Pipeline: Unfitted pipeline.
        """

        reg_type = str(self.model_params.pop("kind", "ols") or "ols").strip().lower()
        if reg_type == "ridge":
            model = Ridge(**self.model_params)
        elif reg_type == "lasso":
            model = Lasso(**self.model_params)
        elif reg_type == "elasticnet":
            model = ElasticNet(**self.model_params)
        else:
            model = LinearRegression(**self.model_params)

        pre = self.build_preprocessor(df)
        return Pipeline([("pre", pre), ("model", model)])

    def evaluate(self, model, x_test, y_test) -> Dict[str, Any]:
        """
        Evaluate regression performance.

        Args:
            model: Fitted pipeline.
            x_test: Feature matrix.
            y_test: True targets.

        Returns:
            dict[str, Any]: {"mae": float, "r2": float}
        """

        y_pred = model.predict(x_test)
        return {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred))
        }


class LogisticRegressionStrategy(BaseModelStrategy):
    def build_pipeline(self, df=None) -> Pipeline:
        """
        Build a classification pipeline with preprocessing + LogisticRegression.

        Ensures a safe default:
          - max_iter defaults to 1000 if not provided.

        Args:
            df: Optional dataframe used to build the preprocessor.

        Returns:
            Pipeline: Unfitted pipeline.
        """

        self_params = dict(self.model_params)
        self_params.setdefault("max_iter", 1000)
        lr = LogisticRegression(**self_params)
        pre = self.build_preprocessor(df)

        return Pipeline([("pre", pre), ("model", lr)])

    def evaluate(self, model, x_test, y_test) -> Dict[str, Any]:
        """
        Evaluate classification performance.

        Args:
            model: Fitted pipeline.
            x_test: Feature matrix.
            y_test: True labels.

        Returns:
            dict[str, Any]:
                {
                  "accuracy": float,
                  "classification_report": dict  # sklearn classification_report output_dict
                }
        """

        y_pred = model.predict(x_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }


class RandomForestStrategy(BaseModelStrategy):
    META_KEYS = {"task"}

    def build_pipeline(self, df=None) -> Pipeline:
        """
        Build a RandomForest pipeline for classification or regression.

        Uses model_params["task"] (META key):
          - "classification" => RandomForestClassifier
          - "regression" => RandomForestRegressor
          - "auto" => choose based on validated target type (self._is_classification)

        IMPORTANT:
          - validate_target_type() must be called before task="auto" can be resolved.

        Args:
            df: Optional dataframe used to build the preprocessor.

        Returns:
            Pipeline: Unfitted pipeline.

        Raises:
            RuntimeError: If task="auto" but target type was not validated yet.
        """

        task = (self.model_params.pop("task", "auto") or "auto").lower()

        if task == "classification":
            model = RandomForestClassifier(**self.model_params)

        elif task == "regression":
            model = RandomForestRegressor(**self.model_params)

        else:
            if self._is_classification is None:
                raise RuntimeError(
                    "Target type not validated before building RandomForest pipeline"
                )

            model = (
                RandomForestClassifier(**self.model_params)
                if self._is_classification
                else RandomForestRegressor(**self.model_params)
            )

        pre = self.build_preprocessor(df)
        return Pipeline([("pre", pre), ("model", model)])

    def evaluate(self, model, x_test, y_test) -> Dict[str, Any]:
        """
        Evaluate RandomForest performance with metrics depending on task type.

        Args:
            model: Fitted pipeline.
            x_test: Feature matrix.
            y_test: True labels/targets.

        Returns:
            dict[str, Any]:
                - classification: {"accuracy": float, "classification_report": dict}
                - regression: {"mae": float, "r2": float}
        """

        y_pred = model.predict(x_test)
        if self._is_classification:
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
            }
        else:
            return {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
