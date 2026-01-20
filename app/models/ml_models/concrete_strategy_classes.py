from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from .base_model_strategy import BaseModelStrategy


class LinearRegressionStrategy(BaseModelStrategy):
    META_KEYS = {"kind"}

    def build_pipeline(self, df=None):
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

    def evaluate(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        return {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred))
        }


class LogisticRegressionStrategy(BaseModelStrategy):
    def build_pipeline(self, df=None):
        self_params = dict(self.model_params)
        self_params.setdefault("max_iter", 1000)
        lr = LogisticRegression(**self_params)
        pre = self.build_preprocessor(df)

        return Pipeline([("pre", pre), ("model", lr)])

    def evaluate(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }


class RandomForestStrategy(BaseModelStrategy):
    META_KEYS = {"task"}

    def build_pipeline(self, df=None):
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

    def evaluate(self, model, x_test, y_test):
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
