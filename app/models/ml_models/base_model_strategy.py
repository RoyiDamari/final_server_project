from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, cross_val_score
from app.exceptions.train_model import ModelTypeMismatchException


class SelectiveSkewFix(BaseEstimator, TransformerMixin):
    def __init__(self, t_pos: float = 1.0, t_any: float = 1.0):
        self.t_pos = t_pos
        self.t_any = t_any
        self._use_log1p = None
        self._use_yj = None
        self._yj: Optional[PowerTransformer] = None
        self._means_: Optional[np.ndarray] = None

    def fit(self, x, _y=None):
        x = np.asarray(x, dtype=float)
        col_means = np.nanmean(x, axis=0)
        xm = np.where(np.isnan(x), col_means, x)
        self._means_ = col_means

        skews = pd.DataFrame(xm).skew(skipna=True).values
        mins  = np.nanmin(xm, axis=0)

        self._use_log1p = (mins >= 0) & (skews >= self.t_pos)
        self._use_yj    = (~self._use_log1p) & (np.abs(skews) >= self.t_any)

        if self._use_yj.any():
            self._yj = PowerTransformer(method="yeo-johnson", standardize=False)
            self._yj.fit(xm[:, self._use_yj])

        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        xm = np.where(np.isnan(x), self._means_, x) if self._means_ is not None else x.copy()
        z = xm.copy()
        if self._use_log1p is not None and self._use_log1p.any():
            z[:, self._use_log1p] = np.log1p(z[:, self._use_log1p])
        if self._use_yj is not None and self._use_yj.any() and self._yj is not None:
            z[:, self._use_yj] = self._yj.transform(z[:, self._use_yj])
        return z


class BaseModelStrategy(ABC):
    def __init__(self, model_type: str, features: list[str], label: str, model_params: dict[str, Any] | None):
        self.model_type = model_type
        self.features = features
        self.label = label
        self.model_params: dict[str, Any] = model_params or {}
        self._is_classification: bool | None = None

    @staticmethod
    def is_classification(y: pd.Series) -> bool:
        """
            Returns True for BOTH binary and multiclass classification targets.
            Returns False only for continuous regression targets.
            """

        # categorical / object → classification
        if isinstance(y.dtype, pd.CategoricalDtype):
            return True

        if pd.api.types.is_object_dtype(y):
            return True

        # Boolean → classification
        if y.dtype.kind == "b":
            return True

        # Numeric → regression by default
        if pd.api.types.is_numeric_dtype(y):
            # Binary numeric labels (0/1) → classification
            uniq = y.dropna().unique()
            if len(uniq) == 2 and set(uniq).issubset({0, 1}):
                return True

            # Everything else → regression
            return False

        return False

    def validate_target_type(self, y: pd.Series) -> None:
        self._is_classification = self.is_classification(y)

        if self.model_type == "logistic" and not self._is_classification:
            raise ModelTypeMismatchException(
                detail=(
                    "Logistic regression supports classification only "
                    "(binary or multiclass). "
                    "Detected a continuous label."
                )
            )

        if self.model_type == "linear" and self._is_classification:
            raise ModelTypeMismatchException(
                detail=(
                    "Linear regression requires a continuous numeric label. "
                    "Detected a categorical/multiclass label."
                )
            )

        if self.model_type == "random_forest":
            task = str(self.model_params.get("task", "auto") or "auto").strip().lower()
            if task == "classification" and not self._is_classification:
                raise ModelTypeMismatchException(
                    detail=(
                        "Random Forest (classification) requires a categorical label "
                        "(binary or multiclass), but the selected label appears to be continuous. "
                        "Choose task='auto' or 'regression'."
                    )
                )
            if task == "regression" and self._is_classification:
                raise ModelTypeMismatchException(
                    detail=(
                        "Random Forest (regression) requires a continuous numeric label. "
                        "The selected label appears to be categorical/binary. "
                        "Choose task='classification' or 'auto'."
                    )
                )

    @staticmethod
    def _is_identity_name(col: str) -> bool:
        col = col.lower()
        return col == "id" or col.endswith("_id")

    @staticmethod
    def _detect_group_col(
        df: pd.DataFrame,
        min_avg_rows_per_group: float = 2.0,
        max_unique_frac: float = 0.90,
    ) -> Optional[str]:

        n = len(df)
        candidates = []

        for col in df.columns:
            if not BaseModelStrategy._is_identity_name(col):
                continue

            s = df[col]
            if s.isna().all():
                continue

            k = s.nunique(dropna=True)
            if k == n:
                continue  # unique per row → i.i.d.

            if k <= 1:
                continue

            avg_rows = n / k
            if avg_rows < min_avg_rows_per_group:
                continue

            if k > max_unique_frac * n:
                continue

            candidates.append((col, avg_rows))

        if not candidates:
            return None

        # choose strongest grouping signal
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0] if candidates else None

    def _choose_cv(self, groups: Optional[pd.Series], n_splits: int):
        if groups is not None:
            unique_groups = groups.nunique(dropna=True)
            if unique_groups >= n_splits:
                return GroupKFold(n_splits=n_splits), groups.values
        if self._is_classification:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None
        return KFold(n_splits=n_splits, shuffle=True, random_state=42), None

    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        x = df[self.features]
        num_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        cat_cols = [c for c in x.columns if c not in num_cols]

        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("skewfix", SelectiveSkewFix(t_pos=1.0, t_any=1.0)),
            ("scale", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(drop="first", handle_unknown="ignore", min_frequency=0.01)),
        ])
        return ColumnTransformer(
            [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
            remainder="drop", verbose_feature_names_out=False
        )

    @abstractmethod
    def build_pipeline(self, df: Optional[pd.DataFrame] = None) -> Pipeline:
        pass

    @abstractmethod
    def evaluate(self, model, x_test, y_test) -> Dict[str, Any]:
        pass

    def train_and_evaluate(
            self,
            df: pd.DataFrame,
            cv_splits: int = 5,
            debug: bool = False,
    ) -> Tuple[Pipeline, dict[str, Any]]:

        x = df[self.features].copy()
        y = df[self.label].copy()

        self.validate_target_type(y)

        group_col = self._detect_group_col(df)
        groups = df[group_col] if group_col else None
        if group_col in x.columns:
            x = x.drop(columns=[group_col])

        pipe = self.build_pipeline(df)
        cv, garr = self._choose_cv(groups, cv_splits)

        metrics_debug = {}
        if debug:
            if self._is_classification:
                n_classes = y.nunique(dropna=True)

                if n_classes <= 2:
                    scoring = "roc_auc"
                else:
                    scoring = "accuracy"  # SAFE default for multiclass
            else:
                scoring = "r2"

            scores = cross_val_score(pipe, x, y, cv=cv, scoring=scoring, groups=garr)

            metrics_debug.update({
                "cv_mean": float(np.mean(scores)),
                "cv_std": float(np.std(scores)),
                "cv_scores": scores.tolist(),
                "cv_splits": cv_splits,
                "scoring": scoring,
            })

        pipe.fit(x, y)

        metrics_final = self.evaluate(pipe, x, y)

        metrics = {**metrics_final, **metrics_debug} if debug else metrics_final
        return pipe, metrics

