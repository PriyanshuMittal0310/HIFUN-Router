"""Baseline routing strategies for comparison against ML-based routing.

Provides three baselines of increasing strength:
  1. TrivialRule: TRAVERSAL → GRAPH, else → SQL  (original paper baseline)
  2. ThresholdBaseline: multi-feature threshold tuned on validation split
  3. LogisticRegressionBaseline: linear classifier on the same 22-dim features

Usage:
    from router.baselines import trivial_rule_route, ThresholdBaseline, LogisticRegressionBaseline
"""

import numpy as np
from parser.ast_nodes import SubExpression


# ── Baseline 1: Trivial Rule (legacy paper baseline) ──────────────

def trivial_rule_route(sub_expr: SubExpression) -> str:
    """TRAVERSAL → GRAPH, everything else → SQL."""
    return "GRAPH" if sub_expr.primary_op_type == "TRAVERSAL" else "SQL"


# ── Baseline 2: Two-Feature Threshold Rule (PRIMARY baseline) ─────

class ThresholdBaseline:
    """Threshold-based routing using the most important features.

    Represents the best a domain expert could do WITHOUT machine learning.
    Thresholds are tuned via grid search on a held-out validation split.
    """

    def __init__(self, avg_degree_thresh=5.0, max_hops_thresh=3,
                 selectivity_thresh=0.01):
        self.avg_degree_thresh = avg_degree_thresh
        self.max_hops_thresh = max_hops_thresh
        self.selectivity_thresh = selectivity_thresh

    def route(self, feature_vector: np.ndarray,
              feature_names: list) -> str:
        """Route a SubExpression using threshold rules.

        Args:
            feature_vector: The 22-dim feature vector from FeatureExtractor
            feature_names:  The ordered list of feature names
        Returns:
            "SQL" or "GRAPH"
        """
        fv = dict(zip(feature_names, feature_vector))

        has_traversal = fv.get("has_traversal", 0)
        avg_degree = fv.get("avg_degree", 0)
        max_hops = fv.get("max_hops", 0)
        selectivity = fv.get("selectivity", 1.0)

        if (has_traversal >= 0.5
                and avg_degree > self.avg_degree_thresh
                and max_hops <= self.max_hops_thresh
                and selectivity >= self.selectivity_thresh):
            return "GRAPH"
        return "SQL"

    @classmethod
    def tune_thresholds(cls, labeled_data_path: str,
                        feature_names: list,
                        val_fraction: float = 0.2):
        """Grid-search over threshold values on a held-out validation split.

        Returns the ThresholdBaseline instance with the best F1.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        df = pd.read_csv(labeled_data_path)
        X = df[feature_names].values
        y = (df["label"] == "GRAPH").astype(int).values

        _, X_val, _, y_val = train_test_split(
            X, y, test_size=val_fraction, stratify=y, random_state=42
        )

        best_f1, best_params = 0, {}

        for d in [2.0, 5.0, 10.0, 20.0]:
            for h in [1, 2, 3, 4]:
                for s in [0.001, 0.01, 0.05, 0.1]:
                    baseline = cls(
                        avg_degree_thresh=d,
                        max_hops_thresh=h,
                        selectivity_thresh=s,
                    )
                    preds = [
                        int(baseline.route(x, feature_names) == "GRAPH")
                        for x in X_val
                    ]
                    f1 = f1_score(y_val, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            "avg_degree_thresh": d,
                            "max_hops_thresh": h,
                            "selectivity_thresh": s,
                        }

        print(f"Best threshold baseline: F1={best_f1:.3f}, params={best_params}")
        return cls(**best_params)


# ── Baseline 3: Logistic Regression ───────────────────────────────

class LogisticRegressionBaseline:
    """Linear classifier on the same 22-dim feature vector.

    If XGBoost only marginally beats this, the non-linearity of the
    problem is limited and simpler models suffice.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on labeled feature vectors (y: 1=GRAPH, 0=SQL)."""
        self.model.fit(X, y)
        self._fitted = True

    def route(self, feature_vector: np.ndarray) -> str:
        """Predict routing for a single feature vector."""
        if not self._fitted:
            raise RuntimeError("LogisticRegressionBaseline has not been fitted")
        pred = self.model.predict(feature_vector.reshape(1, -1))[0]
        return "GRAPH" if pred == 1 else "SQL"

    def route_batch(self, X: np.ndarray) -> list:
        """Predict routing for a batch of feature vectors."""
        if not self._fitted:
            raise RuntimeError("LogisticRegressionBaseline has not been fitted")
        preds = self.model.predict(X)
        return ["GRAPH" if p == 1 else "SQL" for p in preds]
