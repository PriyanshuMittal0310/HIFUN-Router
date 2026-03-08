"""ModelPredictor: loads a trained classifier and predicts SQL/GRAPH routing."""

import os
import sys
import time

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import CLASSIFIER_PATH


class ModelPredictor:
    """Fast inference wrapper for the trained routing classifier."""

    def __init__(self, model_path: str = CLASSIFIER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run model/trainer.py first."
            )
        self.model = joblib.load(model_path)
        self.model_path = model_path

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict engine routing for a feature vector.

        Args:
            feature_vector: 22-dimensional feature vector (1D or 2D array).

        Returns:
            'SQL' or 'GRAPH'
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        t0 = time.perf_counter()
        pred = self.model.predict(feature_vector)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return "GRAPH" if pred == 1 else "SQL"

    def predict_proba(self, feature_vector: np.ndarray) -> dict:
        """Predict with probability scores.

        Returns:
            dict with 'label', 'sql_prob', 'graph_prob', 'inference_ms'
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        t0 = time.perf_counter()
        proba = self.model.predict_proba(feature_vector)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        label = "GRAPH" if proba[1] > 0.5 else "SQL"
        return {
            "label": label,
            "sql_prob": float(proba[0]),
            "graph_prob": float(proba[1]),
            "inference_ms": round(elapsed_ms, 3),
        }

    def predict_batch(self, feature_matrix: np.ndarray) -> list:
        """Predict routing for multiple sub-expressions at once.

        Args:
            feature_matrix: (N, 22) feature matrix.

        Returns:
            List of 'SQL' or 'GRAPH' strings.
        """
        preds = self.model.predict(feature_matrix)
        return ["GRAPH" if p == 1 else "SQL" for p in preds]
