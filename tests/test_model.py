"""Tests for Phase 4: ML Pipeline (collection, training, prediction, importance)."""

import csv
import json
import os
import sys
import tempfile

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.feature_extractor import FEATURE_NAMES, NUM_FEATURES


# ---------------------------------------------------------------------------
# 1. Training Data Collection Tests
# ---------------------------------------------------------------------------

class TestCollectionScript:
    """Test training_data/collection_script.py."""

    def test_collect_produces_csv(self, tmp_path):
        """Collection script generates a valid CSV with correct columns."""
        from training_data.collection_script import collect_training_data

        output = str(tmp_path / "test_labeled.csv")
        path, count = collect_training_data(
            output_path=output,
            augment_factor=2,
        )
        assert os.path.exists(path)
        assert count > 0

        # Validate columns
        import pandas as pd
        df = pd.read_csv(path)
        expected_cols = ["sub_id", "query_id", "dataset"] + FEATURE_NAMES + [
            "sql_runtime_ms", "graph_runtime_ms", "label"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_collect_has_both_labels(self, tmp_path):
        """Dataset should contain both SQL and GRAPH labels."""
        from training_data.collection_script import collect_training_data

        output = str(tmp_path / "test_labels.csv")
        collect_training_data(output_path=output, augment_factor=5)

        import pandas as pd
        df = pd.read_csv(output)
        labels = set(df["label"].unique())
        assert "SQL" in labels, "No SQL labels found"
        assert "GRAPH" in labels, "No GRAPH labels found"

    def test_augmentation_multiplies_rows(self, tmp_path):
        """Augmentation should increase row count by (1 + augment_factor)."""
        from training_data.collection_script import collect_training_data

        output1 = str(tmp_path / "base.csv")
        _, count1 = collect_training_data(output_path=output1, augment_factor=0)

        output2 = str(tmp_path / "augmented.csv")
        _, count2 = collect_training_data(output_path=output2, augment_factor=3)

        # augmented should be base * (1 + 3) = base * 4
        assert count2 == count1 * 4

    def test_features_are_numeric(self, tmp_path):
        """All feature columns should be numeric (no NaN)."""
        from training_data.collection_script import collect_training_data

        output = str(tmp_path / "numeric.csv")
        collect_training_data(output_path=output, augment_factor=1)

        import pandas as pd
        df = pd.read_csv(output)
        for feat in FEATURE_NAMES:
            assert pd.api.types.is_numeric_dtype(df[feat]), f"{feat} is not numeric"
            assert not df[feat].isna().any(), f"{feat} has NaN values"

    def test_runtimes_positive(self, tmp_path):
        """Simulated runtimes should be positive."""
        from training_data.collection_script import collect_training_data

        output = str(tmp_path / "runtimes.csv")
        collect_training_data(output_path=output, augment_factor=1)

        import pandas as pd
        df = pd.read_csv(output)
        assert (df["sql_runtime_ms"] > 0).all()
        assert (df["graph_runtime_ms"] > 0).all()


# ---------------------------------------------------------------------------
# 2. Model Training Tests
# ---------------------------------------------------------------------------

class TestTrainer:
    """Test model/trainer.py."""

    @pytest.fixture(autouse=True)
    def setup_training_data(self, tmp_path):
        """Generate training data and set up paths for tests."""
        from training_data.collection_script import collect_training_data
        self.data_path = str(tmp_path / "labeled_runs.csv")
        self.model_dir = str(tmp_path / "model_artifacts")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "test_classifier.pkl")

        collect_training_data(
            output_path=self.data_path,
            augment_factor=5,
        )

    def test_load_data(self):
        """load_data returns correctly shaped arrays."""
        from model.trainer import load_data
        X, y, df = load_data(self.data_path)
        assert X.shape[1] == NUM_FEATURES
        assert len(y) == len(df)
        assert set(np.unique(y)).issubset({0, 1})

    def test_train_decision_tree(self):
        """Decision Tree trains and returns CV metrics."""
        from model.trainer import load_data, train_decision_tree
        X, y, _ = load_data(self.data_path)
        model, metrics = train_decision_tree(X, y, cv_folds=3)
        assert metrics["cv_f1_mean"] > 0
        assert hasattr(model, "predict")

    def test_train_xgboost(self):
        """XGBoost trains and returns CV metrics."""
        from model.trainer import load_data, train_xgboost
        X, y, _ = load_data(self.data_path)
        model, metrics = train_xgboost(X, y, cv_folds=3)
        assert metrics["cv_f1_mean"] > 0
        assert hasattr(model, "predict")

    def test_full_train_pipeline(self):
        """Full train() pipeline saves model and results."""
        from model.trainer import train
        results = train(
            labeled_data_path=self.data_path,
            model_out=self.model_path,
            cv_folds=3,
        )
        assert os.path.exists(self.model_path)
        assert results["best_model"] in ("DecisionTree", "XGBoost")
        assert results["xgboost"]["cv_f1_mean"] > 0
        assert results["decision_tree"]["cv_f1_mean"] > 0

    def test_model_accuracy_reasonable(self):
        """Model should achieve >70% accuracy on training data."""
        from model.trainer import train
        results = train(
            labeled_data_path=self.data_path,
            model_out=self.model_path,
            cv_folds=3,
        )
        best = results["best_model"].lower().replace(" ", "_")
        best_key = "xgboost" if "xgb" in best else best
        if best_key not in results:
            best_key = "decision_tree" if best == "decisiontree" else "xgboost"
        assert results[best_key]["accuracy"] > 0.7


# ---------------------------------------------------------------------------
# 3. Model Predictor Tests
# ---------------------------------------------------------------------------

class TestPredictor:
    """Test model/predictor.py."""

    @pytest.fixture(autouse=True)
    def setup_model(self, tmp_path):
        """Train and save a model for prediction tests."""
        from training_data.collection_script import collect_training_data
        from model.trainer import train

        data_path = str(tmp_path / "labeled_runs.csv")
        self.model_path = str(tmp_path / "test_classifier.pkl")

        collect_training_data(output_path=data_path, augment_factor=5)
        train(labeled_data_path=data_path, model_out=self.model_path, cv_folds=3)

    def test_predict_returns_sql_or_graph(self):
        """predict() returns 'SQL' or 'GRAPH'."""
        from model.predictor import ModelPredictor
        pred = ModelPredictor(self.model_path)
        fv = np.random.randn(NUM_FEATURES).astype(np.float32)
        result = pred.predict(fv)
        assert result in ("SQL", "GRAPH")

    def test_predict_proba_has_probabilities(self):
        """predict_proba() returns label + probabilities."""
        from model.predictor import ModelPredictor
        pred = ModelPredictor(self.model_path)
        fv = np.random.randn(NUM_FEATURES).astype(np.float32)
        result = pred.predict_proba(fv)
        assert result["label"] in ("SQL", "GRAPH")
        assert 0 <= result["sql_prob"] <= 1
        assert 0 <= result["graph_prob"] <= 1
        assert abs(result["sql_prob"] + result["graph_prob"] - 1.0) < 0.01

    def test_predict_batch(self):
        """predict_batch() handles multiple inputs."""
        from model.predictor import ModelPredictor
        pred = ModelPredictor(self.model_path)
        X = np.random.randn(10, NUM_FEATURES).astype(np.float32)
        results = pred.predict_batch(X)
        assert len(results) == 10
        assert all(r in ("SQL", "GRAPH") for r in results)

    def test_inference_speed(self):
        """Single prediction should be fast (<100ms)."""
        import time
        from model.predictor import ModelPredictor
        pred = ModelPredictor(self.model_path)
        fv = np.random.randn(NUM_FEATURES).astype(np.float32)

        # Warm up
        pred.predict(fv)

        t0 = time.perf_counter()
        for _ in range(100):
            pred.predict(fv)
        avg_ms = (time.perf_counter() - t0) * 1000 / 100
        assert avg_ms < 100, f"Average inference: {avg_ms:.1f}ms (should be <100ms)"

    def test_missing_model_raises(self, tmp_path):
        """ModelPredictor raises FileNotFoundError for missing model."""
        from model.predictor import ModelPredictor
        with pytest.raises(FileNotFoundError):
            ModelPredictor(str(tmp_path / "nonexistent.pkl"))


# ---------------------------------------------------------------------------
# 4. Feature Importance Tests
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    """Test model/feature_importance.py."""

    @pytest.fixture(autouse=True)
    def setup_model(self, tmp_path):
        """Train model and prepare paths for importance tests."""
        from training_data.collection_script import collect_training_data
        from model.trainer import train

        self.data_path = str(tmp_path / "labeled_runs.csv")
        self.model_path = str(tmp_path / "test_classifier.pkl")
        self.output_dir = str(tmp_path / "analysis")

        collect_training_data(output_path=self.data_path, augment_factor=5)
        train(labeled_data_path=self.data_path, model_out=self.model_path, cv_folds=3)

    def test_builtin_importance(self):
        """Built-in importance generates files and returns dict."""
        import joblib
        from model.feature_importance import generate_builtin_importance

        model = joblib.load(self.model_path)
        result = generate_builtin_importance(
            model=model, output_dir=self.output_dir,
        )
        assert len(result) == NUM_FEATURES
        assert os.path.exists(os.path.join(self.output_dir, "builtin_importance.png"))
        assert os.path.exists(os.path.join(self.output_dir, "builtin_importance.json"))

    def test_shap_report(self):
        """SHAP report generates summary plot and values JSON."""
        import joblib
        import pandas as pd
        from model.feature_importance import generate_shap_report

        model = joblib.load(self.model_path)
        df = pd.read_csv(self.data_path)
        X = df[FEATURE_NAMES].values.astype(np.float32)

        result, shap_values = generate_shap_report(
            model=model, X_train=X,
            feature_names=FEATURE_NAMES, output_dir=self.output_dir,
        )
        assert len(result) == NUM_FEATURES
        assert os.path.exists(os.path.join(self.output_dir, "shap_summary.png"))
        assert os.path.exists(os.path.join(self.output_dir, "shap_values.json"))


# ---------------------------------------------------------------------------
# 5. End-to-End Pipeline Test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Test the complete Phase 4 pipeline: collect → train → predict."""

    def test_full_pipeline(self, tmp_path):
        """Run through the entire ML pipeline."""
        from training_data.collection_script import collect_training_data
        from model.trainer import train
        from model.predictor import ModelPredictor
        from parser.dsl_parser import DSLParser
        from decomposer.query_decomposer import QueryDecomposer
        from features.feature_extractor import FeatureExtractor

        # Step 1: Collect training data
        data_path = str(tmp_path / "labeled_runs.csv")
        collect_training_data(output_path=data_path, augment_factor=5)

        # Step 2: Train model
        model_path = str(tmp_path / "classifier.pkl")
        results = train(
            labeled_data_path=data_path,
            model_out=model_path,
            cv_folds=3,
        )
        assert results["best_model"] in ("DecisionTree", "XGBoost")

        # Step 3: Load predictor and run on a real query
        predictor = ModelPredictor(model_path)
        extractor = FeatureExtractor()

        # Parse a sample query
        parser = DSLParser()
        query = {
            "query_id": "test_e2e",
            "operations": [
                {
                    "op_id": "s1",
                    "type": "FILTER",
                    "source": "customer",
                    "fields": ["c_custkey", "c_name"],
                    "predicate": {"column": "c_custkey", "operator": "<", "value": 100},
                    "depends_on": [],
                },
                {
                    "op_id": "s2",
                    "type": "AGGREGATE",
                    "source": "s1",
                    "fields": ["total"],
                    "aggregate": {
                        "group_by": [],
                        "functions": [{"func": "COUNT", "column": "c_custkey"}],
                    },
                    "depends_on": ["s1"],
                },
            ],
        }
        nodes = parser.parse(query)
        decomposer = QueryDecomposer()
        sub_exprs = decomposer.decompose(nodes)

        # Step 4: Route each sub-expression
        for sub in sub_exprs:
            fv = extractor.extract(sub)
            engine = predictor.predict(fv)
            assert engine in ("SQL", "GRAPH")

        extractor.close()
