"""Model package: ML pipeline for SQL/GRAPH routing classification."""

from model.trainer import train, load_data, train_decision_tree, train_xgboost
from model.predictor import ModelPredictor

# SHAP can require a local C/C++ toolchain in some environments.
# Keep core training/evaluation imports available even when SHAP extras are missing.
try:
    from model.feature_importance import (
        generate_shap_report,
        generate_builtin_importance,
        run_ablation_study,
    )
except Exception:  # pragma: no cover - environment-dependent optional dependency
    generate_shap_report = None
    generate_builtin_importance = None
    run_ablation_study = None

__all__ = [
    "train",
    "load_data",
    "train_decision_tree",
    "train_xgboost",
    "ModelPredictor",
    "generate_shap_report",
    "generate_builtin_importance",
    "run_ablation_study",
]
