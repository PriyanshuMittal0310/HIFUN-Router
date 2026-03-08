"""Model package: ML pipeline for SQL/GRAPH routing classification."""

from model.trainer import train, load_data, train_decision_tree, train_xgboost
from model.predictor import ModelPredictor
from model.feature_importance import (
    generate_shap_report,
    generate_builtin_importance,
    run_ablation_study,
)

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
