# Routing Relevance Evaluation

- Train dataset: training_data/fixed_train_base_strict.csv
- Eval dataset: training_data/fixed_eval_set_strict.csv
- Train rows: 230
- Eval rows: 58
- Train label distribution: {'SQL': 120, 'GRAPH': 110}
- Eval label distribution: {'SQL': 31, 'GRAPH': 27}

## Model Comparison

| Model | Accuracy | F1 | Precision | Recall | PR-AUC | Brier | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlwaysSQL | 0.5345 | 0.0000 | 0.0000 | 0.0000 | 0.4655 | 0.4655 | 31 | 0 | 27 | 0 |
| AlwaysGRAPH | 0.4655 | 0.6353 | 0.4655 | 1.0000 | 0.4655 | 0.5345 | 0 | 31 | 0 | 27 |
| TraversalRule | 0.9655 | 0.9643 | 0.9310 | 1.0000 | 0.9310 | 0.0345 | 29 | 2 | 0 | 27 |
| ThresholdBaseline | 0.9310 | 0.9259 | 0.9259 | 0.9259 | 0.8918 | 0.0690 | 29 | 2 | 2 | 25 |
| LogRegBalanced | 0.9655 | 0.9643 | 0.9310 | 1.0000 | 0.9354 | 0.0325 | 29 | 2 | 0 | 27 |
| DecisionTreeBalanced | 0.9655 | 0.9643 | 0.9310 | 1.0000 | 0.9286 | 0.0325 | 29 | 2 | 0 | 27 |
| XGBoostBalanced | 0.9655 | 0.9643 | 0.9310 | 1.0000 | 0.9298 | 0.0326 | 29 | 2 | 0 | 27 |
| XGBoostNoHistory | 0.9655 | 0.9643 | 0.9310 | 1.0000 | 0.9298 | 0.0326 | 29 | 2 | 0 | 27 |

## XGBoost Confidence Thresholds

| Threshold | Coverage | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.5 | 1.0000 | 0.9643 | 0.9310 | 1.0000 |
| 0.6 | 1.0000 | 0.9643 | 0.9310 | 1.0000 |
| 0.7 | 1.0000 | 0.9643 | 0.9310 | 1.0000 |
| 0.8 | 1.0000 | 0.9643 | 0.9310 | 1.0000 |
| 0.9 | 1.0000 | 0.9643 | 0.9310 | 1.0000 |

## XGBoost Per-Dataset Confusion

| Dataset | Rows | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| ogb_real_queries | 12 | 6 | 0 | 0 | 6 |
| snb_real_queries | 46 | 23 | 2 | 0 | 21 |

## Notes

- NoHistory models drop historical runtime features to test leakage risk.
- AlwaysSQL/AlwaysGRAPH quantify class-imbalance floor and ceiling behavior.