# Routing Relevance Evaluation

- Train dataset: training_data/fixed_train_base_strict.csv
- Eval dataset: training_data/fixed_eval_set_strict.csv
- Train rows: 231
- Eval rows: 57
- Train label distribution: {'SQL': 121, 'GRAPH': 110}
- Eval label distribution: {'SQL': 30, 'GRAPH': 27}

## Model Comparison

| Model | Accuracy | F1 | Precision | Recall | PR-AUC | Brier | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlwaysSQL | 0.5263 | 0.0000 | 0.0000 | 0.0000 | 0.4737 | 0.4737 | 30 | 0 | 27 | 0 |
| AlwaysGRAPH | 0.4737 | 0.6429 | 0.4737 | 1.0000 | 0.4737 | 0.5263 | 0 | 30 | 0 | 27 |
| TraversalRule | 0.9825 | 0.9818 | 0.9643 | 1.0000 | 0.9643 | 0.0175 | 29 | 1 | 0 | 27 |
| ThresholdBaseline | 0.9649 | 0.9630 | 0.9630 | 0.9630 | 0.9448 | 0.0351 | 29 | 1 | 1 | 26 |
| LogRegBalanced | 0.9825 | 0.9818 | 0.9643 | 1.0000 | 0.9522 | 0.0171 | 29 | 1 | 0 | 27 |
| DecisionTreeBalanced | 0.9825 | 0.9818 | 0.9643 | 1.0000 | 0.9522 | 0.0172 | 29 | 1 | 0 | 27 |
| XGBoostBalanced | 0.9825 | 0.9818 | 0.9643 | 1.0000 | 0.9482 | 0.0174 | 29 | 1 | 0 | 27 |
| XGBoostNoHistory | 0.9825 | 0.9818 | 0.9643 | 1.0000 | 0.9482 | 0.0174 | 29 | 1 | 0 | 27 |

## XGBoost Confidence Thresholds

| Threshold | Coverage | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.5 | 1.0000 | 0.9818 | 0.9643 | 1.0000 |
| 0.6 | 1.0000 | 0.9818 | 0.9643 | 1.0000 |
| 0.7 | 1.0000 | 0.9818 | 0.9643 | 1.0000 |
| 0.8 | 1.0000 | 0.9818 | 0.9643 | 1.0000 |
| 0.9 | 1.0000 | 0.9818 | 0.9643 | 1.0000 |

## XGBoost Per-Dataset Confusion

| Dataset | Rows | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| ogb_real_queries | 8 | 4 | 0 | 0 | 4 |
| snb_real_queries | 49 | 25 | 1 | 0 | 23 |

## Notes

- NoHistory models drop historical runtime features to test leakage risk.
- AlwaysSQL/AlwaysGRAPH quantify class-imbalance floor and ceiling behavior.