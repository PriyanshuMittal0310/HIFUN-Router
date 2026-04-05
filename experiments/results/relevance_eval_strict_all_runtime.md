# Routing Relevance Evaluation

- Train dataset: training_data/fixed_train_base_strict_all.csv
- Eval dataset: training_data/fixed_eval_set_strict_all.csv
- Train rows: 821
- Eval rows: 205
- Train label distribution: {'SQL': 718, 'GRAPH': 103}
- Eval label distribution: {'SQL': 167, 'GRAPH': 38}

## Model Comparison

| Model | Accuracy | F1 | Precision | Recall | PR-AUC | Brier | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlwaysSQL | 0.8146 | 0.0000 | 0.0000 | 0.0000 | 0.1854 | 0.1854 | 167 | 0 | 38 | 0 |
| AlwaysGRAPH | 0.1854 | 0.3128 | 0.1854 | 1.0000 | 0.1854 | 0.8146 | 0 | 167 | 0 | 38 |
| TraversalRule | 0.7512 | 0.5984 | 0.4270 | 1.0000 | 0.4270 | 0.2488 | 116 | 51 | 0 | 38 |
| ThresholdBaseline | 0.8195 | 0.5934 | 0.5094 | 0.7105 | 0.4156 | 0.1805 | 141 | 26 | 11 | 27 |
| LogRegBalanced | 0.7902 | 0.6387 | 0.4691 | 1.0000 | 0.5417 | 0.1324 | 124 | 43 | 0 | 38 |
| DecisionTreeBalanced | 0.8829 | 0.7551 | 0.6167 | 0.9737 | 0.5785 | 0.0927 | 144 | 23 | 1 | 37 |
| XGBoostBalanced | 0.8927 | 0.7660 | 0.6429 | 0.9474 | 0.5989 | 0.0885 | 147 | 20 | 2 | 36 |
| XGBoostNoHistory | 0.8927 | 0.7660 | 0.6429 | 0.9474 | 0.5989 | 0.0885 | 147 | 20 | 2 | 36 |

## XGBoost Confidence Thresholds

| Threshold | Coverage | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.5 | 1.0000 | 0.7660 | 0.6429 | 0.9474 |
| 0.6 | 0.9610 | 0.7742 | 0.6429 | 0.9730 |
| 0.7 | 0.9610 | 0.7742 | 0.6429 | 0.9730 |
| 0.8 | 0.9415 | 0.7826 | 0.6429 | 1.0000 |
| 0.9 | 0.6341 | 1.0000 | 1.0000 | 1.0000 |

## XGBoost Per-Dataset Confusion

| Dataset | Rows | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| job_real_queries | 3 | 3 | 0 | 0 | 0 |
| ogb_real_queries | 54 | 43 | 2 | 1 | 8 |
| snb_bi_real_queries | 16 | 16 | 0 | 0 | 0 |
| snb_real_queries | 124 | 77 | 18 | 1 | 28 |
| tpcds_real_queries | 8 | 8 | 0 | 0 | 0 |

## Notes

- NoHistory models drop historical runtime features to test leakage risk.
- AlwaysSQL/AlwaysGRAPH quantify class-imbalance floor and ceiling behavior.