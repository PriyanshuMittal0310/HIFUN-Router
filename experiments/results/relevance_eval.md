# Routing Relevance Evaluation

- Train dataset: /home/mitta/HIFUN-Router-clone/training_data/fixed_train_balanced.csv
- Eval dataset: /home/mitta/HIFUN-Router-clone/training_data/fixed_eval_set.csv
- Train rows: 2068
- Eval rows: 259
- Train label distribution: {'SQL': 1034, 'GRAPH': 1034}
- Eval label distribution: {'SQL': 258, 'GRAPH': 1}

## Model Comparison

| Model | Accuracy | F1 | Precision | Recall | PR-AUC | Brier | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AlwaysSQL | 0.9961 | 0.0000 | 0.0000 | 0.0000 | 0.0039 | 0.0039 | 258 | 0 | 1 | 0 |
| AlwaysGRAPH | 0.0039 | 0.0077 | 0.0039 | 1.0000 | 0.0039 | 0.9961 | 0 | 258 | 0 | 1 |
| TraversalRule | 0.5753 | 0.0179 | 0.0090 | 1.0000 | 0.0090 | 0.4247 | 148 | 110 | 0 | 1 |
| ThresholdBaseline | 0.8533 | 0.0000 | 0.0000 | 0.0000 | 0.0039 | 0.1467 | 221 | 37 | 1 | 0 |
| LogRegBalanced | 0.7954 | 0.0000 | 0.0000 | 0.0000 | 0.0041 | 0.1352 | 206 | 52 | 1 | 0 |
| DecisionTreeBalanced | 0.8224 | 0.0000 | 0.0000 | 0.0000 | 0.0039 | 0.1198 | 213 | 45 | 1 | 0 |
| XGBoostBalanced | 0.8224 | 0.0000 | 0.0000 | 0.0000 | 0.0106 | 0.1191 | 213 | 45 | 1 | 0 |
| XGBoostNoHistory | 0.8224 | 0.0000 | 0.0000 | 0.0000 | 0.0130 | 0.1191 | 213 | 45 | 1 | 0 |

## XGBoost Confidence Thresholds

| Threshold | Coverage | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.5 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.6 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.7 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.8 | 0.8842 | 0.0000 | 0.0000 | 0.0000 |
| 0.9 | 0.8842 | 0.0000 | 0.0000 | 0.0000 |

## XGBoost Per-Dataset Confusion

| Dataset | Rows | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| job_real_queries | 1 | 1 | 0 | 0 | 0 |
| ogb_real_queries | 100 | 99 | 0 | 1 | 0 |
| snb_bi_real_queries | 53 | 53 | 0 | 0 | 0 |
| snb_real_queries | 98 | 53 | 45 | 0 | 0 |
| tpcds_real_queries | 7 | 7 | 0 | 0 | 0 |

## Notes

- NoHistory models drop historical runtime features to test leakage risk.
- AlwaysSQL/AlwaysGRAPH quantify class-imbalance floor and ceiling behavior.