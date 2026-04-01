# Routing Relevance Evaluation

- Train dataset: /home/mitta/HIFUN-Router-clone/training_data/fixed_train_balanced.csv
- Eval dataset: /home/mitta/HIFUN-Router-clone/training_data/fixed_eval_set.csv
- Train rows: 1174
- Eval rows: 148
- Train label distribution: {'SQL': 587, 'GRAPH': 587}
- Eval label distribution: {'SQL': 147, 'GRAPH': 1}

## Model Comparison

| Model | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AlwaysSQL | 0.9932 | 0.0000 | 0.0000 | 0.0000 | 147 | 0 | 1 | 0 |
| AlwaysGRAPH | 0.0068 | 0.0134 | 0.0068 | 1.0000 | 0 | 147 | 0 | 1 |
| TraversalRule | 0.6216 | 0.0345 | 0.0175 | 1.0000 | 91 | 56 | 0 | 1 |
| ThresholdBaseline | 0.7432 | 0.0500 | 0.0256 | 1.0000 | 109 | 38 | 0 | 1 |
| LogRegBalanced | 0.8176 | 0.0690 | 0.0357 | 1.0000 | 120 | 27 | 0 | 1 |
| DecisionTreeBalanced | 0.8176 | 0.0690 | 0.0357 | 1.0000 | 120 | 27 | 0 | 1 |
| XGBoostBalanced | 0.8176 | 0.0690 | 0.0357 | 1.0000 | 120 | 27 | 0 | 1 |
| XGBoostNoHistory | 0.8176 | 0.0690 | 0.0357 | 1.0000 | 120 | 27 | 0 | 1 |

## Notes

- NoHistory models drop historical runtime features to test leakage risk.
- AlwaysSQL/AlwaysGRAPH quantify class-imbalance floor and ceiling behavior.