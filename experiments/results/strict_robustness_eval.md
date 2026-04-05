# Strict Robustness Evaluation

- Train: training_data/fixed_train_base_strict.csv
- Eval: training_data/fixed_eval_set_strict.csv
- Transfer source: training_data/real_labeled_runs_strict_curated.csv

## Core Strict Metric

- XGBoost eval F1: 0.9818

## Bootstrap 95% CI (Eval)

- F1: [0.9388, 1.0000] (median 0.9831)
- Precision: [0.8846, 1.0000] (median 0.9667)
- Recall: [1.0000, 1.0000] (median 1.0000)

## Label-Permutation Sanity

- Permuted-label mean F1: 0.5333 ± 0.3022
- Permuted-label range: [0.0000, 0.9818]

## Top Permutation Importance (Eval F1 Drop)

| Feature | Mean Drop | Std | Max Drop |
|---|---:|---:|---:|
| op_count_traversal | 0.4885 | 0.0772 | 0.6545 |
| op_count_filter | 0.0000 | 0.0000 | 0.0000 |
| op_count_join | 0.0000 | 0.0000 | 0.0000 |
| op_count_aggregate | 0.0000 | 0.0000 | 0.0000 |
| op_count_map | 0.0000 | 0.0000 | 0.0000 |
| ast_depth | 0.0000 | 0.0000 | 0.0000 |
| has_traversal | 0.0000 | 0.0000 | 0.0000 |
| max_hops | 0.0000 | 0.0000 | 0.0000 |
| input_cardinality_log | 0.0000 | 0.0000 | 0.0000 |
| output_cardinality_log | 0.0000 | 0.0000 | 0.0000 |

## Cross-Dataset Transfer

| Train | Eval | Rows | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| ogb_real_queries | snb_real_queries | 238 | 0.9741 | 0.9496 | 1.0000 |
| snb_real_queries | ogb_real_queries | 50 | 0.9796 | 0.9600 | 1.0000 |
