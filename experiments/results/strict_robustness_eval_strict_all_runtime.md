# Strict Robustness Evaluation

- Train: training_data/fixed_train_base_strict_all.csv
- Eval: training_data/fixed_eval_set_strict_all.csv
- Transfer source: training_data/real_labeled_runs_strict_all.csv

## Core Strict Metric

- XGBoost eval F1: 0.7660

## Bootstrap 95% CI (Eval)

- F1: [0.6574, 0.8496] (median 0.7619)
- Precision: [0.5167, 0.7619] (median 0.6429)
- Recall: [0.8666, 1.0000] (median 0.9487)

## Label-Permutation Sanity

- Mode: within_group:query_id
- Permuted-label mean F1: 0.1970 ± 0.1824
- Permuted-label range: [0.0000, 0.5773]

## Overlap Audit

- Overlap on source_row_id: 0
- Overlap on query_id: 0
- Overlap on sub_id: 2

## Top Permutation Importance (Eval F1 Drop)

| Feature | Mean Drop | Std | Max Drop |
|---|---:|---:|---:|
| max_hops | 0.3711 | 0.0819 | 0.5905 |
| op_count_traversal | 0.2563 | 0.0486 | 0.3800 |
| input_cardinality_log | 0.0510 | 0.0334 | 0.1081 |
| estimated_traversal_ops | 0.0260 | 0.0153 | 0.0576 |
| max_degree | 0.0060 | 0.0069 | 0.0237 |
| op_count_filter | 0.0000 | 0.0000 | 0.0000 |
| op_count_join | 0.0000 | 0.0000 | 0.0000 |
| op_count_aggregate | 0.0000 | 0.0000 | 0.0000 |
| op_count_map | 0.0000 | 0.0000 | 0.0000 |
| ast_depth | 0.0000 | 0.0000 | 0.0000 |

## Cross-Dataset Transfer

| Train | Eval | Rows | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| ogb_real_queries | job_real_queries | 18 | 0.0000 | 0.0000 | 0.0000 |
| ogb_real_queries | snb_bi_real_queries | 96 | 0.0000 | 0.0000 | 0.0000 |
| ogb_real_queries | snb_real_queries | 670 | 0.4345 | 0.3642 | 0.5385 |
| ogb_real_queries | tpcds_real_queries | 16 | 0.0000 | 0.0000 | 0.0000 |
| snb_real_queries | job_real_queries | 18 | 0.0000 | 0.0000 | 0.0000 |
| snb_real_queries | ogb_real_queries | 226 | 0.4835 | 0.3284 | 0.9167 |
| snb_real_queries | snb_bi_real_queries | 96 | 0.0000 | 0.0000 | 0.0000 |
| snb_real_queries | tpcds_real_queries | 16 | 0.0000 | 0.0000 | 0.0000 |
