# Strict Ablation Summary

- Baseline F1: 0.5846
- Baseline std: 0.0433
- Model: xgboost
- Grouped CV: True
- Baseline 95% interval: [0.5186, 0.6744]
- CV folds: 5
- CV repeats: 3
- Max individual feature F1 drop: +0.0142
- Max feature-group F1 drop: +0.0094

## Top Individual Feature Drops

| Feature | F1 drop | Drop std | Drop CI low | Drop CI high |
|---|---:|---:|---:|---:|
| max_hops | +0.0142 | 0.0244 | +0.0000 | +0.0720 |
| op_count_filter | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| hist_runtime_variance | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| hist_avg_runtime_ms | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| estimated_traversal_ops | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| op_count_join | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| num_tables_joined | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| has_traversal | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| ast_depth | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| op_count_map | +0.0048 | 0.0179 | +0.0000 | +0.0467 |

## Feature Group Drops

| Group | Removed | F1 drop | Drop std | Drop CI low | Drop CI high |
|---|---:|---:|---:|---:|---:|
| graph_features | 5 | +0.0094 | 0.0190 | +0.0000 | +0.0550 |
| operation_counts | 5 | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| cardinality | 3 | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| data_characteristics | 4 | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| historical | 2 | +0.0048 | 0.0179 | +0.0000 | +0.0467 |
| structure | 3 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
