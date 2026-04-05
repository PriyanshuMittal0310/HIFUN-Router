# Strict Ablation Summary

- Baseline F1: 0.9778
- Baseline std: 0.0000
- Baseline 95% interval: [0.9778, 0.9778]
- CV folds: 5
- CV repeats: 2
- Max individual feature F1 drop: +0.0000
- Max feature-group F1 drop: +0.0000

## Top Individual Feature Drops

| Feature | F1 drop | Drop std | Drop CI low | Drop CI high |
|---|---:|---:|---:|---:|
| op_count_filter | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| op_count_join | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| hist_runtime_variance | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| hist_avg_runtime_ms | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| estimated_traversal_ops | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| estimated_shuffle_bytes_log | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| join_fanout | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| has_index | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| num_projected_columns | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| degree_skew | +0.0000 | 0.0000 | +0.0000 | +0.0000 |

## Feature Group Drops

| Group | Removed | F1 drop | Drop std | Drop CI low | Drop CI high |
|---|---:|---:|---:|---:|---:|
| operation_counts | 5 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| structure | 3 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| graph_features | 5 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| cardinality | 3 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| data_characteristics | 4 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
| historical | 2 | +0.0000 | 0.0000 | +0.0000 | +0.0000 |
