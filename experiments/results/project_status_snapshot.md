# Project Status Snapshot

Generated: 2026-04-03T23:32:02.954377+00:00

## Checklist

- [x] Dataset quality gate: PASS
- [x] Eval class coverage: SQL=31, GRAPH=27
- [x] Relevance (XGBoostBalanced F1>=0.95): 96.43%
- [x] Robustness (XGB eval F1>=0.95): 96.43%
- [x] Correctness executable pass rate (==100%): 100.00%
- [x] Non-flat feature evidence (ablation/group or permutation): feature=0.0000, group=0.0000, permutation=0.4762
- [ ] Native TPCH parquet available: FAIL

## Key Metrics

- AlwaysSQL baseline: accuracy=53.45%, F1(GRAPH)=0.0000
- XGBoostBalanced: accuracy=96.55%, F1(GRAPH)=0.9643, precision=0.9310, recall=1.0000

## Open Blockers

- Native TPCH gate is blocked: missing data/parquet/tpch/customer and/or data/parquet/tpch/orders
- To fix: provide TPCH raw .tbl files then run make data-tpch
