# Dataset Shift Evaluation

- Source: training_data/real_labeled_runs.csv

| Train | Eval | Status | Rows | F1 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|
| job_real_queries | ogb_real_queries | skipped_single_class_train | - | - | - | - |
| job_real_queries | snb_bi_real_queries | skipped_single_class_train | - | - | - | - |
| job_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| job_real_queries | tpcds_real_queries | skipped_single_class_train | - | - | - | - |
| ogb_real_queries | job_real_queries | skipped_single_class_train | - | - | - | - |
| ogb_real_queries | snb_bi_real_queries | skipped_single_class_train | - | - | - | - |
| ogb_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| ogb_real_queries | tpcds_real_queries | skipped_single_class_train | - | - | - | - |
| snb_bi_real_queries | job_real_queries | skipped_single_class_train | - | - | - | - |
| snb_bi_real_queries | ogb_real_queries | skipped_single_class_train | - | - | - | - |
| snb_bi_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| snb_bi_real_queries | tpcds_real_queries | skipped_single_class_train | - | - | - | - |
| snb_real_queries | job_real_queries | ok | 18 | 0.0000 | 0.0000 | 0.0000 |
| snb_real_queries | ogb_real_queries | ok | 176 | 0.0000 | 0.0000 | 0.0000 |
| snb_real_queries | snb_bi_real_queries | ok | 96 | 0.0000 | 0.0000 | 0.0000 |
| snb_real_queries | tpcds_real_queries | ok | 16 | 0.0000 | 0.0000 | 0.0000 |
| tpcds_real_queries | job_real_queries | skipped_single_class_train | - | - | - | - |
| tpcds_real_queries | ogb_real_queries | skipped_single_class_train | - | - | - | - |
| tpcds_real_queries | snb_bi_real_queries | skipped_single_class_train | - | - | - | - |
| tpcds_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |