# Dataset Shift Evaluation

- Source: training_data/real_labeled_runs.csv

| Mode | Train | Eval | Status | Rows | F1 | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| one_to_one | job_real_queries | ogb_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | job_real_queries | snb_bi_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | job_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | job_real_queries | tpcds_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | ogb_real_queries | job_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | ogb_real_queries | snb_bi_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | ogb_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | ogb_real_queries | tpcds_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | snb_bi_real_queries | job_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | snb_bi_real_queries | ogb_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | snb_bi_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | snb_bi_real_queries | tpcds_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | snb_real_queries | job_real_queries | ok | 18 | 0.0000 | 0.0000 | 0.0000 |
| one_to_one | snb_real_queries | ogb_real_queries | ok | 176 | 0.0000 | 0.0000 | 0.0000 |
| one_to_one | snb_real_queries | snb_bi_real_queries | ok | 96 | 0.0000 | 0.0000 | 0.0000 |
| one_to_one | snb_real_queries | tpcds_real_queries | ok | 16 | 0.0000 | 0.0000 | 0.0000 |
| one_to_one | tpcds_real_queries | job_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | tpcds_real_queries | ogb_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | tpcds_real_queries | snb_bi_real_queries | skipped_single_class_train | - | - | - | - |
| one_to_one | tpcds_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| leave_one_out | all_except:job_real_queries | job_real_queries | ok | 18 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_out | all_except:ogb_real_queries | ogb_real_queries | ok | 176 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_out | all_except:snb_bi_real_queries | snb_bi_real_queries | ok | 96 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_out | all_except:snb_real_queries | snb_real_queries | skipped_single_class_train | - | - | - | - |
| leave_one_out | all_except:tpcds_real_queries | tpcds_real_queries | ok | 16 | 0.0000 | 0.0000 | 0.0000 |
| grouped_domains | domain:graph | domain:mixed | skipped_single_class_train | - | - | - | - |
| grouped_domains | domain:graph | domain:sql | skipped_single_class_train | - | - | - | - |
| grouped_domains | domain:mixed | domain:graph | ok | 176 | 0.0000 | 0.0000 | 0.0000 |
| grouped_domains | domain:mixed | domain:sql | ok | 34 | 0.0000 | 0.0000 | 0.0000 |
| grouped_domains | domain:sql | domain:graph | skipped_single_class_train | - | - | - | - |
| grouped_domains | domain:sql | domain:mixed | skipped_single_class_train | - | - | - | - |