# Dataset Shift Evaluation

- Source: training_data/real_labeled_runs_strict_curated.csv

| Mode | Train | Eval | Status | Rows | F1 | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| one_to_one | ogb_real_queries | snb_real_queries | ok | 238 | 0.9741 | 0.9496 | 1.0000 |
| one_to_one | snb_real_queries | ogb_real_queries | ok | 50 | 0.9796 | 0.9600 | 1.0000 |
| leave_one_out | all_except:ogb_real_queries | ogb_real_queries | ok | 50 | 0.9796 | 0.9600 | 1.0000 |
| leave_one_out | all_except:snb_real_queries | snb_real_queries | ok | 238 | 0.9741 | 0.9496 | 1.0000 |
| grouped_domains | domain:graph | domain:mixed | ok | 238 | 0.9741 | 0.9496 | 1.0000 |
| grouped_domains | domain:mixed | domain:graph | ok | 50 | 0.9796 | 0.9600 | 1.0000 |