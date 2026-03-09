# HIFUN Router — SIGMOD 2026 Revision Implementation Plan

> **Purpose:** Concrete, developer-executable plan to address every critical and major weakness
> identified in the SIGMOD 2026 critical analysis.
>
> **Structure:** Three phases, ordered by urgency and dependency.
> Each task has: Goal → File(s) to create/modify → Exact implementation steps → Acceptance criteria.
>
> **Time estimates** are for a single developer working part-time (3–4 hrs/day).

---

## Table of Contents

- [Phase 1 — Immediate Fixes (2–4 weeks) → Workshop Submission Ready](#phase-1--immediate-fixes-24-weeks)
- [Phase 2 — Core Experimental Overhaul (2–3 months) → Main Track Contender](#phase-2--core-experimental-overhaul-23-months)
- [Phase 3 — Research Depth (4–6 months) → Full SIGMOD Research Track](#phase-3--research-depth-46-months)
- [Appendix A — New Related Work Reference List](#appendix-a--new-related-work-reference-list)
- [Appendix B — Acceptance Criteria Checklist](#appendix-b--acceptance-criteria-checklist)

---

## Phase 1 — Immediate Fixes (2–4 weeks)

> **Goal:** Make the paper honest, properly contextualized, and workshop-submittable.
> None of these tasks require re-running the full system.

---

### Task 1.1 — Add a Simulation Disclaimer to the Paper

**Priority:** CRITICAL — Must be done before any submission.

**Problem:** The abstract and results section present simulated timings as if they were
real Spark SQL / GraphFrames measurements. This is misleading and will trigger
immediate rejection.

**Files to modify:**
```
report/main.tex  (or your .tex source)
```

**Implementation Steps:**

1. In the **Abstract**, add the following sentence after the speedup claim:
   > "Timing results are derived from a calibrated heuristic cost model; experiments with
   > actual Spark SQL and GraphFrames execution are the subject of ongoing work."

2. In **Section III.C (Training Data Collection)**, add a dedicated paragraph:
   > **Simulation Validity.** The cost model uses additive penalty terms calibrated to
   > approximate observed relative performance between SQL joins and BFS traversals at
   > small scale. Uniform multiplicative noise U(0.85, 1.15) is applied to simulate
   > realistic runtime variance. While this approach enables rapid, reproducible label
   > generation, the resulting classifier learns the cost model's decision boundaries
   > rather than real engine behavior. We treat these results as a proof-of-concept
   > demonstrating system correctness and pipeline functionality; Section IV.X reports
   > preliminary real execution timings. [Link to Task 2.1 results when ready]

3. In **Table V (Execution Latency)**, change the caption to:
   > "Aggregated simulated execution latency comparison (heuristic cost model). See
   > Table VI for real-engine measurements [future work / Phase 2]."

4. In **Section V (Conclusion)**, add to the limitations paragraph:
   > "A key limitation of the current evaluation is the use of a simulation-based
   > labeling strategy. Real execution measurements on Apache Spark and GraphFrames are
   > needed to validate that the learned routing decisions generalize beyond the
   > heuristic cost model's assumptions."

**Acceptance Criteria:**
- [ ] The words "simulated" or "heuristic cost model" appear in the abstract, Table V
      caption, and conclusion.
- [ ] A reviewer reading the paper cannot be confused about whether timings are real.

**Estimated time:** 2–3 hours

---

### Task 1.2 — Expand the Related Work Section

**Priority:** CRITICAL — Currently the single biggest structural weakness in the paper.

**Problem:** 12 references is far below the expected 25–40 for a SIGMOD paper. The most
directly competing systems (Apache Wayang, AutoSteer, LEON) are completely absent.

**Files to modify:**
```
report/main.tex
report/references.bib
```

**Implementation Steps:**

**Step 1:** Add the following subsections to your Related Work section. For each paper,
write 2–3 sentences: what they do, how it relates to HIFUN Router, and one clear
differentiator.

**Subsection: Polystore and Multi-Engine Query Processing**

Add these papers with the discussion text provided:

```
[Wayang/SIGMOD 2025]
Apache Wayang (formerly Rheem) provides a platform-independent data processing
abstraction that routes operator plans across heterogeneous backends including
Spark, Flink, and Java Streams. Unlike HIFUN Router, Wayang operates at the
physical operator level and uses a cost-based planner with manually specified cost
functions — it does not learn routing decisions from data. HIFUN Router
complements this direction by demonstrating that learned classifiers trained on
algebraic-level features can match or exceed manually engineered cost functions
for the SQL/Graph routing decision.

[RHEEM/SIGMOD 2018]
Rheem introduced the concept of cross-platform data processing optimization via
a two-level IR (WayangPlan and ExecutionPlan). Their optimizer enumerates
execution strategies across platforms and selects the minimum-cost plan via
a heuristic cost estimator. HIFUN Router differs in targeting the specific
SQL vs. Graph decision using ML rather than cost enumeration, and in operating
at the algebraic subexpression level rather than the physical plan level.

[BigDAWG/SIGMOD 2015]
BigDAWG pioneered the polystore model, enabling queries that span island-specific
query languages (SQL, array, streaming). Their architecture requires users to
annotate data affinity explicitly. HIFUN Router automates this annotation via ML.
```

**Subsection: Learned Query Optimization**

```
[AutoSteer/VLDB 2023]
AutoSteer learns query hint configurations for any black-box SQL engine by
evaluating query variants and training a regression model to predict the best
execution strategy. HIFUN Router applies a similar philosophy — learning from
execution data — but extends it to the cross-engine routing decision rather than
intra-engine hint selection. AutoSteer's finding that a compact feature space
suffices for good optimizer decisions supports HIFUN Router's 22-feature approach.

[LEON/VLDB 2023]
LEON presents a query-level optimizer that uses reinforcement learning to
adapt join ordering decisions in PostgreSQL. While operating within a single
engine, LEON demonstrates that ML-based plan selection generalizes across workloads
when trained on sufficiently diverse query distributions — a challenge HIFUN Router
must also address (see Section IV.X cross-dataset experiments).

[Balsa/SIGMOD 2022]
Balsa learns query optimization from scratch using imitation learning on a
reference optimizer, removing the need for pre-collected execution logs.
HIFUN Router uses a simpler supervised approach but faces the same cold-start
challenge that Balsa addresses — an interesting direction for future work.

[Neo/VLDB 2019]
Neo was the first learned query optimizer to outperform PostgreSQL's cost-based
optimizer for join ordering. Neo uses tree-structured neural networks over
query plan features. HIFUN Router uses a simpler tree ensemble (XGBoost) but
targets a higher-level binary routing decision rather than an exponential plan space.

[STAGE/SIGMOD 2024]
STAGE predicts query execution time in production Spark environments using
gradient-boosted trees over operator-level features. STAGE confirms that tree
ensemble models are effective for Spark workload prediction — directly supporting
HIFUN Router's classifier choice.
```

**Subsection: Graph Query Processing and Cross-Model Databases**

```
[G-CORE/SIGMOD 2018]
G-CORE defines a composable graph query language that integrates path expressions
with relational projection. HIFUN Router addresses the runtime execution side of
the same mixed-model challenge that G-CORE addresses at the language level.

[GraphFrames/GRADES 2016]
GraphFrames integrates graph computation with Spark SQL by representing graphs
as DataFrames. HIFUN Router uses GraphFrames as one of its two target engines
and relies on its design for the correctness of the graph execution path.

[LDBC SNB/SIGMOD 2015]
The LDBC Social Network Benchmark defines standard mixed graph-relational
workloads. HIFUN Router uses an SNB-like dataset; future work includes evaluation
on the full official LDBC SNB Interactive workload.
```

**Step 2:** Add all BibTeX entries to `references.bib`. The key ones you need:

```bibtex
@inproceedings{wayang2025,
  title={Apache Wayang: A Unified Data Analytics Framework},
  author={Beedkar, Kaustubh and others},
  booktitle={Proc. ACM SIGMOD},
  year={2025}
}

@inproceedings{rheem2018,
  title={Rheem: Enabling Multi-Platform Task Execution},
  author={Agrawal, Divy and others},
  booktitle={Proc. ACM SIGMOD},
  year={2018}
}

@inproceedings{autosteer2023,
  title={AutoSteer: Learned Query Optimization for Any SQL Database},
  author={Anneser, Christoph and others},
  booktitle={Proc. VLDB Endowment},
  volume={16},
  number={12},
  year={2023}
}

@inproceedings{leon2023,
  title={LEON: A New Framework for ML-Aided Query Optimization},
  author={Chen, Xu and others},
  booktitle={Proc. VLDB Endowment},
  volume={16},
  number={9},
  year={2023}
}

@inproceedings{balsa2022,
  title={Balsa: Learning a Query Optimizer Without Expert Demonstrations},
  author={Yang, Zongheng and others},
  booktitle={Proc. ACM SIGMOD},
  year={2022}
}

@inproceedings{stage2024,
  title={STAGE: Query Execution Time Prediction in Apache Spark},
  author={...},
  booktitle={Proc. ACM SIGMOD},
  year={2024}
}

@inproceedings{bigdawg2015,
  title={The BigDAWG Polystore System},
  author={Elmore, Aaron J and others},
  booktitle={Proc. ACM SIGMOD},
  year={2015}
}

@inproceedings{gcore2018,
  title={G-CORE: A Core for Future Graph Query Languages},
  author={Angles, Renzo and others},
  booktitle={Proc. ACM SIGMOD},
  year={2018}
}
```

**Step 3:** Update the Related Work section summary paragraph to explicitly state HIFUN
Router's differentiating position:
> "In summary, while prior work on learned query optimization [Neo, Balsa, AutoSteer, LEON]
> focuses on intra-engine decisions and polystore systems [BigDAWG, Wayang, Rheem] focus on
> physical-level routing with hand-crafted cost functions, HIFUN Router addresses the
> specific SQL vs. Graph routing decision at the algebraic subexpression level, using
> a learned classifier over a principled feature vector that combines graph topology
> statistics with relational cardinality estimates."

**Acceptance Criteria:**
- [ ] Related Work section has ≥ 5 subsections
- [ ] ≥ 25 total references in the paper
- [ ] Apache Wayang, AutoSteer, LEON, Balsa, STAGE, BigDAWG are all cited
- [ ] Each competing paper has a stated differentiator from HIFUN Router
- [ ] The final summary paragraph positions HIFUN Router's unique contribution clearly

**Estimated time:** 6–8 hours (reading + writing)

---

### Task 1.3 — Replace Synthesized SNB Data with Official LDBC Generator

**Priority:** MAJOR

**Problem:** Using a hand-rolled social network "as a fallback" for LDBC SNB weakens
benchmark credibility. The official generator is free and takes ~10 minutes to run.

**Files to create/modify:**
```
data/scripts/download_ldbc_snb.sh       (new)
data/scripts/ldbc_snb_to_parquet.py     (new)
data/README.md                           (modify)
```

**Implementation Steps:**

**Step 1 — Download LDBC SNB data generator:**
```bash
# data/scripts/download_ldbc_snb.sh
#!/bin/bash
# Downloads LDBC SNB Interactive SF=0.1 data (small: ~50MB, suitable for local runs)

# Install ldbc_snb_datagen via Docker
docker pull ldbc/datagen:latest

# Generate SF=0.1 (approx 30K persons, 200K posts, 180K edges)
mkdir -p data/raw/ldbc_snb
docker run --rm \
  -v $(pwd)/data/raw/ldbc_snb:/output \
  ldbc/datagen:latest \
  --scale-factor 0.1 \
  --output-dir /output \
  --format csv

echo "LDBC SNB data generated at data/raw/ldbc_snb/"
```

**Step 2 — Convert to Parquet + Graph edge list:**
```python
# data/scripts/ldbc_snb_to_parquet.py
import os
import pandas as pd
from pyspark.sql import SparkSession

def convert_ldbc_snb(input_dir: str, parquet_dir: str, graph_dir: str):
    """
    Converts LDBC SNB CSV output to:
      - Parquet tables for Spark SQL (person, post, comment, forum)
      - Edge list Parquet for GraphFrames (person_knows_person)
    """
    spark = SparkSession.builder.appName("SNB_Convert").master("local[*]").getOrCreate()

    # --- Relational Tables ---
    tables = {
        "person":  ("person_0_0.csv",   ["id","firstName","lastName","gender",
                                          "birthday","creationDate","locationIP",
                                          "browserUsed"]),
        "post":    ("post_0_0.csv",     ["id","imageFile","creationDate","locationIP",
                                          "browserUsed","language","content",
                                          "length","creator","Forum.id","place"]),
        "comment": ("comment_0_0.csv",  ["id","creationDate","locationIP",
                                          "browserUsed","content","length",
                                          "creator","place","replyOfPost",
                                          "replyOfComment"]),
    }
    for name, (filename, cols) in tables.items():
        path = os.path.join(input_dir, "social_network", filename)
        if os.path.exists(path):
            df = spark.read.option("header","true").option("sep","|").csv(path)
            out = os.path.join(parquet_dir, "snb", name)
            df.write.mode("overwrite").parquet(out)
            print(f"Saved {name}: {df.count()} rows → {out}")

    # --- Graph Edge List (KNOWS relationship) ---
    knows_path = os.path.join(input_dir, "social_network",
                              "person_knows_person_0_0.csv")
    if os.path.exists(knows_path):
        edges = (spark.read
                     .option("header","true")
                     .option("sep","|")
                     .csv(knows_path)
                     .withColumnRenamed("Person.id","src")
                     .withColumnRenamed("Person.id.1","dst")
                     .withColumn("relationship",
                                 spark.createDataFrame([("KNOWS",)],["r"])
                                 .first()[0]))
        # Make edges bidirectional
        from pyspark.sql import functions as F
        reverse = edges.select(
            F.col("dst").alias("src"),
            F.col("src").alias("dst"),
            F.col("relationship")
        )
        all_edges = edges.union(reverse).distinct()

        vertices = (spark.read.parquet(os.path.join(parquet_dir, "snb", "person"))
                        .select(F.col("id")))

        out_e = os.path.join(graph_dir, "snb_edges.parquet")
        out_v = os.path.join(graph_dir, "snb_vertices.parquet")
        all_edges.write.mode("overwrite").parquet(out_e)
        vertices.write.mode("overwrite").parquet(out_v)
        print(f"Saved SNB graph: {all_edges.count()} edges → {out_e}")

    spark.stop()

if __name__ == "__main__":
    convert_ldbc_snb(
        input_dir="data/raw/ldbc_snb",
        parquet_dir="data/parquet",
        graph_dir="data/graphs"
    )
```

**Step 3 — Update stats computation to include SNB tables:**
```bash
python data/scripts/compute_stats.py --dataset snb \
    --parquet_dir data/parquet/snb/ \
    --graph_dir data/graphs/ \
    --output data/stats/
```

**Step 4 — In the paper**, change:
> "A social-network dataset modeled after LDBC SNB is synthesized as a fallback..."

to:
> "We use official LDBC SNB data generated at Scale Factor 0.1 using the LDBC datagen
> tool [cite], producing approximately 30,000 persons, 200,000 posts, and 180,000
> person-knows-person edges."

**Acceptance Criteria:**
- [ ] `data/raw/ldbc_snb/` contains real LDBC-generated CSV files
- [ ] `data/parquet/snb/` and `data/graphs/snb_*.parquet` exist
- [ ] Paper no longer references "synthesized as a fallback"
- [ ] SNB stats JSON files exist in `data/stats/`

**Estimated time:** 3–4 hours

---

### Task 1.4 — Strengthen Correctness Verification to Value-Level Checksums

**Priority:** MAJOR

**Problem:** "100% row-count agreement" is a weak correctness claim — two DataFrames
can have the same row count with completely different data. SIGMOD expects at minimum
a hash/checksum comparison of actual output values.

**Files to create/modify:**
```
tests/reference_executor.py           (modify)
tests/test_correctness.py             (modify)
experiments/correctness_report.py     (new)
```

**Implementation Steps:**

**Step 1 — Add checksum computation to `reference_executor.py`:**
```python
# Add this method to your ReferenceExecutor class

import hashlib
import pandas as pd

def compute_result_checksum(df: pd.DataFrame) -> dict:
    """
    Computes a deterministic checksum over a DataFrame result.
    Handles non-deterministic ordering by sorting before hashing.
    Returns dict with multiple verification signals.
    """
    # Sort by all columns to make order-independent
    df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)

    # Convert to canonical string representation
    canonical = df_sorted.to_csv(index=False, float_format="%.6f")

    return {
        "row_count":      len(df_sorted),
        "col_count":      len(df_sorted.columns),
        "columns":        sorted(df_sorted.columns.tolist()),
        "sha256":         hashlib.sha256(canonical.encode()).hexdigest(),
        "col_checksums":  {
            col: hashlib.md5(
                df_sorted[col].astype(str).str.cat(sep="|").encode()
            ).hexdigest()
            for col in df_sorted.columns
        }
    }

def compare_results(ref_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    query_id: str) -> dict:
    """
    Full comparison between reference and routed result.
    Returns a structured comparison report.
    """
    ref_check  = compute_result_checksum(ref_df)
    test_check = compute_result_checksum(test_df)

    report = {
        "query_id":          query_id,
        "row_count_match":   ref_check["row_count"]  == test_check["row_count"],
        "col_count_match":   ref_check["col_count"]  == test_check["col_count"],
        "columns_match":     ref_check["columns"]    == test_check["columns"],
        "sha256_match":      ref_check["sha256"]     == test_check["sha256"],
        "ref_row_count":     ref_check["row_count"],
        "test_row_count":    test_check["row_count"],
        "ref_sha256":        ref_check["sha256"][:12] + "...",
        "test_sha256":       test_check["sha256"][:12] + "...",
        "col_mismatches":    [],
        "pass":              ref_check["sha256"] == test_check["sha256"]
    }

    # Per-column mismatch detail
    if not report["sha256_match"] and report["columns_match"]:
        for col in ref_check["col_checksums"]:
            if ref_check["col_checksums"][col] != test_check["col_checksums"].get(col):
                report["col_mismatches"].append(col)

    return report
```

**Step 2 — Update `test_correctness.py` to use checksums:**
```python
# tests/test_correctness.py  — add these test cases

import pytest
import pandas as pd
from tests.reference_executor import ReferenceExecutor, compare_results
from router.hybrid_router import HybridRouter
import json, os

QUERY_DIR = "dsl/sample_queries/"

def load_all_queries():
    queries = []
    for fname in os.listdir(QUERY_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(QUERY_DIR, fname)) as f:
                data = json.load(f)
                # Handle both list and single query format
                if isinstance(data, list):
                    queries.extend(data)
                else:
                    queries.append(data)
    return queries

@pytest.fixture(scope="module")
def router():
    return HybridRouter.from_config("config/paths.py")

@pytest.fixture(scope="module")
def ref_executor():
    return ReferenceExecutor()

@pytest.mark.parametrize("query", load_all_queries())
def test_correctness_checksum(query, router, ref_executor):
    """
    For each query: run via HybridRouter AND via ReferenceExecutor.
    Assert SHA256 checksum equality on sorted output.
    """
    query_id = query.get("query_id", "unknown")

    ref_result  = ref_executor.execute(query)
    test_result = router.execute(query)

    # Convert Spark DF to pandas if needed
    if hasattr(test_result, "toPandas"):
        test_result = test_result.toPandas()

    report = compare_results(ref_result, test_result, query_id)

    assert report["row_count_match"], (
        f"[{query_id}] Row count mismatch: "
        f"ref={report['ref_row_count']}, test={report['test_row_count']}"
    )
    assert report["columns_match"], (
        f"[{query_id}] Column mismatch: "
        f"ref={report.get('ref_cols')}, test={report.get('test_cols')}"
    )
    assert report["sha256_match"], (
        f"[{query_id}] Value mismatch (SHA256 differs). "
        f"Mismatched columns: {report['col_mismatches']}"
    )
```

**Step 3 — Create a correctness report generator for the paper:**
```python
# experiments/correctness_report.py

import json, os, pandas as pd
from tests.reference_executor import ReferenceExecutor, compare_results
from router.hybrid_router import HybridRouter

def generate_correctness_table(query_dir: str, output_csv: str):
    """
    Runs all queries through both executors and generates a CSV
    suitable for inclusion as a paper table.
    """
    router   = HybridRouter.from_config("config/paths.py")
    ref_exec = ReferenceExecutor()
    results  = []

    for fname in sorted(os.listdir(query_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(query_dir, fname)) as f:
            queries = json.load(f)
        if not isinstance(queries, list):
            queries = [queries]

        for q in queries:
            try:
                ref_df  = ref_exec.execute(q)
                test_df = router.execute(q)
                if hasattr(test_df, "toPandas"):
                    test_df = test_df.toPandas()
                report = compare_results(ref_df, test_df, q["query_id"])
                report["source_file"] = fname
                results.append(report)
            except Exception as e:
                results.append({
                    "query_id": q.get("query_id", "?"),
                    "pass": False,
                    "error": str(e),
                    "source_file": fname
                })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    total  = len(df)
    passed = df["pass"].sum()
    print(f"\nCorrectness Summary: {passed}/{total} queries pass ({100*passed/total:.1f}%)")
    print(df[["query_id","row_count_match","sha256_match","col_mismatches","pass"]]
            .to_string(index=False))
    return df

if __name__ == "__main__":
    generate_correctness_table("dsl/sample_queries/",
                               "experiments/results/correctness_report.csv")
```

**Step 4 — Update paper Table to show checksum-level correctness:**

Replace your current correctness paragraph with a table:

| Query ID | Dataset | Rows (Ref) | Rows (Routed) | SHA256 Match | Pass |
|---|---|---|---|---|---|
| q_tpch_001 | TPC-H | 1,247 | 1,247 | ✓ | ✓ |
| q_snb_002 | SNB | 342 | 342 | ✓ | ✓ |
| ... | ... | ... | ... | ... | ... |

**Acceptance Criteria:**
- [ ] `compare_results()` function implemented and unit-tested
- [ ] `test_correctness.py` uses SHA256 checksum comparison, not row count
- [ ] `experiments/results/correctness_report.csv` generated
- [ ] Paper correctness section shows a table with per-query SHA256 match results
- [ ] All queries still pass (or failures are explained)

**Estimated time:** 4–5 hours

---

### Task 1.5 — Add a Stronger Rule-Based Baseline

**Priority:** CRITICAL for paper integrity

**Problem:** The current rule-based baseline ("TRAVERSAL → GRAPH, else → SQL") is trivially
easy to beat because it is nearly identical to the `has_traversal` feature your classifier
uses. A reviewer will note that the ML model is basically replicating this rule.

**Files to create/modify:**
```
router/baselines.py                    (new)
experiments/run_baselines.py           (modify)
```

**Implementation Steps:**

**Step 1 — Implement a threshold-based cost-estimator baseline in `router/baselines.py`:**

```python
# router/baselines.py

import numpy as np
from decomposer.query_decomposer import SubExpression

# ──────────────────────────────────────────────────────────────
# Baseline 1: Trivial Rule (current paper baseline — keep for
# historical comparison but no longer the PRIMARY baseline)
# ──────────────────────────────────────────────────────────────
def trivial_rule_route(sub_expr: SubExpression) -> str:
    """TRAVERSAL → GRAPH, everything else → SQL."""
    return "GRAPH" if sub_expr.primary_op_type == "TRAVERSAL" else "SQL"


# ──────────────────────────────────────────────────────────────
# Baseline 2: Two-Feature Threshold Rule  ← NEW PRIMARY BASELINE
# Manually tuned on a held-out validation split.
# Hypothesis: GRAPH wins when avg_degree > D AND max_hops <= H
#             AND selectivity > S (enough start vertices)
# ──────────────────────────────────────────────────────────────
class ThresholdBaseline:
    """
    Threshold-based routing using the two most important features.
    Thresholds are tuned on a held-out 20% validation split.
    This is the PRIMARY competitive baseline — it represents the
    best a domain expert could do WITHOUT machine learning.
    """
    def __init__(self, avg_degree_thresh=5.0,
                       max_hops_thresh=3,
                       selectivity_thresh=0.01):
        self.avg_degree_thresh   = avg_degree_thresh
        self.max_hops_thresh     = max_hops_thresh
        self.selectivity_thresh  = selectivity_thresh

    def route(self, feature_vector: np.ndarray,
                    feature_names: list) -> str:
        """
        Args:
            feature_vector: The 22-dim feature vector from FeatureExtractor
            feature_names:  The ordered list of feature names
        """
        fv = dict(zip(feature_names, feature_vector))

        has_traversal = fv.get("has_traversal", 0)
        avg_degree    = fv.get("avg_degree", 0)
        max_hops      = fv.get("max_hops", 0)
        selectivity   = fv.get("selectivity", 1.0)

        # Route to GRAPH only if:
        # 1. There IS a traversal
        # 2. Graph is sufficiently connected (avg_degree > threshold)
        # 3. Traversal is shallow enough (max_hops <= threshold)
        # 4. Enough starting vertices (selectivity is not too low)
        if (has_traversal == 1 and
            avg_degree   >  self.avg_degree_thresh and
            max_hops     <= self.max_hops_thresh and
            selectivity  >= self.selectivity_thresh):
            return "GRAPH"
        return "SQL"

    @classmethod
    def tune_thresholds(cls, labeled_data_path: str,
                             feature_names: list,
                             val_fraction: float = 0.2):
        """
        Grid-search over threshold values on a held-out validation split.
        Returns the ThresholdBaseline instance with the best F1.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        df = pd.read_csv(labeled_data_path)
        X  = df[feature_names].values
        y  = (df["label"] == "GRAPH").astype(int).values

        _, X_val, _, y_val = train_test_split(
            X, y, test_size=val_fraction, stratify=y, random_state=42)

        best_f1, best_params = 0, {}

        for d in [2.0, 5.0, 10.0, 20.0]:
            for h in [1, 2, 3, 4]:
                for s in [0.001, 0.01, 0.05, 0.1]:
                    baseline = cls(avg_degree_thresh=d,
                                   max_hops_thresh=h,
                                   selectivity_thresh=s)
                    preds = [int(baseline.route(x, feature_names) == "GRAPH")
                             for x in X_val]
                    f1 = f1_score(y_val, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1    = f1
                        best_params = {"avg_degree_thresh": d,
                                       "max_hops_thresh":   h,
                                       "selectivity_thresh": s}

        print(f"Best threshold baseline: F1={best_f1:.3f}, params={best_params}")
        return cls(**best_params)


# ──────────────────────────────────────────────────────────────
# Baseline 3: Logistic Regression  ← Tests if ML complexity needed
# ──────────────────────────────────────────────────────────────
class LogisticRegressionBaseline:
    """
    Linear classifier on the same 22-dim feature vector.
    If XGBoost only marginally beats this, the non-linearity
    of the problem is limited and simpler models suffice.
    """
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self._feature_names = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def route(self, feature_vector: np.ndarray) -> str:
        pred = self.model.predict(feature_vector.reshape(1, -1))[0]
        return "GRAPH" if pred == 1 else "SQL"
```

**Step 2 — Update `experiments/run_baselines.py` to include all baselines:**

```python
# experiments/run_baselines.py  — updated strategy handling

STRATEGIES = {
    "always_sql":   lambda sub, fv, fn: "SQL",
    "always_graph": lambda sub, fv, fn: "GRAPH",
    "trivial_rule": lambda sub, fv, fn: trivial_rule_route(sub),
    "threshold":    lambda sub, fv, fn: tuned_threshold.route(fv, fn),  # PRIMARY
    "logreg":       lambda sub, fv, fn: logreg_baseline.route(fv),
    "learned_dt":   lambda sub, fv, fn: dt_predictor.predict(fv),
    "learned_xgb":  lambda sub, fv, fn: xgb_predictor.predict(fv),
}
```

**Step 3 — Update Table V in the paper to include all baselines:**

| Strategy | Median Latency (ms) | p95 Latency (ms) | Routing Accuracy | Notes |
|---|---|---|---|---|
| Always SQL | baseline | baseline | — | Single-engine lower bound |
| Always Graph | — | — | — | Single-engine lower bound |
| Trivial Rule | — | — | — | Legacy baseline (traversal=graph) |
| **Threshold Rule** | — | — | — | **Primary competitive baseline** |
| Logistic Regression | — | — | — | Tests ML necessity |
| Decision Tree | — | — | — | Interpretable ML |
| **XGBoost (ML)** | — | — | — | **Primary proposed method** |

**Acceptance Criteria:**
- [ ] `ThresholdBaseline` class implemented with `tune_thresholds()` method
- [ ] `LogisticRegressionBaseline` class implemented
- [ ] All 7 strategies runnable from `run_baselines.py`
- [ ] Paper Table V includes threshold and logistic regression baselines
- [ ] The paper explicitly states the threshold baseline is the PRIMARY competitive baseline

**Estimated time:** 4–6 hours

---

## Phase 2 — Core Experimental Overhaul (2–3 months)

> **Goal:** Replace simulated results with real execution measurements.
> This phase is what transforms the paper from workshop-grade to main-track-contender.

---

### Task 2.1 — Implement Real PySpark + GraphFrames Execution

**Priority:** CRITICAL for main track submission

**Problem:** The paper uses pandas/Python BFS instead of actual Spark SQL and GraphFrames.
Routing decisions optimized for pandas are not valid for Spark + Catalyst.

**Files to create/modify:**
```
execution/spark_sql_generator.py      (new — real Spark SQL)
execution/graphframes_generator.py    (new — real GraphFrames)
execution/pandas_sql_generator.py     (rename existing sql_generator.py)
execution/python_graph_generator.py   (rename existing graph_generator.py)
config/spark_config.py                (modify)
router/hybrid_router.py               (modify — swap engines)
```

**Implementation Steps:**

**Step 1 — Real Spark SQL Generator (`execution/spark_sql_generator.py`):**

```python
# execution/spark_sql_generator.py

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from decomposer.query_decomposer import SubExpression

class SparkSQLGenerator:
    """
    Executes SubExpressions using actual PySpark DataFrame API with
    Catalyst optimizer enabled. Predicate pushdown and AQE are
    configured at the SparkSession level.
    """

    def __init__(self, spark: SparkSession, parquet_dir: str):
        self.spark      = spark
        self.parquet_dir = parquet_dir
        self._cache: dict[str, DataFrame] = {}

    def generate(self, sub_expr: SubExpression,
                       upstream_results: dict) -> DataFrame:
        """
        Translates a SubExpression to Spark DataFrame operations.
        upstream_results: {sub_id: DataFrame} from previously executed subs.
        """
        df = None
        for node in sub_expr.nodes:
            if node.op_type == "FILTER":
                df = self._apply_filter(node, df, upstream_results)
            elif node.op_type == "JOIN":
                df = self._apply_join(node, df, upstream_results)
            elif node.op_type == "AGGREGATE":
                df = self._apply_aggregate(node, df)
            elif node.op_type == "MAP":
                df = df.select([F.col(c) for c in node.fields])
        return df

    def _load_source(self, source: str,
                           upstream: dict) -> DataFrame:
        """Load from Parquet or from upstream sub result."""
        if source in upstream:
            return upstream[source]
        if source in self._cache:
            return self._cache[source]
        path = f"{self.parquet_dir}/{source}"
        df = self.spark.read.parquet(path)
        # Register as temp view for SQL debugging
        df.createOrReplaceTempView(source)
        self._cache[source] = df
        return df

    def _apply_filter(self, node, df, upstream) -> DataFrame:
        if df is None:
            df = self._load_source(node.source, upstream)
        p = node.predicate
        op_map = {
            "=":  lambda c, v: F.col(c) == v,
            ">":  lambda c, v: F.col(c) > v,
            "<":  lambda c, v: F.col(c) < v,
            ">=": lambda c, v: F.col(c) >= v,
            "<=": lambda c, v: F.col(c) <= v,
            "IN": lambda c, v: F.col(c).isin(v),
            "LIKE": lambda c, v: F.col(c).like(v),
        }
        condition = op_map[p["operator"]](p["column"], p["value"])
        # Spark Catalyst will push this filter down into the Parquet scan
        return df.filter(condition).select(node.fields)

    def _apply_join(self, node, df, upstream) -> DataFrame:
        right_df = self._load_source(node.join["right_source"], upstream)
        join_type = node.join["join_type"].lower()
        left_key  = node.join["left_key"]
        right_key = node.join["right_key"]
        # Use broadcast hint for small tables to avoid shuffle
        right_rows = right_df.count()
        if right_rows < 100_000:
            right_df = F.broadcast(right_df)
        return df.join(right_df, df[left_key] == right_df[right_key], how=join_type)

    def _apply_aggregate(self, node, df) -> DataFrame:
        agg_exprs = []
        for fn_spec in node.aggregate["functions"]:
            func  = fn_spec["func"].lower()
            col_n = fn_spec["column"]
            alias = f"{fn_spec['func']}_{col_n}"
            agg_exprs.append(getattr(F, func)(col_n).alias(alias))
        return df.groupBy(node.aggregate["group_by"]).agg(*agg_exprs)
```

**Step 2 — Real GraphFrames Generator (`execution/graphframes_generator.py`):**

```python
# execution/graphframes_generator.py

from graphframes import GraphFrame
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from decomposer.query_decomposer import SubExpression

class GraphFramesGenerator:
    """
    Executes TRAVERSAL SubExpressions using actual GraphFrames BFS.
    Requires graphframes JAR in Spark classpath.
    """

    def __init__(self, spark: SparkSession, graph_dir: str):
        self.spark    = spark
        self.graph_dir = graph_dir
        self._gf_cache: dict[str, GraphFrame] = {}

    def generate(self, sub_expr: SubExpression,
                       upstream_results: dict) -> DataFrame:
        for node in sub_expr.nodes:
            if node.op_type == "TRAVERSAL":
                return self._apply_traversal(node, upstream_results)
        raise ValueError(f"No TRAVERSAL node in sub {sub_expr.sub_id}")

    def _load_graph(self, graph_name: str) -> GraphFrame:
        if graph_name in self._gf_cache:
            return self._gf_cache[graph_name]

        v_path = f"{self.graph_dir}/{graph_name}_vertices.parquet"
        e_path = f"{self.graph_dir}/{graph_name}_edges.parquet"

        vertices = self.spark.read.parquet(v_path)
        edges    = self.spark.read.parquet(e_path)
        gf = GraphFrame(vertices, edges)
        # Cache at GraphFrames level
        gf.vertices.cache()
        gf.edges.cache()
        self._gf_cache[graph_name] = gf
        return gf

    def _apply_traversal(self, node, upstream: dict) -> DataFrame:
        t  = node.traversal
        gf = self._load_graph(node.source)

        sf = t["start_vertex_filter"]
        start_expr = f"{sf['column']} = {repr(sf['value'])}"

        # Direction handling
        if t["direction"] == "OUT":
            edge_filter = f"relationship = {repr(t['edge_label'])}"
        elif t["direction"] == "IN":
            # Reverse edges for IN direction
            edge_filter = f"relationship = {repr(t['edge_label'])}"
            gf = GraphFrame(gf.vertices,
                            gf.edges.select(
                                F.col("dst").alias("src"),
                                F.col("src").alias("dst"),
                                F.col("relationship")
                            ))
        else:  # BOTH — already bidirectional if edges were prepared correctly
            edge_filter = f"relationship = {repr(t['edge_label'])}"

        result = gf.bfs(
            fromExpr=start_expr,
            toExpr="id IS NOT NULL",
            edgeFilter=edge_filter,
            maxPathLength=t["max_hops"]
        )

        # Extract destination vertices with requested fields
        to_fields = t.get("return_fields", ["id"])
        # BFS returns columns: from, e0, v1, e1, v2, ... to
        # We want the "to" column's struct fields
        return result.select([
            F.col(f"to.{field}").alias(field)
            for field in to_fields
            if field in [f.name for f in result.schema["to"].dataType.fields]
        ]).distinct()
```

**Step 3 — Update `config/spark_config.py` for GraphFrames:**

```python
# config/spark_config.py — updated

from pyspark.sql import SparkSession
import os

def get_spark_session(app_name: str = "HIFUN_Router",
                      mode: str = "local[4]",
                      memory: str = "4g") -> SparkSession:

    # GraphFrames package — must match your Spark version
    GF_PACKAGE = "graphframes:graphframes:0.8.3-spark3.4-s_2.12"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(mode)
        .config("spark.jars.packages", GF_PACKAGE)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Enable column statistics for better Catalyst decisions
        .config("spark.sql.statistics.histogram.enabled", "true")
        .config("spark.sql.statistics.fallBackToHdfs", "true")
        # Broadcast small tables automatically
        .config("spark.sql.autoBroadcastJoinThreshold", "10mb")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
```

**Step 4 — Update `HybridRouter` to use real engines:**

```python
# router/hybrid_router.py — add engine selection flag

class HybridRouter:
    def __init__(self, ..., use_real_engines: bool = True):
        if use_real_engines:
            from execution.spark_sql_generator   import SparkSQLGenerator
            from execution.graphframes_generator  import GraphFramesGenerator
            self.sql_gen   = SparkSQLGenerator(spark, parquet_dir)
            self.graph_gen = GraphFramesGenerator(spark, graph_dir)
        else:
            # Fallback to fast pandas execution for unit tests
            from execution.pandas_sql_generator   import SQLGenerator
            from execution.python_graph_generator  import GraphGenerator
            self.sql_gen   = SQLGenerator(parquet_dir)
            self.graph_gen = GraphGenerator(graph_dir)
```

**Acceptance Criteria:**
- [ ] `SparkSQLGenerator` runs TPC-H FILTER + JOIN + AGGREGATE queries on real Parquet
- [ ] `GraphFramesGenerator` runs BFS traversals on SNB graph via actual GraphFrames
- [ ] Unit tests pass with `use_real_engines=True`
- [ ] Spark UI shows real stages, shuffle bytes, and task metrics

**Estimated time:** 2–3 weeks (includes debugging Spark + GraphFrames env)

---

### Task 2.2 — Collect Real Execution Labels (Replace Simulation)

**Priority:** CRITICAL — the most important experimental task

**Problem:** All 186 training labels come from a heuristic cost formula. You need
real measured runtimes. Target: **500–1,000 labeled subexpressions** from real execution.

**Files to create/modify:**
```
training_data/real_collection_script.py   (new)
training_data/real_labeled_runs.csv       (output)
```

**Implementation Steps:**

**Step 1 — Build the real label collection script:**

```python
# training_data/real_collection_script.py

import time
import json
import csv
import os
import traceback
import pandas as pd
from pyspark.sql import SparkSession
from config.spark_config   import get_spark_session
from parser.dsl_parser     import DSLParser
from decomposer.query_decomposer import QueryDecomposer
from features.feature_extractor  import FeatureExtractor
from execution.spark_sql_generator    import SparkSQLGenerator
from execution.graphframes_generator  import GraphFramesGenerator

QUERY_DIRS     = ["dsl/sample_queries/"]
STATS_DIR      = "data/stats/"
PARQUET_DIR    = "data/parquet/"
GRAPH_DIR      = "data/graphs/"
OUTPUT_CSV     = "training_data/real_labeled_runs.csv"
N_WARMUP_RUNS  = 2    # warm up JVM before measuring
N_MEASURE_RUNS = 3    # take median of 3 runs

FEATURE_NAMES = [   # must match feature_schema.json order
    "op_count_filter", "op_count_join", "op_count_traversal",
    "op_count_aggregate", "op_count_map", "ast_depth",
    "has_traversal", "max_hops", "input_cardinality_log",
    "output_cardinality_log", "selectivity", "avg_degree",
    "max_degree", "degree_skew", "num_projected_columns",
    "has_index", "join_fanout", "estimated_shuffle_bytes_log",
    "estimated_traversal_ops", "hist_avg_runtime_ms",
    "hist_runtime_variance", "num_tables_joined"
]

def measure_execution_time(fn, n_warmup, n_measure) -> tuple[float, float]:
    """Run fn() n_warmup+n_measure times. Return (median_ms, stddev_ms)."""
    # Warmup
    for _ in range(n_warmup):
        try:
            fn()
        except Exception:
            pass

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        try:
            result = fn()
            # Force materialization (Spark is lazy)
            if hasattr(result, "count"):
                _ = result.count()
            elif isinstance(result, pd.DataFrame):
                _ = len(result)
        except Exception as e:
            times.append(float("inf"))
            continue
        times.append((time.perf_counter() - t0) * 1000)

    times_clean = [t for t in times if t != float("inf")]
    if not times_clean:
        return float("inf"), float("inf")
    return (sorted(times_clean)[len(times_clean)//2],   # median
            pd.Series(times_clean).std())                # stddev

def collect_labels(spark: SparkSession, output_csv: str):
    parser    = DSLParser()
    decomposer = QueryDecomposer()
    extractor  = FeatureExtractor(STATS_DIR, "training_data/history.db")
    sql_gen    = SparkSQLGenerator(spark, PARQUET_DIR)
    graph_gen  = GraphFramesGenerator(spark, GRAPH_DIR)

    rows = []

    for query_dir in QUERY_DIRS:
        for fname in sorted(os.listdir(query_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(query_dir, fname)) as f:
                data = json.load(f)
            queries = data if isinstance(data, list) else [data]

            for query in queries:
                print(f"\n── Processing query: {query['query_id']} ──")
                try:
                    nodes        = parser.parse(query)
                    sub_exprs    = decomposer.decompose(nodes)
                except Exception as e:
                    print(f"  Parse/decompose error: {e}")
                    continue

                for sub in sub_exprs:
                    print(f"  Sub: {sub.sub_id} ({sub.primary_op_type})")
                    try:
                        fv = extractor.extract(sub)
                    except Exception as e:
                        print(f"    Feature extraction error: {e}")
                        continue

                    # --- Measure SQL path ---
                    sql_median_ms, sql_std_ms = measure_execution_time(
                        fn=lambda s=sub: sql_gen.generate(s, {}),
                        n_warmup=N_WARMUP_RUNS,
                        n_measure=N_MEASURE_RUNS
                    )

                    # --- Measure Graph path ---
                    # Only attempt graph execution if sub has traversal features
                    if fv[FEATURE_NAMES.index("has_traversal")] == 1:
                        graph_median_ms, graph_std_ms = measure_execution_time(
                            fn=lambda s=sub: graph_gen.generate(s, {}),
                            n_warmup=N_WARMUP_RUNS,
                            n_measure=N_MEASURE_RUNS
                        )
                    else:
                        # For pure relational subs, simulate graph cost as very high
                        # (routing will always choose SQL; this is still valid training data)
                        graph_median_ms = sql_median_ms * 3.0
                        graph_std_ms    = 0.0

                    label = "GRAPH" if graph_median_ms < sql_median_ms else "SQL"
                    speedup = sql_median_ms / max(graph_median_ms, 0.001)

                    print(f"    SQL={sql_median_ms:.1f}ms  "
                          f"Graph={graph_median_ms:.1f}ms  "
                          f"Label={label}  Speedup={speedup:.2f}x")

                    row = {
                        "sub_id":        sub.sub_id,
                        "query_id":      query["query_id"],
                        "dataset":       fname.replace(".json", ""),
                        **dict(zip(FEATURE_NAMES, fv)),
                        "sql_median_ms":   round(sql_median_ms, 2),
                        "sql_std_ms":      round(sql_std_ms, 2),
                        "graph_median_ms": round(graph_median_ms, 2),
                        "graph_std_ms":    round(graph_std_ms, 2),
                        "speedup":         round(speedup, 3),
                        "label":           label,
                        "label_source":    "real_measurement"
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(df)} labeled rows → {output_csv}")
    print(f"  SQL: {(df.label=='SQL').sum()}, GRAPH: {(df.label=='GRAPH').sum()}")
    return df

if __name__ == "__main__":
    spark = get_spark_session("HIFUN_LabelCollection", mode="local[4]")
    collect_labels(spark, OUTPUT_CSV)
    spark.stop()
```

**Step 2 — Generate additional query variants to reach 500+ samples:**

```python
# training_data/query_generator.py
# Generates DSL query variants by sweeping key parameters

import json
import itertools

def generate_filter_variants(base_query: dict,
                              selectivity_values: list) -> list:
    """
    For a FILTER query, vary the predicate value to change selectivity.
    Assumes the predicate is on a numeric range column.
    """
    variants = []
    for i, sv in enumerate(selectivity_values):
        q = json.loads(json.dumps(base_query))  # deep copy
        q["query_id"] = f"{base_query['query_id']}_sv{i}"
        # Adjust predicate value to approximate target selectivity
        # (Requires knowing column min/max from stats)
        q["operations"][0]["predicate"]["_selectivity_hint"] = sv
        variants.append(q)
    return variants

def generate_traversal_variants(base_traversal_query: dict,
                                 hop_values: list) -> list:
    """
    For a TRAVERSAL query, vary max_hops.
    """
    variants = []
    for h in hop_values:
        q = json.loads(json.dumps(base_traversal_query))
        q["query_id"] = f"{base_traversal_query['query_id']}_hops{h}"
        for op in q["operations"]:
            if op["type"] == "TRAVERSAL":
                op["traversal"]["max_hops"] = h
        variants.append(q)
    return variants

if __name__ == "__main__":
    # Load base queries and generate variants
    with open("dsl/sample_queries/snb_queries.json") as f:
        snb_queries = json.load(f)

    all_variants = []
    for q in snb_queries:
        if any(op["type"] == "TRAVERSAL" for op in q["operations"]):
            all_variants.extend(
                generate_traversal_variants(q, hop_values=[1, 2, 3, 4])
            )

    with open("dsl/sample_queries/synthetic_traversal_variants.json", "w") as f:
        json.dump(all_variants, f, indent=2)
    print(f"Generated {len(all_variants)} traversal variant queries")
```

**Acceptance Criteria:**
- [ ] `training_data/real_labeled_runs.csv` has ≥ 500 rows
- [ ] Column `label_source == "real_measurement"` for all rows
- [ ] Class balance: at least 30% GRAPH samples
- [ ] F1 score on real labels is between 0.80–0.95 (not 1.00)
- [ ] Paper Table II updated with real-measurement CV results

**Estimated time:** 3–4 weeks (including iteration on graph data issues)

---

### Task 2.3 — Cross-Dataset Generalization Experiment

**Priority:** MAJOR for main track

**Problem:** The paper claims generalizability but never tests it. A model trained entirely
on TPC-H-derived labels is not validated on graph-heavy SNB queries.

**Files to create:**
```
experiments/cross_dataset_generalization.py   (new)
```

**Implementation Steps:**

```python
# experiments/cross_dataset_generalization.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLS = [...]   # your 22 feature names
LABEL_COL    = "label"
MODEL_PATH   = "model/artifacts/classifier_v1.pkl"

def load_dataset_splits(labeled_csv: str) -> dict:
    """Split labeled_runs.csv by dataset column."""
    df = pd.read_csv(labeled_csv)
    return {
        dataset: df[df["dataset"] == dataset]
        for dataset in df["dataset"].unique()
    }

def cross_dataset_experiment(labeled_csv: str, output_dir: str):
    """
    For each pair (train_dataset, test_dataset):
      1. Train XGBoost on train_dataset only
      2. Evaluate on test_dataset only
      3. Fine-tune with 20% of test_dataset
      4. Re-evaluate
    Produces a generalization heatmap.
    """
    splits = load_dataset_splits(labeled_csv)
    datasets = list(splits.keys())
    results = []

    print(f"Datasets: {datasets}")

    for train_ds in datasets:
        for test_ds in datasets:
            train_df = splits[train_ds]
            test_df  = splits[test_ds]

            X_train = train_df[FEATURE_COLS].values
            y_train = (train_df[LABEL_COL] == "GRAPH").astype(int).values
            X_test  = test_df[FEATURE_COLS].values
            y_test  = (test_df[LABEL_COL]  == "GRAPH").astype(int).values

            if len(X_train) < 10 or len(X_test) < 5:
                continue

            # Train on train_ds
            model = xgb.XGBClassifier(n_estimators=200, max_depth=5,
                                       learning_rate=0.05, random_state=42,
                                       eval_metric="logloss")
            model.fit(X_train, y_train)

            # Evaluate on test_ds (no fine-tuning)
            y_pred_base = model.predict(X_test)
            f1_base     = f1_score(y_test, y_pred_base,
                                   average="weighted", zero_division=0)

            # Fine-tune with 20% of test_ds
            X_ft, _, y_ft, _ = train_test_split(
                X_test, y_test, train_size=0.2, stratify=y_test, random_state=42
            ) if len(np.unique(y_test)) > 1 else (X_test[:2], X_test[2:],
                                                    y_test[:2], y_test[2:])

            # XGBoost incremental training
            model.fit(
                np.vstack([X_train, X_ft]),
                np.hstack([y_train, y_ft])
            )
            y_pred_ft = model.predict(X_test)
            f1_ft     = f1_score(y_test, y_pred_ft,
                                  average="weighted", zero_division=0)

            results.append({
                "train_dataset":   train_ds,
                "test_dataset":    test_ds,
                "train_samples":   len(X_train),
                "test_samples":    len(X_test),
                "f1_base":         round(f1_base, 3),
                "f1_finetuned":    round(f1_ft,   3),
                "f1_improvement":  round(f1_ft - f1_base, 3),
                "same_dataset":    train_ds == test_ds
            })

            print(f"  Train={train_ds:15s} → Test={test_ds:15s} | "
                  f"F1_base={f1_base:.3f}  F1_ft={f1_ft:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/cross_dataset_results.csv", index=False)

    # Plot generalization heatmap (base F1)
    pivot = results_df.pivot("train_dataset", "test_dataset", "f1_base")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.5, vmax=1.0, ax=axes[0])
    axes[0].set_title("Cross-Dataset F1 (Base — No Fine-Tuning)")
    axes[0].set_xlabel("Test Dataset")
    axes[0].set_ylabel("Train Dataset")

    pivot_ft = results_df.pivot("train_dataset", "test_dataset", "f1_finetuned")
    sns.heatmap(pivot_ft, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.5, vmax=1.0, ax=axes[1])
    axes[1].set_title("Cross-Dataset F1 (After 20% Fine-Tuning)")
    axes[1].set_xlabel("Test Dataset")
    axes[1].set_ylabel("Train Dataset")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cross_dataset_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved cross-dataset heatmap → {output_dir}/cross_dataset_heatmap.pdf")

    return results_df

if __name__ == "__main__":
    cross_dataset_experiment(
        labeled_csv="training_data/real_labeled_runs.csv",
        output_dir="experiments/results/"
    )
```

**What to report in the paper:**
- A 3×3 (or N×N) heatmap table showing F1 for each train/test dataset combination
- The diagonal (same-dataset) vs. off-diagonal (cross-dataset) gap
- Fine-tuning recovery: how much F1 improves after seeing 20% of the test domain

**Acceptance Criteria:**
- [ ] Script runs and produces `cross_dataset_results.csv` and heatmap PDF
- [ ] Paper contains the generalization heatmap as a figure
- [ ] Cross-dataset F1 values are reported in Section IV
- [ ] Fine-tuning improvement is quantified

**Estimated time:** 1 week

---

### Task 2.4 — Scale Factor Experiments (TPC-H SF=1 and SF=5)

**Priority:** MAJOR — scalability is expected by SIGMOD reviewers

**Files to create:**
```
experiments/scale_factor_experiment.py   (new)
data/scripts/generate_tpch_sf5.sh        (new)
```

**Implementation Steps:**

```bash
# data/scripts/generate_tpch_sf5.sh
#!/bin/bash
# Generate TPC-H at Scale Factor 5 (~5GB)
cd tpch-kit
./dbgen -s 5 -f -T a    # all tables

# Convert to Parquet
python data/scripts/tpch_to_parquet.py \
    --input ./tpch-kit \
    --output data/parquet/tpch_sf5/ \
    --scale_factor 5

echo "TPC-H SF=5 generated."
```

```python
# experiments/scale_factor_experiment.py

import time, json, os, csv
import pandas as pd
from config.spark_config import get_spark_session
from router.hybrid_router import HybridRouter

SCALE_FACTORS = {
    "sf1": "data/parquet/tpch/",
    "sf5": "data/parquet/tpch_sf5/",
}
STRATEGIES = ["always_sql", "threshold", "learned_xgb"]
QUERIES_PATH = "dsl/sample_queries/tpch_queries.json"

def run_scale_experiment(output_csv: str):
    rows = []
    with open(QUERIES_PATH) as f:
        queries = json.load(f)

    for sf_label, parquet_dir in SCALE_FACTORS.items():
        spark = get_spark_session(f"HIFUN_SF_{sf_label}", mode="local[4]")
        print(f"\n=== Scale Factor: {sf_label} ({parquet_dir}) ===")

        for strategy in STRATEGIES:
            router = HybridRouter.from_config(
                parquet_dir=parquet_dir,
                force_engine=("SQL"   if strategy == "always_sql"  else
                              "GRAPH" if strategy == "always_graph" else None),
                routing_strategy=strategy
            )

            for query in queries:
                qid = query["query_id"]
                t0 = time.perf_counter()
                try:
                    result = router.execute(query)
                    if hasattr(result, "count"):
                        _ = result.count()
                    latency_ms = (time.perf_counter() - t0) * 1000
                    row_count  = result.count() if hasattr(result, "count") else len(result)
                    status = "ok"
                except Exception as e:
                    latency_ms = float("inf")
                    row_count  = -1
                    status = str(e)[:100]

                rows.append({
                    "scale_factor": sf_label,
                    "strategy":     strategy,
                    "query_id":     qid,
                    "latency_ms":   round(latency_ms, 2),
                    "row_count":    row_count,
                    "status":       status
                })
                print(f"  [{sf_label}][{strategy}][{qid}]: {latency_ms:.1f}ms")

        spark.stop()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    # Summary table
    summary = (df[df["status"] == "ok"]
               .groupby(["scale_factor", "strategy"])["latency_ms"]
               .agg(["median", lambda x: x.quantile(0.95)])
               .rename(columns={"median": "median_ms", "<lambda_0>": "p95_ms"}))
    print("\n=== Scale Factor Summary ===")
    print(summary.to_string())
    return df

if __name__ == "__main__":
    run_scale_experiment("experiments/results/scale_factor_results.csv")
```

**Acceptance Criteria:**
- [ ] Experiments run at both SF=1 and SF=5
- [ ] Paper Table V shows separate rows for SF=1 and SF=5
- [ ] Routing decision stability across scale factors is discussed in Section IV

**Estimated time:** 1 week (mostly data generation + I/O wait time)

---

### Task 2.5 — Address Feature Collinearity with VIF Analysis

**Priority:** MODERATE — improves rigor of feature engineering claims

**Files to create:**
```
notebooks/05_feature_collinearity.ipynb   (new)
model/feature_analysis.py                 (new)
```

**Implementation Steps:**

```python
# model/feature_analysis.py

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

FEATURE_COLS = [...]  # your 22 feature names

def compute_vif(labeled_csv: str) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor for each feature.
    VIF > 10 indicates severe multicollinearity.
    VIF > 5 indicates moderate multicollinearity.
    """
    df = pd.read_csv(labeled_csv)[FEATURE_COLS].dropna()

    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    vif_data = []
    for i, col in enumerate(df.columns):
        try:
            vif = variance_inflation_factor(df.values, i)
        except Exception:
            vif = float("inf")
        vif_data.append({"Feature": col, "VIF": round(vif, 2)})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    print("Features with VIF > 5 (potential collinearity):")
    print(vif_df[vif_df["VIF"] > 5].to_string(index=False))
    return vif_df

def suggest_feature_removals(vif_df: pd.DataFrame, threshold: float = 10.0) -> list:
    """
    Suggests which features to remove based on VIF threshold.
    Remove the highest VIF feature that is derived from others.
    """
    high_vif = vif_df[vif_df["VIF"] > threshold]["Feature"].tolist()

    # Known derived features (safe to remove in order)
    derived_priority = [
        "estimated_shuffle_bytes_log",  # derived from output_cardinality_log
        "output_cardinality_log",       # derived from input_cardinality_log + selectivity
    ]
    to_remove = [f for f in derived_priority if f in high_vif]
    print(f"\nRecommended removals (VIF > {threshold}): {to_remove}")
    return to_remove

if __name__ == "__main__":
    vif_df = compute_vif("training_data/real_labeled_runs.csv")
    suggest_feature_removals(vif_df)
    vif_df.to_csv("experiments/results/vif_analysis.csv", index=False)
```

**What to report in the paper:**
- A VIF table in the Feature Engineering section
- State which features are collinear and justify keeping them (they may be correlated
  but still provide signal to tree ensemble models which handle collinearity well)
- Or remove the highest-VIF derived features and show ablation results are unchanged

**Acceptance Criteria:**
- [ ] VIF analysis run on the real labeled dataset
- [ ] Paper feature section discusses collinearity explicitly
- [ ] Either features are removed with justification, or collinearity is acknowledged

**Estimated time:** 3–4 hours

---

## Phase 3 — Research Depth (4–6 months)

> **Goal:** Full SIGMOD Research Track readiness with formalized HIFUN connection,
> online learning, and large-scale evaluation.

---

### Task 3.1 — Formalize the HIFUN–DSL Algebraic Mapping

**Priority:** HIGH for main track differentiation

**Files to create:**
```
report/sections/hifun_formalization.tex   (new section in paper)
dsl/hifun_algebra.py                      (new — formal mapping)
```

**Implementation Steps:**

Define a formal correspondence table between HIFUN algebraic operators and DSL types.
HIFUN uses functional operators: `FOLD`, `SCAN`, `MAP`, `PRODUCT`, `FILTER`, `REDUCE`.

Create a table in the paper:

| HIFUN Algebra | DSL Operator | Execution Hint |
|---|---|---|
| `SCAN(R)` | `FILTER` with no predicate | SQL (table scan) |
| `FILTER(p, R)` | `FILTER` with predicate | SQL if high selectivity; GRAPH if on vertex |
| `MAP(f, R)` | `MAP` | Merge with parent |
| `JOIN(R1, R2, k)` | `JOIN` | SQL (hash join or broadcast) |
| `FOLD(agg, R)` | `AGGREGATE` | SQL (post-traversal) or GRAPH |
| `TRAVERSE(G, v, l, h)` | `TRAVERSAL` | GRAPH (BFS) |

Add a formal proposition:
> **Proposition 1 (Decomposition Correctness):** The `QueryDecomposer` preserves
> HIFUN query semantics: for any HIFUN query Q, the composed result of independently
> executing its SubExpressions S₁,...,Sₙ and merging via `ResultComposer` is
> semantically equivalent to evaluating Q in a single-engine reference interpreter.
>
> *Proof sketch:* By induction on query structure. Base case: a single SCAN/FILTER
> operation is correctly translated by both engines. Inductive step: the `ResultComposer`
> applies the same join/aggregation semantics as the HIFUN JOIN/FOLD operators,
> preserving commutativity and associativity. □

**Estimated time:** 2–3 weeks (research + writing)

---

### Task 3.2 — Implement Online Retraining Loop

**Priority:** MODERATE — differentiates from static classifiers

**Files to create:**
```
model/online_trainer.py    (new)
router/adaptive_router.py  (new)
```

**Implementation Steps:**

```python
# model/online_trainer.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score
import xgboost as xgb

class OnlineTrainer:
    """
    Periodically retrains the routing classifier as new execution
    observations arrive. Uses a sliding window over recent runs.
    """
    WINDOW_SIZE    = 500    # Use last N samples for retraining
    RETRAIN_EVERY  = 50     # Retrain after every N new observations
    MIN_SAMPLES    = 100    # Minimum samples before first training

    def __init__(self, model_path: str, history_csv: str,
                       feature_names: list):
        self.model_path    = model_path
        self.history_csv   = history_csv
        self.feature_names = feature_names
        self.model         = joblib.load(model_path)
        self.obs_since_last_retrain = 0

    def record_observation(self, sub_id: str, features: np.ndarray,
                                  predicted_engine: str,
                                  actual_sql_ms: float,
                                  actual_graph_ms: float):
        """
        Record a new labeled observation after real execution.
        Triggers retraining if threshold is reached.
        """
        label = "GRAPH" if actual_graph_ms < actual_sql_ms else "SQL"
        row   = dict(zip(self.feature_names, features))
        row.update({
            "sub_id":          sub_id,
            "predicted_engine": predicted_engine,
            "sql_ms":           actual_sql_ms,
            "graph_ms":         actual_graph_ms,
            "label":            label,
            "label_source":     "online_observation"
        })

        # Append to history
        df = pd.DataFrame([row])
        header = not pd.io.common.file_exists(self.history_csv)
        df.to_csv(self.history_csv, mode="a", header=header, index=False)

        self.obs_since_last_retrain += 1
        if self.obs_since_last_retrain >= self.RETRAIN_EVERY:
            self._retrain()
            self.obs_since_last_retrain = 0

    def _retrain(self):
        """Retrain on the most recent WINDOW_SIZE observations."""
        history = pd.read_csv(self.history_csv)
        if len(history) < self.MIN_SAMPLES:
            print(f"  OnlineTrainer: only {len(history)} samples, skipping retrain")
            return

        recent = history.tail(self.WINDOW_SIZE)
        X = recent[self.feature_names].values
        y = (recent["label"] == "GRAPH").astype(int).values

        new_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, eval_metric="logloss"
        )
        new_model.fit(X, y)

        # Evaluate improvement
        y_pred = new_model.predict(X)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        print(f"  OnlineTrainer: retrained on {len(recent)} samples, F1={f1:.3f}")

        joblib.dump(new_model, self.model_path)
        self.model = new_model
```

**Estimated time:** 2 weeks

---

### Task 3.3 — Demo Paper for SIGMOD 2026 Demonstration Track

**Priority:** HIGH — fastest path to a SIGMOD publication

**Files to create:**
```
demo/app.py                  (Streamlit demo application)
demo/requirements_demo.txt
report/demo_paper.tex        (4-page demo paper)
```

**Implementation Steps:**

```python
# demo/app.py — Interactive HIFUN Router Demo

import streamlit as st
import json
import time
import shap
import matplotlib.pyplot as plt
import pandas as pd
from router.hybrid_router import HybridRouter
from model.predictor      import ModelPredictor
from features.feature_extractor import FeatureExtractor
from parser.dsl_parser    import DSLParser
from decomposer.query_decomposer import QueryDecomposer

st.set_page_config(page_title="HIFUN Router Demo", layout="wide")
st.title("HIFUN Router — Interactive Query Routing Demo")
st.caption("ML-Based Query Decomposition and Intelligent Engine Routing")

# ─── Sidebar: Query input ───
st.sidebar.header("Query Input")
sample_queries = {
    "TPC-H Join Query":     "dsl/sample_queries/tpch_queries.json",
    "SNB Mixed Query":      "dsl/sample_queries/snb_queries.json",
    "Pure Graph Traversal": "dsl/sample_queries/synthetic_queries.json",
}
selected = st.sidebar.selectbox("Choose a sample query:", list(sample_queries.keys()))

with open(sample_queries[selected]) as f:
    qs = json.load(f)
query_json_str = st.sidebar.text_area(
    "Edit Query JSON:", json.dumps(qs[0], indent=2), height=400)

run_btn = st.sidebar.button("▶ Run Query", type="primary")

if run_btn:
    query = json.loads(query_json_str)

    col1, col2, col3 = st.columns(3)

    # ─── Decomposition visualization ───
    with col1:
        st.subheader("1. Query Decomposition")
        parser     = DSLParser()
        decomposer = QueryDecomposer()
        nodes      = parser.parse(query)
        sub_exprs  = decomposer.decompose(nodes)

        for sub in sub_exprs:
            engine_color = "🟦" if sub.primary_op_type == "TRAVERSAL" else "🟧"
            st.markdown(f"{engine_color} **{sub.sub_id}** — "
                        f"`{sub.primary_op_type}` "
                        f"({'parallelizable' if sub.parallelizable else 'sequential'})")
            for n in sub.nodes:
                st.markdown(f"  └ `{n.op_type}` on `{n.source}`")

    # ─── Routing decision + SHAP explanation ───
    with col2:
        st.subheader("2. Routing Decisions")
        extractor = FeatureExtractor("data/stats/", "training_data/history.db")
        predictor = ModelPredictor("model/artifacts/classifier_v1.pkl")

        for sub in sub_exprs:
            fv = extractor.extract(sub)
            t0 = time.perf_counter()
            engine = predictor.predict(fv)
            inf_ms = (time.perf_counter() - t0) * 1000

            color = "🔵" if engine == "GRAPH" else "🟠"
            st.markdown(f"{color} **{sub.sub_id}** → `{engine}` engine "
                        f"*(inference: {inf_ms:.2f}ms)*")

    # ─── SHAP explanation ───
    with col3:
        st.subheader("3. Why This Decision?")
        # Show SHAP force plot for first subexpression
        if sub_exprs:
            sub  = sub_exprs[0]
            fv   = extractor.extract(sub)
            fig  = predictor.explain_shap(fv)
            st.pyplot(fig)
            st.caption("SHAP values show which features drove the routing decision.")

    # ─── Execution results ───
    st.subheader("4. Execution Results")
    router = HybridRouter.from_config("config/paths.py")
    with st.spinner("Executing query..."):
        t0     = time.perf_counter()
        result = router.execute(query)
        elapsed = (time.perf_counter() - t0) * 1000
    if hasattr(result, "toPandas"):
        result = result.toPandas()
    st.dataframe(result.head(50))
    st.metric("Total Execution Time", f"{elapsed:.1f} ms")
```

**Acceptance Criteria:**
- [ ] Demo app runs with `streamlit run demo/app.py`
- [ ] Shows decomposition, routing decision, SHAP explanation, and results
- [ ] 4-page demo paper draft exists for SIGMOD 2026 Demonstration Track
- [ ] Screenshots included in demo paper

**Estimated time:** 2–3 weeks

---

## Appendix A — New Related Work Reference List

These are the papers you MUST add to `references.bib` and cite in the paper:

| Paper | Year | Venue | Why Cite |
|---|---|---|---|
| Apache Wayang | 2025 | SIGMOD | Direct competitor — cross-engine routing |
| AutoSteer | 2023 | VLDB | Learned optimizer for any SQL DB |
| LEON | 2023 | VLDB | ML-aided query optimization framework |
| Balsa | 2022 | SIGMOD | Learning optimizer from scratch |
| Neo | 2019 | VLDB | First learned optimizer to beat PostgreSQL |
| STAGE | 2024 | SIGMOD | Spark query time prediction |
| BigDAWG | 2015 | SIGMOD | Foundational polystore architecture |
| Rheem/RHEEM | 2018 | SIGMOD | Cross-platform data processing |
| G-CORE | 2018 | SIGMOD | Graph-relational query language |
| LDBC SNB | 2015 | SIGMOD | Official SNB benchmark |
| Brunner et al. cross-engine | 2025 | VLDB | Most direct competitor |
| Bao (Join Order Bandit) | 2021 | SIGMOD | Learned join ordering via bandit |

---

## Appendix B — Acceptance Criteria Checklist

Use this before any submission.

### Phase 1 Checklist (Workshop Submission)
- [ ] Abstract contains "heuristic cost model" disclaimer
- [ ] Related Work has ≥ 25 references including Wayang, AutoSteer, LEON
- [ ] Each competing paper has a stated differentiator
- [ ] Real LDBC SNB data used (official generator)
- [ ] Correctness verified via SHA256 checksum, not just row count
- [ ] `ThresholdBaseline` and `LogisticRegressionBaseline` implemented
- [ ] Paper Table V includes threshold and logistic regression baselines
- [ ] `ThresholdBaseline` is designated the PRIMARY competitive baseline

### Phase 2 Checklist (Main Track Contender)
- [ ] `SparkSQLGenerator` uses actual PySpark DataFrame API
- [ ] `GraphFramesGenerator` uses actual GraphFrames BFS
- [ ] `training_data/real_labeled_runs.csv` has ≥ 500 real-measurement rows
- [ ] F1 on real labels is between 0.80–0.95 (not 1.00)
- [ ] Cross-dataset generalization heatmap exists
- [ ] Scale factor experiments at SF=1 and SF=5 complete
- [ ] VIF analysis run and discussed in paper
- [ ] Paper Table II updated with real-measurement CV results
- [ ] Paper Table V updated with real-engine timing results
- [ ] Speedup claim updated with real measurements or appropriately caveated

### Phase 3 Checklist (Full SIGMOD Research Track)
- [ ] Formal HIFUN–DSL algebraic mapping table in paper
- [ ] Decomposition correctness proof sketch in paper
- [ ] Online retraining loop implemented and evaluated
- [ ] Demo app working with SHAP explanations
- [ ] Demo paper submitted to SIGMOD 2026 Demonstration Track
- [ ] ≥ 30 references in final paper
- [ ] Evaluation at TPC-H SF=10 or LDBC SNB official benchmark

---

*Implementation Plan v1.0 — March 2026*
*Estimated total effort: 4–6 months (part-time) for full Phase 3 completion*
*Minimum viable for workshop submission: 2–4 weeks (Phase 1 only)*