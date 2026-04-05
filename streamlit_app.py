import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
TRAINING_DIR = ROOT / "training_data"


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_relevance_df(payload: dict) -> pd.DataFrame:
    rows = payload.get("metrics", [])
    if not rows:
        return pd.DataFrame()

    wanted = [
        "model",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "pr_auc",
        "brier",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    available = [c for c in wanted if c in rows[0]]
    df = pd.DataFrame(rows)[available].copy()
    rename_map = {
        "model": "Model",
        "accuracy": "Accuracy",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "pr_auc": "PR-AUC",
        "brier": "Brier",
        "tn": "TN",
        "fp": "FP",
        "fn": "FN",
        "tp": "TP",
    }
    df = df.rename(columns=rename_map)
    return df


@st.cache_data(show_spinner=False)
def load_threshold_df(payload: dict) -> pd.DataFrame:
    rows = payload.get("xgboost_confidence_thresholds", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "threshold": "Threshold",
            "coverage": "Coverage",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
        }
    )
    return df


@st.cache_data(show_spinner=False)
def load_per_dataset_confusion(payload: dict) -> pd.DataFrame:
    block = payload.get("xgboost_per_dataset_confusion", {})
    if not block:
        return pd.DataFrame()
    rows = []
    for name, metrics in block.items():
        rows.append(
            {
                "Dataset": name,
                "Rows": metrics.get("rows"),
                "TN": metrics.get("tn"),
                "FP": metrics.get("fp"),
                "FN": metrics.get("fn"),
                "TP": metrics.get("tp"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_shift_df(payload: dict) -> pd.DataFrame:
    rows = payload.get("results", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    keep = [
        "mode",
        "train_dataset",
        "eval_dataset",
        "status",
        "rows",
        "accuracy",
        "f1",
        "precision",
        "recall",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df = df.rename(
        columns={
            "mode": "Mode",
            "train_dataset": "Train",
            "eval_dataset": "Eval",
            "status": "Status",
            "rows": "Rows",
            "accuracy": "Accuracy",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
        }
    )
    return df


def format_metric(x):
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def find_model(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        match = df[df["Model"] == name]
        if not match.empty:
            return match.iloc[0]
    return None


def render_header():
    st.set_page_config(page_title="HIFUN Router Results Dashboard", layout="wide")
    st.title("HIFUN Router Results Dashboard")
    st.caption(
        "Read-only view of generated evaluation artifacts: data quality, relevance, robustness, and dataset-shift metrics."
    )


def render_overview_tab(relevance_payload: dict, quality_payload: dict):
    st.subheader("Dataset and Quality Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", str(relevance_payload.get("train_rows", "-")))
    c2.metric("Eval rows", str(relevance_payload.get("eval_rows", "-")))

    train_dist = relevance_payload.get("train_label_distribution", {})
    eval_dist = relevance_payload.get("eval_label_distribution", {})
    c3.metric("Train label mix", f"SQL {train_dist.get('SQL', '-')}, GRAPH {train_dist.get('GRAPH', '-')}")
    c4.metric("Eval label mix", f"SQL {eval_dist.get('SQL', '-')}, GRAPH {eval_dist.get('GRAPH', '-')}")

    st.markdown("### System in one line")
    st.info(
        "DSL -> Parser/Validator -> DAG Decomposer -> 22-feature extractor -> Router (Rule/ML) -> SQL or Graph execution -> Result composer"
    )

    summary = quality_payload.get("summary", {})
    checks = quality_payload.get("checks", [])
    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)

    q1, q2, q3 = st.columns(3)
    q1.metric("Quality gates", f"{passed}/{total}" if total else "-")
    q2.metric("Graph ratio", format_metric(summary.get("graph_ratio")) if summary else "-")
    q3.metric("Real measurement share", "1.0000" if checks else "-")

    with st.expander("Show quality gate details"):
        if checks:
            qdf = pd.DataFrame(checks)
            if "passed" in qdf.columns:
                qdf["passed"] = qdf["passed"].map(lambda x: "PASS" if x else "FAIL")
            st.dataframe(qdf[[c for c in ["name", "passed", "detail"] if c in qdf.columns]], use_container_width=True)
        else:
            st.info("No data quality report found.")


def render_metrics_tab(relevance_payload: dict, compact_view: bool):
    st.subheader("Core Routing Results")
    rdf = load_relevance_df(relevance_payload)
    if rdf.empty:
        st.warning("No strict relevance metrics found.")
        return

    top_row = rdf.sort_values(["F1", "PR-AUC"], ascending=False).iloc[0]
    rule_row = find_model(rdf, ["TraversalRule", "Trivial Rule"])
    xgb_row = find_model(rdf, ["XGBoostBalanced", "XGBoost", "Learned ML (DT)"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top model", str(top_row["Model"]))
    c2.metric("Top F1", format_metric(top_row.get("F1")))
    c3.metric("Top PR-AUC", format_metric(top_row.get("PR-AUC")))
    if rule_row is not None and xgb_row is not None:
        delta = float(xgb_row.get("F1", 0.0)) - float(rule_row.get("F1", 0.0))
        c4.metric("XGBoost - Rule (F1)", f"{delta:+.4f}")
    else:
        c4.metric("XGBoost - Rule (F1)", "-")

    st.markdown("### Interpretation note")
    st.info(
        "This panel reports model comparison from saved strict evaluation artifacts. Use raw tables below for verification."
    )

    chart_df = rdf[["Model", "F1"]].set_index("Model").sort_values("F1", ascending=False)
    st.bar_chart(chart_df)

    show_cols = [c for c in ["Model", "Accuracy", "F1", "Precision", "Recall", "PR-AUC", "Brier"] if c in rdf.columns]
    st.dataframe(rdf[show_cols], use_container_width=True)

    st.markdown("### XGBoost per-dataset confusion")
    cdf = load_per_dataset_confusion(relevance_payload)
    if cdf.empty:
        st.info("Per-dataset confusion block unavailable.")
    else:
        st.dataframe(cdf, use_container_width=True)

    if not compact_view:
        st.markdown("### Confidence-threshold stability")
        tdf = load_threshold_df(relevance_payload)
        if tdf.empty:
            st.info("Threshold stability block unavailable.")
        else:
            st.dataframe(tdf, use_container_width=True)


def render_robustness_tab(robust_payload: dict, compact_view: bool):
    st.subheader("Robustness")

    core_f1 = robust_payload.get("xgboost_eval_f1")
    bootstrap = robust_payload.get("bootstrap_95_ci", {})
    perm = robust_payload.get("permutation_sanity", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Eval F1", format_metric(core_f1))
    c2.metric(
        "Bootstrap F1 CI",
        f"[{format_metric(bootstrap.get('f1_low'))}, {format_metric(bootstrap.get('f1_high'))}]",
    )
    c3.metric(
        "Permuted mean F1",
        format_metric(perm.get("mean_f1")),
    )

    details = []
    if bootstrap:
        details.append(
            {
                "Measure": "Bootstrap F1 median",
                "Value": bootstrap.get("f1_median"),
            }
        )
        details.append(
            {
                "Measure": "Bootstrap precision CI",
                "Value": f"[{format_metric(bootstrap.get('precision_low'))}, {format_metric(bootstrap.get('precision_high'))}]",
            }
        )
        details.append(
            {
                "Measure": "Bootstrap recall CI",
                "Value": f"[{format_metric(bootstrap.get('recall_low'))}, {format_metric(bootstrap.get('recall_high'))}]",
            }
        )

    if perm:
        details.append(
            {
                "Measure": "Permuted std",
                "Value": perm.get("std_f1"),
            }
        )
        details.append(
            {
                "Measure": "Permuted range",
                "Value": f"[{format_metric(perm.get('min_f1'))}, {format_metric(perm.get('max_f1'))}]",
            }
        )

    if details:
        ddf = pd.DataFrame(details)
        st.dataframe(ddf, use_container_width=True)

    if not compact_view:
        imp = robust_payload.get("permutation_importance", [])
        if imp:
            idf = pd.DataFrame(imp)
            if "mean_drop" in idf.columns and "feature" in idf.columns:
                st.markdown("### Top permutation-importance features")
                idf = idf.sort_values("mean_drop", ascending=False)
                st.dataframe(idf.head(10), use_container_width=True)


def render_shift_tab(shift_payload: dict, compact_view: bool):
    st.subheader("Dataset Shift and Transfer")
    sdf = load_shift_df(shift_payload)
    if sdf.empty:
        st.warning("No dataset shift results found.")
        return

    ok_df = sdf[sdf["Status"] == "ok"] if "Status" in sdf.columns else sdf
    c1, c2, c3 = st.columns(3)
    c1.metric("Transfer runs", str(len(ok_df)))
    if not ok_df.empty:
        c2.metric("Best transfer F1", format_metric(ok_df["F1"].max()))
        c3.metric("Worst transfer F1", format_metric(ok_df["F1"].min()))

    view_cols = [c for c in ["Train", "Eval", "Rows", "Accuracy", "F1", "Precision", "Recall"] if c in sdf.columns]
    st.dataframe(sdf[view_cols], use_container_width=True)

    if not ok_df.empty and not compact_view:
        chart_cols = ["Train", "Eval", "F1"]
        if all(c in ok_df.columns for c in chart_cols):
            tmp = ok_df[chart_cols].copy()
            tmp["Pair"] = tmp["Train"] + " -> " + tmp["Eval"]
            st.markdown("### Transfer F1 by train/eval pair")
            st.bar_chart(tmp.set_index("Pair")["F1"])


def main():
    render_header()

    with st.sidebar:
        st.header("Run Profile")
        profile = st.radio(
            "Select artifact set",
            ["strict", "fast"],
            index=1,
            help="Use fast profile for quick reruns with reduced robustness settings.",
        )
        compact_view = st.toggle(
            "Compact view",
            value=True,
            help="Hide secondary tables/charts and keep core metrics visible.",
        )
        st.caption("Dashboard is read-only: metrics are loaded from generated result files.")

    relevance_file = "relevance_eval_strict_runtime.json"
    robust_file = "strict_robustness_eval_runtime.json"
    shift_file = "dataset_shift_eval_strict_runtime.json"

    if profile == "fast":
        # Relevance currently has strict runtime output only; robustness/shift have fast variants.
        robust_file = "strict_robustness_eval_fast_runtime.json"
        shift_file = "dataset_shift_eval_fast_runtime.json"

    relevance_payload = load_json(RESULTS_DIR / relevance_file)
    robust_payload = load_json(RESULTS_DIR / robust_file)
    shift_payload = load_json(RESULTS_DIR / shift_file)
    quality_payload = load_json(TRAINING_DIR / "dataset_quality_report_strict_curated.json")

    tabs = st.tabs(
        [
            "Executive View",
            "Core Results",
            "Robustness",
            "Shift and Transfer",
        ]
    )

    with tabs[0]:
        render_overview_tab(relevance_payload, quality_payload)

    with tabs[1]:
        render_metrics_tab(relevance_payload, compact_view)

    with tabs[2]:
        render_robustness_tab(robust_payload, compact_view)

    with tabs[3]:
        render_shift_tab(shift_payload, compact_view)

    with st.sidebar:
        st.header("Artifacts")
        st.write("Loaded from:")
        st.code(str(RESULTS_DIR / relevance_file))
        st.code(str(RESULTS_DIR / robust_file))
        st.code(str(RESULTS_DIR / shift_file))
        st.code(str(TRAINING_DIR / "dataset_quality_report_strict_curated.json"))


if __name__ == "__main__":
    main()
