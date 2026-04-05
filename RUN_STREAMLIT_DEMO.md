# HIFUN 4-Min Video Script (Fast Terminal + Strict UI)

Use this guide to record a clean 4-minute project demo with:
- Fast mode findings shown in terminal.
- Strict mode findings shown in Streamlit UI.

All commands are from project root.

## 1) Pre-record Setup (2-3 minutes before recording)

```bash
cd /home/mitta/HIFUN-Router-clone
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

Install Streamlit once if needed:

```bash
pip install streamlit==1.44.1
```

## 2) Prepare fast artifacts for terminal section

This generates quick runtime artifacts used by the dashboard fast profile.

```bash
# Quick shift artifact
python experiments/dataset_shift_evaluation.py \
	--source training_data/real_labeled_runs_strict_curated.csv \
	--out_json experiments/results/dataset_shift_eval_fast_runtime.json \
	--out_md experiments/results/dataset_shift_eval_fast_runtime.md

# Quick robustness artifact (reduced bootstrap/permutation counts)
python experiments/strict_robustness_evaluation.py \
	--train training_data/fixed_train_base_strict.csv \
	--eval training_data/fixed_eval_set_strict.csv \
	--transfer_source training_data/real_labeled_runs_strict_curated.csv \
	--n_bootstrap 150 \
	--n_perm_labels 20 \
	--n_perm_features 8 \
	--out_json experiments/results/strict_robustness_eval_fast_runtime.json \
	--out_md experiments/results/strict_robustness_eval_fast_runtime.md
```

Optional strict refresh (longer, skip during recording):

```bash
./run_project_strict.sh smoke
```

## 3) Start dashboard

```bash
streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

Open:

```text
http://localhost:8501
```

## 4) Recording script (4:00 total)

## Segment A (0:00-0:20) - Intro

On screen:
- Show project root in terminal and Streamlit tab open.

Say:
"This is HIFUN Router, a hybrid SQL-vs-GRAPH routing system. In this demo, I show quick terminal findings in fast mode, then strict evidence in the dashboard for final reporting quality."

## Segment B (0:20-1:25) - Fast mode findings in terminal

On screen:
- Terminal only.
- Run and show command outputs with saved artifact paths.

Commands to run live:

```bash
python experiments/dataset_shift_evaluation.py \
	--source training_data/real_labeled_runs_strict_curated.csv \
	--out_json experiments/results/dataset_shift_eval_fast_runtime.json \
	--out_md experiments/results/dataset_shift_eval_fast_runtime.md

python experiments/strict_robustness_evaluation.py \
	--train training_data/fixed_train_base_strict.csv \
	--eval training_data/fixed_eval_set_strict.csv \
	--transfer_source training_data/real_labeled_runs_strict_curated.csv \
	--n_bootstrap 150 \
	--n_perm_labels 20 \
	--n_perm_features 8 \
	--out_json experiments/results/strict_robustness_eval_fast_runtime.json \
	--out_md experiments/results/strict_robustness_eval_fast_runtime.md
```

Say:
"Fast mode is for quick iteration. We reduce robustness sampling so we can quickly validate trend direction and artifact health before strict reporting."

Optional one-line verification:

```bash
ls -lh experiments/results/*fast_runtime.json experiments/results/*fast_runtime.md
```

## Segment C (1:25-1:40) - Transition from terminal to UI

On screen:
- Switch to browser with Streamlit app.
- Sidebar: select `fast` briefly to show that quick artifacts are loaded.

Say:
"The dashboard can load this fast profile for quick checks. For publication-grade findings, we switch to strict profile."

## Segment D (1:40-3:35) - Strict mode findings in UI

On screen:
- Sidebar: select `strict`.
- Keep `Compact view` enabled.
- Walk tabs in this order:
	1. Dataset and Quality
	2. Relevance Evaluation
	3. Robustness Evaluation
	4. Cross-Dataset Generalization

Say (suggested lines):

1) Dataset and Quality (1:40-2:00)
"This summarizes train/eval size, label distribution, and quality gates. We use strict curated real measurements for reliable claims."

2) Relevance Evaluation (2:00-2:35)
"Here are model-level routing metrics. The key finding is parity between learned routing and the traversal rule under strict evaluation, which supports robustness of the decision boundary."

3) Robustness Evaluation (2:35-3:05)
"Bootstrap confidence intervals and permutation sanity checks indicate the model signal is stable and not due to leakage."

4) Cross-Dataset Generalization (3:05-3:35)
"Cross-dataset transfer shows where routing generalizes and where domain shift remains challenging."

## Segment E (3:35-4:00) - Conclusion

On screen:
- Return to Executive View.

Say:
"In summary, fast mode supports rapid iteration in terminal, and strict mode provides trustworthy dashboard evidence for reporting. Next steps are improving graph-class coverage and stronger cross-domain transfer."

## 5) Dashboard utilization checklist (during recording)

- Use sidebar `Run Profile`:
	- `fast` for quick runtime artifacts.
	- `strict` for final reporting artifacts.
- Keep `Compact view` ON during recording to avoid extra detail.
- In `Artifacts` sidebar section, briefly show loaded file paths to reinforce reproducibility.
- Prefer stable tab order: Dataset and Quality -> Relevance Evaluation -> Robustness Evaluation -> Cross-Dataset Generalization.
- Avoid scrolling too much; pause 2-3 seconds per key metric card.

## 6) Camera and pacing tips

- Record at 1080p, 16:9, terminal zoom >= 125%.
- Use one terminal pane and one browser tab only.
- Keep command copy/paste prepared in a scratchpad to avoid typing delays.
- Speak at 130-150 words/minute so the 4-minute target is maintained.

## 7) Stop dashboard

Press `Ctrl+C` in the Streamlit terminal.

## Optional: Run dashboard without activating shell

```bash
cd /home/mitta/HIFUN-Router-clone
.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.headless true
```
