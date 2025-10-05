#!/usr/bin/env bash
# run_ppd_pipeline_safe.sh — End-to-end pipeline with safe venv handling.
# - If already inside a venv, re-use it.
# - Else, try to create a new venv if python3-venv is available.
# - Else, fall back to system Python and --user installs.
#
# Usage:
#   bash run_ppd_pipeline_safe.sh -d dataset.csv -o artifacts -e ppd_eval_pro -m gpt-4o-mini -n 200 [-c controls.csv] [-s strata]

set -euo pipefail

DATASET="dataset.csv"
OUTDIR="artifacts"
EVALDIR="ppd_eval_pro"
MODEL="gpt-4o-mini"
MAXREC=200
CONTROLS=""
STRATA="social_support,feeding,marital_strain"

usage() {
  cat <<EOF
Usage: $0 [-d dataset.csv] [-o artifacts] [-e eval_dir] [-m model] [-n max_records] [-c controls.csv] [-s strata]
EOF
}

while getopts ":d:o:e:m:n:c:s:h" opt; do
  case $opt in
    d) DATASET="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    e) EVALDIR="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    n) MAXREC="$OPTARG" ;;
    c) CONTROLS="$OPTARG" ;;
    s) STRATA="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done

for s in ppd_generate_openai_plus.py ppd_eval_pro.py plot_roc_and_metrics.py report_onepager.py; do
  [ -f "$s" ] || { echo "Missing $s"; exit 1; }
done
[ -f "$DATASET" ] || { echo "Missing dataset: $DATASET"; exit 1; }

# ---- Environment setup ----
USE_USER=0

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Using existing venv: $VIRTUAL_ENV"
  PY=python
  PIP=pip
else
  if python3 -m venv --help >/dev/null 2>&1; then
    echo "Creating .venv..."
    python3 -m venv .venv
    # shellcheck source=/dev/null
    source .venv/bin/activate
    PY=python
    PIP=pip
  else
    echo "WARNING: python3-venv not available. Installing packages with --user."
    PY=python3
    PIP="python3 -m pip --user"
    USE_USER=1
  fi
fi

# Upgrade pip
$PY -m pip install -U pip

# Install deps
if [ "$USE_USER" -eq 1 ]; then
  python3 -m pip install --user openai pandas numpy scikit-learn matplotlib reportlab
else
  $PIP install openai pandas numpy scikit-learn matplotlib reportlab
fi

# Require key
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "YOUR_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY is not set to a real key."
  exit 1
fi

# 1) Generate
GEN_CMD="$PY ppd_generate_openai_plus.py --input \"$DATASET\" --outdir \"$OUTDIR\" --model \"$MODEL\" --max-records \"$MAXREC\""
if [ -n "$CONTROLS" ]; then GEN_CMD="$GEN_CMD --controls \"$CONTROLS\""; fi
eval $GEN_CMD

# 2) Evaluate
EVAL_CMD="$PY ppd_eval_pro.py --input \"$DATASET\" --synth \"$OUTDIR/step2_synthetic.jsonl\" --outdir \"$EVALDIR\""
if [ -n "$CONTROLS" ]; then EVAL_CMD="$EVAL_CMD --controls \"$CONTROLS\""; fi
if [ -n "$STRATA" ]; then EVAL_CMD="$EVAL_CMD --strata \"$STRATA\""; fi
eval $EVAL_CMD

# 3) Plots + 4) PDF
$PY plot_roc_and_metrics.py --evaldir "$EVALDIR" --outdir "$EVALDIR"
$PY report_onepager.py --evaldir "$EVALDIR" --outdir "$EVALDIR" --title "PPD Synthetic Data Evaluation"

echo "All done."
echo "Synthetic -> $OUTDIR"
echo "Evaluation -> $EVALDIR (metrics CSVs, ROC CSVs, plots, ppd_report.pdf)"
