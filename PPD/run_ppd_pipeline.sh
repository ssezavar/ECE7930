#!/usr/bin/env bash
# run_ppd_pipeline.sh — End-to-end pipeline to generate synthetic PPD data, evaluate, plot, and report.
# Usage:
#   bash run_ppd_pipeline.sh -d dataset.csv -o artifacts -e ppd_eval_pro -m gpt-4o-mini -n 200 -c controls.csv -s social_support,feeding,marital_strain
# Notes:
#   - Requires Python 3.9+.
#   - Set OPENAI_API_KEY in your environment for generation.

set -euo pipefail

# Defaults
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
  -d  Path to dataset CSV (default: dataset.csv)
  -o  Output directory for synthetic generation (default: artifacts)
  -e  Evaluation output directory (default: ppd_eval_pro)
  -m  OpenAI model for generation (default: gpt-4o-mini)
  -n  Max records to process for generation (default: 200)
  -c  Controls CSV (optional)
  -s  Strata columns for evaluation (comma-separated; default: social_support,feeding,marital_strain)
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

# Check dataset
if [ ! -f "$DATASET" ]; then
  echo "ERROR: dataset file not found: $DATASET"
  exit 1
fi

# Check scripts
need=(ppd_generate_openai_plus.py ppd_eval_pro.py plot_roc_and_metrics.py report_onepager.py)
for s in "${need[@]}"; do
  if [ ! -f "$s" ]; then
    echo "ERROR: missing script $s. Please download it to the working directory."
    exit 1
  fi
done

# Env + deps
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Core deps
pip install openai pandas numpy scikit-learn matplotlib reportlab

# Require OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it and re-run."
  exit 1
fi

# 1) Generate synthetic data
python ppd_generate_openai_plus.py --input "$DATASET" --outdir "$OUTDIR" --model "$MODEL" --max-records "$MAXREC" ${CONTROLS:+--controls "$CONTROLS"}

# 2) Evaluate (pro)
python ppd_eval_pro.py --input "$DATASET" --synth "$OUTDIR/step2_synthetic.jsonl" --outdir "$EVALDIR" ${CONTROLS:+--controls "$CONTROLS"} ${STRATA:+--strata "$STRATA"}

# 3) Plots
python plot_roc_and_metrics.py --evaldir "$EVALDIR" --outdir "$EVALDIR"

# 4) One-page PDF report
python report_onepager.py --evaldir "$EVALDIR" --outdir "$EVALDIR" --title "PPD Synthetic Data Evaluation"

echo "All done."
echo "Artifacts:"
echo "- Synthetic: $OUTDIR"
echo "- Evaluation: $EVALDIR (metrics CSVs, ROC CSVs, plots, ppd_report.pdf)"
