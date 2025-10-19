#!/usr/bin/env bash
set -euo pipefail

### ── Parameters (edit as needed) ─────────────────────────────────────────────
MODEL="mistral:Q4_K_M"          # fast / memory–light model
INPUT="${HOME}/dataset.csv"     # path to dataset (CSV with header)
SCRIPT="${HOME}/ppd_generate_ollama.py"  # your generation script
OUTROOT="${HOME}/artifacts_ollama"       # output folder
CHUNK_SIZE=5000                 # rows per chunk (excluding header)
JOBS=2                          # number of parallel processes

### ── Prerequisites ───────────────────────────────────────────────────────────
echo "[*] Updating & installing deps..."
sudo apt-get update -y
sudo apt-get install -y curl git python3 python3-pip parallel

# Ensure GPU is active (nvidia-smi should produce output)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[!] nvidia-smi not found. Launch a GPU instance (e.g., g5.xlarge)."
  exit 1
fi
nvidia-smi || true

### ── Install Ollama ──────────────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
  echo "[*] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  sudo systemctl enable ollama
  sudo systemctl start ollama
fi

# Simple optimizations
export OLLAMA_NUM_GPU=1
export OLLAMA_FLASH_ATTENTION=1

echo "[*] Pulling model: ${MODEL}"
ollama pull "${MODEL}"

### ── Validate inputs ─────────────────────────────────────────────────────────
[[ -f "${INPUT}" ]]  || { echo "[!] INPUT not found: ${INPUT}";  exit 1; }
[[ -f "${SCRIPT}" ]] || { echo "[!] SCRIPT not found: ${SCRIPT}"; exit 1; }

mkdir -p "${OUTROOT}"
CHUNK_DIR="${OUTROOT}/chunks"
RUN_DIR="${OUTROOT}/runs"
MERGE_DIR="${OUTROOT}/merged"
mkdir -p "${CHUNK_DIR}" "${RUN_DIR}" "${MERGE_DIR}"

### ── Split CSV into chunks while keeping header ──────────────────────────────
echo "[*] Chunking dataset into ${CHUNK_SIZE}-row parts..."
HEADER_FILE="${CHUNK_DIR}/__header__.csv"
head -n 1 "${INPUT}" > "${HEADER_FILE}"
# Total number of lines (including header)
TOTAL_LINES=$(wc -l < "${INPUT}")
# Split the body (without header)
tail -n +2 "${INPUT}" | split -l "${CHUNK_SIZE}" - "${CHUNK_DIR}/part_"

# For each part_* file, prepend the header to make a valid CSV
for f in "${CHUNK_DIR}"/part_*; do
  cat "${HEADER_FILE}" "$f" > "${f}.csv"
  rm -f "$f"
done

PARTS=( "${CHUNK_DIR}"/part_*.csv )
echo "[*] Total parts: ${#PARTS[@]}"

### ── Run generation on each chunk in parallel ────────────────────────────────
echo "[*] Running generation on parts with ${JOBS} parallel jobs..."
export MODEL INPUT SCRIPT RUN_DIR
run_one() {
  local csv="$1"
  local base="$(basename "$csv" .csv)"
  local out="${RUN_DIR}/${base}"
  mkdir -p "${out}"
  python3 "${SCRIPT}" \
    --input "${csv}" \
    --outdir "${out}" \
    --provider ollama \
    --ollama-model "${MODEL}"
}
export -f run_one

# GNU parallel
parallel -j "${JOBS}" --halt now,fail=1 run_one ::: "${PARTS[@]}"

### ── Merge all combined_preview.csv files ────────────────────────────────────
echo "[*] Merging combined_preview.csv files..."
MERGED="${MERGE_DIR}/combined_preview_full.csv"
# Write header once
head -n 1 "${RUN_DIR}/$(basename "${PARTS[0]}" .csv)/combined_preview.csv" > "${MERGED}"
# Append the bodies from all files (without header)
for d in "${RUN_DIR}"/part_*; do
  tail -n +2 "${d}/combined_preview.csv" >> "${MERGED}"
done

echo "[✓] Done."
echo "  - Parts dir:      ${CHUNK_DIR}"
echo "  - Per-part runs:  ${RUN_DIR}"
echo "  - Merged preview: ${MERGED}"
