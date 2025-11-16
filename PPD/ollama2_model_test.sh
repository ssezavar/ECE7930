#!/bin/bash

head -n 21 dataset.csv > dataset_small.csv


MODELS=(
  "llama3:8b-instruct-q4_K_M"
  "mixtral:8x7b-instruct-q4_K_M"
  "llama3:8b"
  "mistral:instruct"
)

echo "===== MODEL QUALITY TEST ====="

for M in "${MODELS[@]}"; do
  echo
  echo "----------------------------------------------"
  echo "Testing model: $M"
  echo "----------------------------------------------"

  # Pull model if not installed
  ollama pull $M

  # Generate synthetic data
  python3 ppd_generate_ollama_v2.py \
      --input dataset_small.csv \
      --outdir test_${M//[:\/]/_} \
      --provider ollama \
      --ollama-model "$M" \
      --max-rows 20 \
      --dedup-threshold 0.70

  # Evaluate synthetic vs real
  python3 ppd_eval_quant.py \
      --real dataset_small.csv \
      --synthetic test_${M//[:\/]/_}/combined_preview.csv \
      --outdir eval_${M//[:\/]/_}

  echo "Done evaluating $M"
done

echo "===== ALL MODELS TESTED ====="
