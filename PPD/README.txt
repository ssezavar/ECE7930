Folder v5__artifacts_llama: this folder includes a script subdirectory that contains two scripts:

## ppd_generate_batches_v5.py — Synthetic Postpartum Narrative Generator (v5)

This script implements the full v5 synthetic–data generation pipeline for postpartum
depression (PPD) research. It reads an input CSV of real postpartum posts and processes
them in parallel batches to produce high-fidelity synthetic diary-style narratives.

The pipeline uses a two-step LLM workflow:
1. **Step 1 – Abstraction:** The model generates structured summaries, salient factors,
   and an estimated EPDS risk bucket for each real post.
2. **Step 2 – Generation:** Using the Step-1 outputs, the model generates a 120–250 word
   first-person postpartum diary entry with associated labels.

Key features of the script:
- **Batch processing** with multiprocessing for efficient GPU usage.
- **Automatic retry + throttling logic** to handle Ollama overload, empty outputs, or short responses.
- **Two-stage prompt system** designed specifically for postpartum mental-health language.
- **Configurable sampling parameters** (temperature, top-p, top-k, repeat-penalty, max tokens).
- **Within-batch deduplication** to avoid synthetic repetition.
- **Checkpointing** so interrupted runs resume where they left off.
- **Progress bar** for real-time batch progress monitoring.
- **Optional email notification** when the run is complete.
- **Automatic shutdown** of the AWS instance at the end of the job.
- **Final output** is a merged CSV containing:
  - the original post and metadata,
  - Step-1 summary + estimated risk,
  - Step-2 synthetic diary text + style + target label.

In short, this script is the core engine that produces the v5 synthetic dataset, managing
LLM calls, batching, robustness, fault tolerance, and parallel GPU execution.


## run_ppd_generate_batches_v5.sh — Example Command to Launch the v5 Generator

This file provides a ready-to-run command for generating the full v5 synthetic
postpartum dataset using the `ppd_generate_batches_v5.py` pipeline. It serves as a
template for running the generator with recommended parameters on either local
hardware or an AWS GPU instance.

The command specifies:
- the input dataset (`dataset.csv`)
- the output directory where batch artifacts and logs are saved
- the Llama 3.1 8B model to load via Ollama
- number of parallel workers (GPU-backed)
- two-stage sampling parameters (temperature, top-p, top-k, repeat penalty)
- token limits for Step-1 summaries and Step-2 diary generation

This script does not contain logic itself — it simply demonstrates the full,
correct invocation of the generator with the optimized parameter configuration
used for the v5 dataset.

Researchers can copy, modify, or extend this command to:
- adjust model parameters,
- run on different datasets,
- test alternative sampling settings,
- or reproduce the dataset used in this report.

--------------------------------------------------------------------------------------------------------------------
v5__quant_llama_eval:
this folder includes the "dedupe_and_eval" script. 
## dedupe_and_eval.py — Deduplication + Utility/Fidelity/Privacy Evaluation

This script performs an end-to-end cleaning and evaluation workflow for real and 
synthetic postpartum-depression (PPD) text data. It is designed to both sanitize 
the synthetic dataset and compute the full set of ML-based evaluation metrics 
reported in the study.

### What the script does

#### 1. Load & Clean Data
- Loads `real.csv` and `synthetic.csv`.
- Normalizes text column names (`Post` → `text`, `post` → `text`).
- Cleans empty, null, or malformed rows to prevent TF–IDF parsing errors.

#### 2. Label Mapping
- Converts text labels into a **binary classification target**  
  (`1 = postpartum/PPD`, `0 = non-PPD`) using keyword heuristics.

#### 3. Deduplication (TF–IDF similarity > 0.90)
- Removes exact duplicates.
- Computes TF–IDF cosine similarity across all synthetic rows.
- Drops near-duplicates above the similarity threshold.
- Saves the cleaned output as `synthetic_clean.csv`.

#### 4. Utility Evaluation (Classifier Performance)
- Trains a Multinomial Naive Bayes classifier using:
  - **real-only data**,  
  - **synthetic-only data**,  
  - **combined data (real + synthetic)**.
- Computes and saves:
  - accuracy, precision, recall, F1
  - `utility_metrics.csv`
  - bar-chart visualizations (`utility_bar_v0.png` and the readable `utility_bar.png`)

#### 5. Fidelity Evaluation (Similarity to Real Data)
- Computes TF–IDF **centroid cosine similarity**.
- Performs **joint PCA** for real and synthetic embeddings.
- Saves the PCA visualization to `pca_real_vs_synth.png`.
- Saves fidelity metrics to `fidelity_metrics.json`.

#### 6. Privacy Evaluation
- Computes 3- to 5-gram overlap using a CountVectorizer.
- Measures maximum cross-similarity between real and synthetic text.
- Saves results to `privacy_report.json`.

### Output Summary
Running this script produces:
- Cleaned synthetic dataset  
- Utility metrics + plots  
- Fidelity metrics + PCA plot  
- Privacy metrics  
- All outputs written to the working directory

In short, `dedupe_and_eval.py` provides the complete evaluation pipeline for analyzing the 
quality, realism, and privacy characteristics of synthetic postpartum narratives.
