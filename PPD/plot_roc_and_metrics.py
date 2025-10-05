#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Plot ROC curves, accuracy/F1 bars, and privacy histogram from evaluator CSVs.

import argparse, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def safe_read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None

def plot_roc(evaldir: Path, outdir: Path):
    files = [
        ("real_trained", evaldir / "roc_real_trained_on_real_test.csv"),
        ("synth_trained_on_real_test", evaldir / "roc_synth_trained_on_real_test.csv"),
        ("combined_trained_on_real_test", evaldir / "roc_combined_trained_on_real_test.csv"),
    ]
    plt.figure()
    found = False
    for label, fp in files:
        df = safe_read_csv(fp)
        if df is None or not {"fpr","tpr"}.issubset(df.columns):
            continue
        found = True
        plt.plot(df["fpr"], df["tpr"], label=label.replace("_"," "))
    if not found:
        return
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Real test set")
    plt.legend(loc="lower right")
    out = outdir / "roc_curves.png"
    plt.savefig(out, bbox_inches="tight", dpi=180)
    plt.close()

def plot_bars(evaldir: Path, outdir: Path):
    df = safe_read_csv(evaldir / "cls_metrics_overall.csv")
    if df is None: return
    for metric in ["accuracy","f1"]:
        if metric not in df.columns: 
            continue
        plt.figure()
        sub = df[["dataset",metric]].copy()
        sub[metric] = sub[metric].astype(float)
        plt.bar(sub["dataset"], sub[metric])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} by model (real test)")
        out = outdir / f"bar_{metric}.png"
        plt.savefig(out, bbox_inches="tight", dpi=180)
        plt.close()

def plot_privacy(evaldir: Path, outdir: Path):
    fp = evaldir / "privacy_nearest_neighbor.csv"
    df = safe_read_csv(fp)
    if df is None or "max_cosine" not in df.columns:
        return
    plt.figure()
    plt.hist(df["max_cosine"], bins=40)
    plt.xlabel("Max cosine similarity to nearest real sample")
    plt.ylabel("Count")
    plt.title("Nearest-neighbor privacy check (synth vs real)")
    out = outdir / "privacy_hist.png"
    plt.savefig(out, bbox_inches="tight", dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldir", default="ppd_eval_pro")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    evaldir = Path(args.evaldir)
    outdir = Path(args.outdir) if args.outdir else evaldir
    outdir.mkdir(parents=True, exist_ok=True)
    plot_roc(evaldir, outdir)
    plot_bars(evaldir, outdir)
    plot_privacy(evaldir, outdir)
    print("Saved plots to", outdir)

if __name__ == "__main__":
    main()
