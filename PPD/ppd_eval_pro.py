#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PPD Evaluation (PRO): overall & stratified metrics, ROC CSVs, nearest-neighbor privacy, EPDS regression.
#
# Examples:
#   python ppd_eval_pro.py --input dataset.csv --synth artifacts/step2_synthetic.jsonl --outdir ppd_eval_pro
#   python ppd_eval_pro.py --input dataset.csv --synth artifacts/step2_synthetic.jsonl --controls controls_template.csv --strata social_support,feeding,marital_strain --outdir ppd_eval_pro
#   python ppd_eval_pro.py --jsonl-input posts.jsonl --synth artifacts/step2_synthetic.jsonl --epds-input epds_dataset.csv --outdir ppd_eval_pro
#
# Notes:
#   - Real dataset must have text (Tweets/Post/text) and label column (Labels/raw_label/label) where 'postpartum' => positive.
#   - Synthetic JSONL must have 'vignette' (or 'text') and 'epds_total' (or 'label').
#   - EPDS input (optional) can be CSV or JSONL with text (Tweets/Post/text) and epds score (epds, EPDS, or epds_total).

import os, math, json, argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, mean_absolute_error, mean_squared_error
)
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------- Loading helpers -----------------------

def _resolve_text_and_label_cols(df: pd.DataFrame):
    text_col = None
    for c in ["text", "Post", "Tweets"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError("Could not find a text column. Expected one of: text, Post, Tweets.")
    label_col = None
    for c in ["Labels", "raw_label", "label"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("Could not find a label column. Expected one of: Labels, raw_label, label.")
    return text_col, label_col

def _to_binary_label(series: pd.Series) -> pd.Series:
    def map_fn(x):
        s = str(x).strip().lower()
        if s in ["1", "true", "ppd", "postpartum", "positive", "yes"]:
            return 1
        if s in ["0", "false", "non-ppd", "control", "negative", "no"]:
            return 0
        return 1 if s == "postpartum" else 0
    try:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(int).clip(0,1)
    except Exception:
        pass
    return series.apply(map_fn).astype(int)

def load_real_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Tweets" in df.columns and "Labels" in df.columns:
        df = df.rename(columns={"Tweets":"text","Labels":"raw_label"})
    text_col, label_col = _resolve_text_and_label_cols(df)
    df = df.rename(columns={text_col:"text", label_col:"raw_label"})
    df["label"] = _to_binary_label(df["raw_label"])
    return df

def load_real_jsonl(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if "text" not in df.columns and "Post" in df.columns:
        df = df.rename(columns={"Post":"text"})
    if "text" not in df.columns:
        raise ValueError("JSONL must contain 'text' or 'Post'.")
    if not any(c in df.columns for c in ["label","raw_label","Labels"]):
        raise ValueError("JSONL must contain 'label' or 'raw_label' or 'Labels'.")
    for c in ["label","raw_label","Labels"]:
        if c in df.columns:
            df = df.rename(columns={c:"raw_label"})
            break
    df["label"] = _to_binary_label(df["raw_label"])
    return df

def load_synth_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if "vignette" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"vignette":"text"})
    if "text" not in df.columns:
        raise ValueError("Synthetic JSONL must contain 'vignette' or 'text'.")
    if "epds_total" in df.columns:
        df["label"] = (pd.to_numeric(df["epds_total"], errors="coerce").fillna(0) >= 13).astype(int)
    elif "label" in df.columns:
        df["label"] = _to_binary_label(df["label"])
    else:
        raise ValueError("Synthetic JSONL must contain 'epds_total' or 'label'.")
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))
    return df[["text","label","row_id"]]

def load_controls(path: Path) -> pd.DataFrame:
    cdf = pd.read_csv(path)
    key = None
    for k in ["row_id","id","index"]:
        if k in cdf.columns:
            key = k; break
    if key is None:
        raise ValueError("Controls CSV must include a row_id (or id/index) column.")
    return cdf.rename(columns={key:"row_id"})

def load_epds(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    # text column
    if "text" not in df.columns:
        if "Post" in df.columns:
            df = df.rename(columns={"Post":"text"})
        elif "Tweets" in df.columns:
            df = df.rename(columns={"Tweets":"text"})
        else:
            raise ValueError("EPDS file must contain text/Post/Tweets")
    # epds score
    epds_col = None
    for c in ["epds","EPDS","epds_total","score","label"]:
        if c in df.columns:
            epds_col = c; break
    if epds_col is None:
        raise ValueError("EPDS file must contain an EPDS-like column (epds, EPDS, epds_total, score, label).")
    df = df.rename(columns={epds_col:"epds"})
    df["epds"] = pd.to_numeric(df["epds"], errors="coerce")
    df = df.dropna(subset=["epds"])
    return df[["text","epds"]]

# ----------------------- Metrics helpers -----------------------

def compute_cls_metrics(y_true, y_prob):
    if len(set(y_true)) < 2:
        return {"accuracy": float("nan"),"precision": float("nan"),"recall": float("nan"),"f1": float("nan"),"auroc": float("nan")}
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob)
    }

def save_roc_csv(path: Path, y_true, y_prob, model_name: str):
    if len(set(y_true)) < 2:
        pd.DataFrame([{"model":model_name,"note":"single-class test set"}]).to_csv(path, index=False)
        return
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    pd.DataFrame({"model":model_name, "threshold": thr, "fpr": fpr, "tpr": tpr}).to_csv(path, index=False)

# ----------------------- Privacy NN -----------------------

def nearest_neighbor_privacy(real_texts, synth_texts, out_csv: Path, thresh: float = 0.92, chunk: int = 512):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)
    X_real = vec.fit_transform(real_texts)
    X_synth = vec.transform(synth_texts)
    from sklearn.metrics.pairwise import cosine_similarity
    rows = []
    for i in range(0, X_synth.shape[0], chunk):
        block = X_synth[i:i+chunk]
        sims = cosine_similarity(block, X_real)
        max_sim = sims.max(axis=1)
        argmax = sims.argmax(axis=1)
        for j in range(block.shape[0]):
            rows.append({"synth_index": i+j, "nearest_real_index": int(argmax[j]), "max_cosine": float(max_sim[j]), "flagged": bool(max_sim[j] >= thresh)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(description="PPD Evaluation (PRO)")
    ap.add_argument("--input", help="Real CSV")
    ap.add_argument("--jsonl-input", help="Real JSONL")
    ap.add_argument("--synth", required=True, help="Synthetic JSONL (step2_synthetic.jsonl)")
    ap.add_argument("--outdir", default="ppd_eval_pro", help="Output directory")
    ap.add_argument("--controls", help="Controls CSV with row_id and strata columns")
    ap.add_argument("--strata", default="social_support,feeding,marital_strain", help="Comma-separated column names from controls to compute per-stratum metrics")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    ap.add_argument("--privacy-threshold", type=float, default=0.92, help="Flag synthetic samples with cosine >= threshold")
    ap.add_argument("--epds-input", help="Optional CSV/JSONL with EPDS-labeled texts for regression")
    args = ap.parse_args()

    if not args.input and not args.jsonl_input:
        raise SystemExit("Provide --input CSV or --jsonl-input JSONL for the real dataset.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    real_df = load_real_csv(Path(args.input)) if args.input else load_real_jsonl(Path(args.jsonl_input))
    synth_df = load_synth_jsonl(Path(args.synth))

    # Attach row_id to real after loading
    real_df = real_df.reset_index(drop=True)
    real_df["row_id"] = np.arange(len(real_df))

    # Controls
    controls_df = None
    strata_cols = [s.strip() for s in args.strata.split(",") if s.strip()]
    if args.controls:
        controls_df = load_controls(Path(args.controls))
        controls_df["row_id"] = pd.to_numeric(controls_df["row_id"], errors="coerce").astype("Int64")

    # Train/test split on REAL
    X_real = real_df["text"].astype(str).tolist()
    y_real = real_df["label"].astype(int).tolist()
    Xr_tr, Xr_te, yr_tr, yr_te, idx_tr, idx_te = train_test_split(
        X_real, y_real, real_df["row_id"].tolist(),
        test_size=args.test_size, random_state=42, stratify=y_real if len(set(y_real))>1 else None
    )

    def build_clf():
        return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)),
                         ("logreg", LogisticRegression(max_iter=1000))])

    # Real-only
    clf_real = build_clf(); clf_real.fit(Xr_tr, yr_tr)
    yprob_real_on_real = clf_real.predict_proba(Xr_te)[:,1]
    # Synth-only
    X_synth = synth_df["text"].astype(str).tolist()
    y_synth = synth_df["label"].astype(int).tolist()
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_synth, y_synth, test_size=min(0.2, 0.2 if len(set(y_synth))>1 else 0.1),
                                                  random_state=42, stratify=y_synth if len(set(y_synth))>1 else None)
    clf_synth = build_clf(); clf_synth.fit(Xs_tr, ys_tr)
    yprob_synth_on_real = clf_synth.predict_proba(Xr_te)[:,1]
    yprob_synth_on_synth = clf_synth.predict_proba(Xs_te)[:,1]
    # Combined
    X_comb_tr = Xr_tr + Xs_tr; y_comb_tr = yr_tr + ys_tr
    clf_comb = build_clf(); clf_comb.fit(X_comb_tr, y_comb_tr)
    yprob_comb_on_real = clf_comb.predict_proba(Xr_te)[:,1]

    # Overall metrics
    overall_rows = []
    overall_rows.append({"dataset":"real_trained", **compute_cls_metrics(yr_te, yprob_real_on_real)})
    overall_rows.append({"dataset":"synth_trained_on_real_test", **compute_cls_metrics(yr_te, yprob_synth_on_real)})
    overall_rows.append({"dataset":"combined_trained_on_real_test", **compute_cls_metrics(yr_te, yprob_comb_on_real)})
    overall_rows.append({"dataset":"synth_trained_on_synth_holdout", **compute_cls_metrics(ys_te, yprob_synth_on_synth)})
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(outdir / "cls_metrics_overall.csv", index=False)

    # ROC CSVs (overall, on the same REAL test set)
    save_roc_csv(outdir / "roc_real_trained_on_real_test.csv", yr_te, yprob_real_on_real, "real_trained")
    save_roc_csv(outdir / "roc_synth_trained_on_real_test.csv", yr_te, yprob_synth_on_real, "synth_trained_on_real_test")
    save_roc_csv(outdir / "roc_combined_trained_on_real_test.csv", yr_te, yprob_comb_on_real, "combined_trained_on_real_test")

    # Fidelity (centroid cosine, TF-IDF + SVD) between REAL and SYNTH
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)
    mat = vec.fit_transform(real_df["text"].tolist() + synth_df["text"].tolist())
    svd = TruncatedSVD(n_components=50, random_state=42)
    emb = svd.fit_transform(mat)
    real_emb = emb[:len(real_df)]; synth_emb = emb[len(real_df):]
    centroid_cosine = float(cosine_similarity(real_emb.mean(axis=0, keepdims=True), synth_emb.mean(axis=0, keepdims=True))[0][0])
    pd.DataFrame([{"metric":"centroid_cosine", "value": centroid_cosine}]).to_csv(outdir / "fidelity_centroid_cosine.csv", index=False)

    # Stratified metrics (by controls, on REAL test subset only)
    strata_rows = []
    if controls_df is not None and len(strata_cols) > 0:
        te_df = pd.DataFrame({"row_id": idx_te, "y": yr_te, "yprob_real": yprob_real_on_real,
                              "yprob_synth": yprob_synth_on_real, "yprob_comb": yprob_comb_on_real})
        te_df = te_df.merge(controls_df, on="row_id", how="left")
        for col in strata_cols:
            if col not in te_df.columns: 
                continue
            for val, grp in te_df.groupby(col, dropna=False):
                if len(grp) < 10 or len(set(grp["y"])) < 2:
                    metrics_real = {k: float("nan") for k in ["accuracy","precision","recall","f1","auroc"]}
                    metrics_synth = metrics_real.copy()
                    metrics_comb  = metrics_real.copy()
                else:
                    metrics_real  = compute_cls_metrics(grp["y"].tolist(), grp["yprob_real"].tolist())
                    metrics_synth = compute_cls_metrics(grp["y"].tolist(), grp["yprob_synth"].tolist())
                    metrics_comb  = compute_cls_metrics(grp["y"].tolist(), grp["yprob_comb"].tolist())
                strata_rows.append({"dataset":"real_trained", "stratum":col, "value":val, "n":len(grp), **metrics_real})
                strata_rows.append({"dataset":"synth_trained_on_real_test", "stratum":col, "value":val, "n":len(grp), **metrics_synth})
                strata_rows.append({"dataset":"combined_trained_on_real_test", "stratum":col, "value":val, "n":len(grp), **metrics_comb})
        if strata_rows:
            pd.DataFrame(strata_rows).to_csv(outdir / "cls_metrics_by_stratum.csv", index=False)

    # Nearest-neighbor privacy (synth vs real)
    nearest_neighbor_privacy(
        real_texts=real_df["text"].tolist(),
        synth_texts=synth_df["text"].tolist(),
        out_csv=outdir / "privacy_nearest_neighbor.csv",
        thresh=args.privacy_threshold
    )

    # Optional: EPDS regression on external labeled dataset
    if args.epds_input:
        epds_df = load_epds(Path(args.epds_input))
        X, y = epds_df["text"].astype(str).tolist(), epds_df["epds"].astype(float).tolist()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=10000)),
                        ("linreg", LinearRegression())])
        reg.fit(Xtr, ytr)
        yhat = reg.predict(Xte)
        mae = mean_absolute_error(yte, yhat)
        rmse = math.sqrt(mean_squared_error(yte, yhat))
        pd.DataFrame([{"dataset":"external_epds", "n_train":len(Xtr), "n_test":len(Xte), "mae":mae, "rmse":rmse}]).to_csv(outdir / "epds_regression_metrics.csv", index=False)
        with open(outdir / "epds_regression.pkl", "wb") as f: pickle.dump(reg, f)

    # Save classification models
    with open(outdir / "clf_real.pkl", "wb") as f: pickle.dump(clf_real, f)
    with open(outdir / "clf_synth.pkl", "wb") as f: pickle.dump(clf_synth, f)
    with open(outdir / "clf_combined.pkl", "wb") as f: pickle.dump(clf_comb, f)

    print("Saved files in:", outdir)
    print("- cls_metrics_overall.csv")
    print("- roc_*.csv")
    if strata_rows:
        print("- cls_metrics_by_stratum.csv")
    print("- privacy_nearest_neighbor.csv")
    print("- fidelity_centroid_cosine.csv")
    if args.epds_input:
        print("- epds_regression_metrics.csv, epds_regression.pkl")
    print("- clf_real.pkl, clf_synth.pkl, clf_combined.pkl")

if __name__ == "__main__":
    main()
