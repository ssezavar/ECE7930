#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PPD Utility-Fidelity-Privacy Evaluation
# Usage: python ppd_eval.py --input dataset.csv --synth artifacts/step2_synthetic.jsonl --outdir ppd_eval

import os, json, math, argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    ap = argparse.ArgumentParser(description="PPD Utility-Fidelity-Privacy Evaluation")
    ap.add_argument("--input", required=True, help="Real dataset CSV (Tweets,Labels or text,label)")
    ap.add_argument("--synth", required=True, help="Path to step2_synthetic.jsonl")
    ap.add_argument("--outdir", default="ppd_eval", help="Output directory")
    return ap.parse_args()

def load_real(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(["Tweets","Labels"]).issubset(df.columns):
        df = df.rename(columns={"Tweets":"text","Labels":"raw_label"})
    else:
        if "text" not in df.columns:
            raise ValueError("Expected columns (Tweets, Labels) or (text, raw_label).")
        if "raw_label" not in df.columns:
            df["raw_label"] = df.get("label", "no")
    df["label"] = df["raw_label"].apply(lambda x: 1 if str(x).strip().lower()=="postpartum" else 0)
    return df

def load_synth(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    df = pd.DataFrame(rows).rename(columns={"vignette":"text","epds_total":"epds"})
    if "text" not in df.columns:
        raise ValueError("Synthetic file missing vignette/text")
    df["label"] = (df["epds"] >= 13).astype(int)
    return df

def train_eval_classification(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)),
        ("logreg", LogisticRegression(max_iter=1000))
    ])
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    return clf, {
        "dataset": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_test, y_prob) if len(set(y_test))>1 else float("nan")
    }

def train_eval_regression(name, X, y):
    pairs = [(tx, ty) for tx, ty in zip(X, y) if not (ty is None or (isinstance(ty, float) and np.isnan(ty)))]
    if len(pairs) < 20:
        return None, {"dataset":name,"n_train":0,"n_test":0,"mae":float("nan"),"rmse":float("nan")}
    Xf, yf = zip(*pairs)
    X_train, X_test, y_train, y_test = train_test_split(list(Xf), list(yf), test_size=0.2, random_state=42)
    reg = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)),
        ("linreg", LinearRegression())
    ])
    reg.fit(X_train, y_train)
    y_hat = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    rmse = math.sqrt(mean_squared_error(y_test, y_hat))
    return reg, {"dataset":name,"n_train":len(X_train),"n_test":len(X_test),"mae":mae,"rmse":rmse}

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    real_df = load_real(Path(args.input))
    synth_df = load_synth(Path(args.synth))

    MAX_REAL = 2000
    if len(real_df) > MAX_REAL:
        ppd = real_df[real_df["label"]==1].sample(min(1000, real_df["label"].sum()), random_state=42)
        non = real_df[real_df["label"]==0].sample(min(1000, (real_df["label"]==0).sum()), random_state=42)
        real_df = pd.concat([ppd, non]).sample(frac=1, random_state=42)

    real_X, real_y = real_df["text"].astype(str).tolist(), real_df["label"].astype(int).tolist()
    synth_X, synth_y = synth_df["text"].astype(str).tolist(), synth_df["label"].astype(int).tolist()
    synth_epds = synth_df.get("epds", pd.Series([np.nan]*len(synth_df))).values
    combined_X = real_X + synth_X
    combined_y = real_y + synth_y

    clf_real, m_real = train_eval_classification("real", real_X, real_y)
    clf_synth, m_synth = train_eval_classification("synthetic", synth_X, synth_y)
    clf_comb, m_comb = train_eval_classification("combined", combined_X, combined_y)
    reg_synth, r_synth = train_eval_regression("synthetic_epds", synth_X, synth_epds)

    metrics_cls = pd.DataFrame([m_real, m_synth, m_comb])
    metrics_reg = pd.DataFrame([r_synth])

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)
    mat = vec.fit_transform(combined_X)
    svd = TruncatedSVD(n_components=50, random_state=42)
    emb = svd.fit_transform(mat)
    real_emb = emb[:len(real_X)]
    synth_emb = emb[len(real_X):]
    centroid_cosine = float(cosine_similarity(real_emb.mean(axis=0, keepdims=True),
                                             synth_emb.mean(axis=0, keepdims=True))[0][0])
    fidelity_df = pd.DataFrame({"metric":["centroid_cosine"], "value":[centroid_cosine]})

    metrics_cls.to_csv(outdir / "cls_metrics.csv", index=False)
    metrics_reg.to_csv(outdir / "reg_metrics.csv", index=False)
    fidelity_df.to_csv(outdir / "fidelity_centroid_cosine.csv", index=False)

    with open(outdir / "clf_real.pkl", "wb") as f: pickle.dump(clf_real, f)
    with open(outdir / "clf_synth.pkl", "wb") as f: pickle.dump(clf_synth, f)
    with open(outdir / "clf_combined.pkl", "wb") as f: pickle.dump(clf_comb, f)
    if reg_synth:
        with open(outdir / "reg_synth_epds.pkl", "wb") as f: pickle.dump(reg_synth, f)

    print("Saved to:", outdir)

if __name__ == "__main__":
    main()
