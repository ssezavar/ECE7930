#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantitative evaluation for PPD synthetic data (utility + fidelity).
- Utility: multinomial Naive Bayes (from scratch) on TF/counted features.
- Fidelity: centroid cosine (TF-IDF) + PCA 2D (fit once on joint matrix).
Inputs in current folder:
  - dataset.csv              # real set
  - combined_preview.csv     # synthetic set (uses s2_text + s2_target)
Outputs (./quant/):
  - utility_metrics.csv
  - utility_bar.png
  - fidelity_metrics.json
  - pca_real_vs_synth.png
"""

import os, re, json, math, random, argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="dataset.csv")
    p.add_argument("--synthetic", default="combined_preview.csv")
    p.add_argument("--outdir", default="quant")
    p.add_argument("--sample", type=int, default=0,
                   help="If >0, sample up to N examples from each set (stratified when labels available).")
    p.add_argument("--min-df", type=int, default=5, help="Min document frequency for vocab.")
    p.add_argument("--max-df", type=float, default=0.98, help="Max document frequency ratio for vocab.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--balance-synth", action="store_true",
                   help="Downsample/upsample synthetic to balance classes before training.")
    return p.parse_args()

# --------------------- helpers ---------------------
def tokenize(s: str):
    s = (s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks

def build_vocab(docs, min_df=5, max_df=0.98):
    df_counts = {}
    N = max(1, len(docs))
    for doc in docs:
        seen = set(tokenize(doc))
        for t in seen:
            df_counts[t] = df_counts.get(t, 0) + 1
    vocab = {}
    for t, dfc in df_counts.items():
        if dfc >= min_df and (dfc / N) <= max_df:
            vocab[t] = len(vocab)
    return vocab

def transform_counts(docs, vocab, dtype=np.float32):
    # Bag-of-words counts (faster for NB than tf-idf)
    V = len(vocab)
    rows = []
    for d in docs:
        vec = np.zeros(V, dtype=dtype)
        for t in tokenize(d):
            j = vocab.get(t)
            if j is not None:
                vec[j] += 1.0
        rows.append(vec)
    return np.vstack(rows) if rows else np.zeros((0, V), dtype=dtype)

def transform_tfidf(docs, vocab, dtype=np.float32):
    # For fidelity embeddings
    V = len(vocab); N = len(docs)
    toks_list = [tokenize(d) for d in docs]
    df = np.zeros(V, dtype=np.float32)
    for toks in toks_list:
        seen = set(t for t in toks if t in vocab)
        for t in seen:
            df[vocab[t]] += 1.0
    idf = np.log((N + 1) / (df + 1)) + 1.0
    rows = []
    for toks in toks_list:
        vec = np.zeros(V, dtype=dtype)
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                vec[j] += 1.0
        if vec.sum() > 0:
            vec = vec / vec.sum()
        rows.append(vec * idf)
    return np.vstack(rows) if rows else np.zeros((0, V), dtype=dtype)

def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def pca_2d_joint(X):
    # PCA on joint matrix (so both sets share the same axes)
    if X.size == 0:
        return np.zeros((0, 2))
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

class NB:  # multinomial naive bayes (counts)
    def fit(self, X, y, alpha=1.0):
        self.alpha = alpha
        self.classes_ = np.unique(y)
        self.k = len(self.classes_)
        self.cls_to_idx = {c:i for i,c in enumerate(self.classes_)}
        yidx = np.array([self.cls_to_idx[c] for c in y])
        self.priors_ = np.array([(yidx==i).mean() for i in range(self.k)], dtype=np.float32)
        self.word_sum = np.zeros((self.k, X.shape[1]), dtype=np.float32)
        for i in range(self.k):
            if (yidx==i).any():
                self.word_sum[i] = X[yidx==i].sum(axis=0)
        self.totals = self.word_sum.sum(axis=1)
        self.logprob_ = (np.log(self.word_sum + alpha) -
                         np.log(self.totals[:,None] + alpha*X.shape[1]))
        return self
    def predict(self, X):
        scores = X @ self.logprob_.T + np.log(self.priors_ + 1e-12)
        idx = scores.argmax(axis=1)
        return np.array([self.classes_[i] for i in idx])

def bin_metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    acc = (tp+tn)/max(1,len(y_true))
    prec = tp/max(1,(tp+fp))
    rec = tp/max(1,(tp+fn))
    f1 = 2*prec*rec/max(1,(prec+rec))
    return {"accuracy":acc,"precision":prec,"recall":rec,"F1":f1}

def map_real_label(v):
    v = str(v).strip().lower()
    if v in {"ppd","depressed","high_risk","possible_ppd","postpartum","negative"}: return 1
    if v in {"normal","none","low_risk","positive"}: return 0
    if "ppd" in v or "depress" in v or "neg" in v: return 1
    return 0

def epds_to_binary(v):
    v = str(v)
    if ">=13" in v or "10–12" in v or "10-12" in v: return 1
    if "5–9" in v or "0–4" in v or "5-9" in v or "0-4" in v: return 0
    vl = v.lower()
    return 1 if ("probable" in vl or "possible" in vl) else 0

def stratified_split(texts, labels, frac=0.8, seed=7):
    rng = np.random.default_rng(seed)
    texts = np.array(texts); labels = np.array(labels)
    idx_pos = np.where(labels==1)[0]
    idx_neg = np.where(labels==0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    cut_pos = int(frac*len(idx_pos))
    cut_neg = int(frac*len(idx_neg))
    tr = np.concatenate([idx_pos[:cut_pos], idx_neg[:cut_neg]])
    te = np.concatenate([idx_pos[cut_pos:], idx_neg[cut_neg:]])
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te

def maybe_sample(texts, labels, n, seed=7):
    if n <= 0 or len(texts) <= n: return texts, labels
    rng = np.random.default_rng(seed)
    if labels is None:
        idx = rng.choice(len(texts), size=n, replace=False)
        return [texts[i] for i in idx], None
    # stratified sample
    texts = np.array(texts); labels = np.array(labels)
    pos = np.where(labels==1)[0]; neg = np.where(labels==0)[0]
    n_half = n//2
    n_pos = min(n_half, len(pos)); n_neg = min(n - n_pos, len(neg))
    idx = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                          rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(idx)
    return [texts[i] for i in idx], labels[idx]

def balance_binary(texts, labels, seed=7):
    # down/upsample to 50/50
    rng = np.random.default_rng(seed)
    texts = np.array(texts); labels = np.array(labels)
    pos = np.where(labels==1)[0]; neg = np.where(labels==0)[0]
    if len(pos)==0 or len(neg)==0: return list(texts), labels
    if len(pos) > len(neg):
        add = rng.choice(neg, size=len(pos)-len(neg), replace=True)
        idx = np.concatenate([pos, neg, add])
    else:
        add = rng.choice(pos, size=len(neg)-len(pos), replace=True)
        idx = np.concatenate([pos, neg, add])
    rng.shuffle(idx)
    return list(texts[idx]), labels[idx]

# --------------------- main ---------------------
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    print("[info] loading data…")
    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)

    rcols = {c.lower(): c for c in real.columns}
    scols = {c.lower(): c for c in synth.columns}

    r_text = rcols.get("post") or rcols.get("tweets") or rcols.get("text") or list(real.columns)[0]
    r_labelcol = rcols.get("label") or rcols.get("labels") or rcols.get("target")

    s_text = scols.get("s2_text") or scols.get("synthetic_text") or scols.get("post") or list(synth.columns)[0]
    s_label_src = scols.get("s2_target") or scols.get("label") or scols.get("labels")

    r_texts = real[r_text].astype(str).fillna("").tolist()
    rY = None
    if r_labelcol is not None and real[r_labelcol].nunique() >= 2:
        rY = real[r_labelcol].map(map_real_label).astype(int).values

    s_texts = synth[s_text].astype(str).fillna("").tolist()
    sY = synth[s_label_src].map(epds_to_binary).astype(int).values if s_label_src else np.array([0,1]*(len(synth)//2+1))[:len(synth)]

    # optional sampling
    if args.sample > 0:
        r_texts, rY = maybe_sample(r_texts, rY, args.sample, seed=args.seed)
        s_texts, sY = maybe_sample(s_texts, sY, args.sample, seed=args.seed)

    # optional balance for synthetic (for meaningful synthetic-only metrics)
    if args.balance_synth:
        s_texts, sY = balance_binary(s_texts, sY, seed=args.seed)

    # ---------------- Utility ----------------
    results = []

    def train_eval(texts, labels, name):
        print(f"[info] training {name}…")
        tr, te = stratified_split(texts, labels, frac=0.8, seed=args.seed)
        vocab = build_vocab([texts[i] for i in tr], min_df=args.min_df, max_df=args.max_df)
        Xtr = transform_counts([texts[i] for i in tr], vocab)
        Xte = transform_counts([texts[i] for i in te], vocab)
        ytr = labels[tr]; yte = labels[te]
        if len(np.unique(ytr)) < 2:  # guard against single-class
            return {"dataset":name, "n_train":len(tr), "n_test":len(te),
                    "accuracy":np.nan, "precision":np.nan, "recall":np.nan, "F1":np.nan}
        clf = NB().fit(Xtr, ytr, alpha=1.0)
        yhat = clf.predict(Xte)
        m = bin_metrics(yte, yhat)
        return {"dataset":name, "n_train":len(tr), "n_test":len(te),
                **{k:round(v,3) for k,v in m.items()}}

    # synthetic-only
    results.append(train_eval(s_texts, sY, "synthetic"))

    # real-only + combined if real labels usable
    if rY is not None and len(set(rY))>1:
        results.append(train_eval(r_texts, rY, "real"))
        # combined
        comb_texts = r_texts + s_texts
        comb_labels = np.concatenate([rY, sY])
        results.append(train_eval(comb_texts, comb_labels, "combined"))

    util = pd.DataFrame(results)
    util_path = os.path.join(args.outdir, "utility_metrics.csv")
    util.to_csv(util_path, index=False)

    # bar figure
    plt.figure(figsize=(6,4))
    metrics_order = ["accuracy","precision","recall","F1"]
    x = np.arange(len(metrics_order))
    width = 0.25
    for i, (_,r) in enumerate(util.set_index("dataset").iterrows()):
        vals = [r.get(m, np.nan) for m in metrics_order]
        plt.bar(x + i*width, vals, width, label=r.name)
    plt.xticks(x + width*(max(1,len(util))-1)/2, metrics_order)
    plt.ylim(0,1.0); plt.ylabel("score"); plt.legend()
    plt.tight_layout()
    util_plot_path = os.path.join(args.outdir, "utility_bar.png")
    plt.savefig(util_plot_path, dpi=160)
    plt.close()

    # ---------------- Fidelity ----------------
    print("[info] computing fidelity (TF-IDF + PCA)…")
    union_vocab = build_vocab(r_texts + s_texts, min_df=args.min_df, max_df=args.max_df)
    Xr = transform_tfidf(r_texts, union_vocab)
    Xs = transform_tfidf(s_texts, union_vocab)
    cr = Xr.mean(axis=0) if len(Xr) else np.zeros(len(union_vocab))
    cs = Xs.mean(axis=0) if len(Xs) else np.zeros(len(union_vocab))
    centroid_cos = float(0.0 if (np.linalg.norm(cr)==0 or np.linalg.norm(cs)==0)
                         else (np.dot(cr,cs)/(np.linalg.norm(cr)*np.linalg.norm(cs))))

    X_joint = np.vstack([Xr, Xs]) if len(Xr) and len(Xs) else np.zeros((0, len(union_vocab)))
    Z = pca_2d_joint(X_joint)
    zr = Z[:len(Xr)] if len(Xr) else np.zeros((0,2))
    zs = Z[len(Xr):] if len(Xs) else np.zeros((0,2))

    plt.figure(figsize=(6,5))
    if len(zr): plt.scatter(zr[:,0], zr[:,1], s=10, alpha=0.4, label="Real")
    if len(zs): plt.scatter(zs[:,0], zs[:,1], s=10, alpha=0.6, label="Synthetic")
    plt.title("Real vs Synthetic (TF-IDF PCA-2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.tight_layout()
    pca_path = os.path.join(args.outdir, "pca_real_vs_synth.png")
    plt.savefig(pca_path, dpi=160)
    plt.close()

    fid_path = os.path.join(args.outdir, "fidelity_metrics.json")
    with open(fid_path, "w", encoding="utf-8") as f:
        json.dump({
            "centroid_cosine_tfidf": round(centroid_cos, 4),
            "n_real": int(len(r_texts)),
            "n_synthetic": int(len(s_texts)),
            "vocab_size": int(len(union_vocab)),
            "params": {"min_df": args.min_df, "max_df": args.max_df, "sample": args.sample},
            "note": "Lower centroid cosine => larger style/semantic gap; utility can still remain high."
        }, f, indent=2)

    print("[done] wrote:")
    print(" -", util_path)
    print(" -", util_plot_path)
    print(" -", fid_path)
    print(" -", pca_path)
    print(f"[time] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
