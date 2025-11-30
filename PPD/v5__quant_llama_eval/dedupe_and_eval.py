import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA


###############################################################################
# 1. LOAD CSVs
###############################################################################
real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")


###############################################################################
# 2. RENAME TEXT COLUMNS
###############################################################################
# Your real.csv uses "Post"
real = real.rename(columns={"Post": "text"})

# Your synthetic.csv uses "post"
synthetic = synthetic.rename(columns={"post": "text"})


###############################################################################
# 3. CLEAN TEXT TO PREVENT TF-IDF CRASHES
###############################################################################
def clean_text(df):
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].replace(
        ["nan", "None", "none", "NaN", "NULL", "null"], ""
    )
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df

real = clean_text(real)
synthetic = clean_text(synthetic)


###############################################################################
# 4. MAP LABEL → binary (1 = postpartum/PPD, 0 = other)
###############################################################################
def map_to_binary(df):
    df = df.copy()
    def convert(lbl):
        s = str(lbl).lower()
        if "postpartum" in s or "ppd" in s or "depress" in s:
            return 1
        return 0
    df["label"] = df["Label"].apply(convert) if "Label" in df.columns else df["label"].apply(convert)
    return df

real = map_to_binary(real)
synthetic = map_to_binary(synthetic)


###############################################################################
# 5. DEDUPLICATE SYNTHETIC (TF-IDF similarity > 0.90)
###############################################################################
def dedupe(df, threshold=0.90):
    df = df.copy()
    df = df.drop_duplicates(subset=["text"])

    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(df["text"])

    to_drop = set()
    for i in range(X.shape[0]):
        if i in to_drop:
            continue
        sims = cosine_similarity(X[i], X).flatten()
        dup_idx = np.where(sims > threshold)[0]
        for j in dup_idx:
            if j != i:
                to_drop.add(j)

    df_clean = df.drop(df.index[list(to_drop)]).reset_index(drop=True)
    print(f"[INFO] Removed {len(to_drop)} near-duplicates")
    return df_clean

synthetic_clean = dedupe(synthetic)
synthetic_clean.to_csv("synthetic_clean.csv", index=False)


###############################################################################
# 6. CLASSIFIER UTILITY
###############################################################################
def eval_classifier(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vec = TfidfVectorizer(stop_words="english")
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    return dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
    )


metrics = {
    "real": eval_classifier(real),
    "synthetic": eval_classifier(synthetic_clean),
}

combined = pd.concat([real, synthetic_clean]).reset_index(drop=True)
metrics["combined"] = eval_classifier(combined)


###############################################################################
# 7. SAVE utility_metrics.csv + BAR PLOT
###############################################################################
df_out = pd.DataFrame([
    ["real",       *metrics["real"].values()],
    ["synthetic",  *metrics["synthetic"].values()],
    ["combined",   *metrics["combined"].values()],
], columns=["source","accuracy","precision","recall","f1"])

df_out.to_csv("utility_metrics.csv", index=False)

# the older version
plt.figure(figsize=(7,5))
df_out.set_index("source").plot(kind="bar")
plt.ylim(0,1)
plt.title("Classification Performance")
plt.ylabel("score")
plt.tight_layout()
plt.savefig("utility_bar_v0.png")
plt.close()

###############################################################################
# 7. SAVE utility_metrics.csv + OLD-STYLE BAR PLOT (readable version)
###############################################################################
df_out = pd.DataFrame([
    ["real",       *metrics["real"].values()],
    ["synthetic",  *metrics["synthetic"].values()],
    ["combined",   *metrics["combined"].values()],
], columns=["source","accuracy","precision","recall","f1"])

df_out.to_csv("utility_metrics.csv", index=False)

# ------- NEW READABLE BAR PLOT -------
metrics_list = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrics_list))
width = 0.25

synthetic_vals = [metrics["synthetic"][m] for m in metrics_list]
real_vals      = [metrics["real"][m]      for m in metrics_list]
combined_vals  = [metrics["combined"][m]  for m in metrics_list]

plt.figure(figsize=(10,6))

plt.bar(x - width, synthetic_vals, width, label='synthetic')
plt.bar(x,         real_vals,      width, label='real')
plt.bar(x + width, combined_vals,  width, label='combined')

plt.xticks(x, [m.capitalize() for m in metrics_list])
plt.ylabel("score")
plt.ylim(0, 1.0)
plt.title("Classification Performance")
plt.legend()

plt.tight_layout()
plt.savefig("utility_bar.png", dpi=300)
plt.close()

###############################################################################
# 8. FIDELITY EVALUATION (centroids + PCA)
###############################################################################
def evaluate_fidelity(real, synth):
    vec = TfidfVectorizer(stop_words='english')
    Xr = vec.fit_transform(real["text"])
    Xs = vec.transform(synth["text"])

    centroid_r = Xr.mean(axis=0).A1
    centroid_s = Xs.mean(axis=0).A1
    centroid_sim = float(cosine_similarity([centroid_r],[centroid_s])[0][0])

    X = np.vstack([Xr.toarray(), Xs.toarray()])
    labels = np.array([0]*len(real) + [1]*len(synth))

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X2[labels==0,0], X2[labels==0,1], alpha=0.4, s=10, label="Real")
    plt.scatter(X2[labels==1,0], X2[labels==1,1], alpha=0.4, s=10, label="Synthetic")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend()
    plt.title("Real vs Synthetic (PCA TF-IDF)")
    plt.tight_layout()
    plt.savefig("pca_real_vs_synth.png")
    plt.close()

    return {"centroid_similarity": centroid_sim}

fidelity = evaluate_fidelity(real, synthetic_clean)
json.dump(fidelity, open("fidelity_metrics.json","w"), indent=2)


###############################################################################
# 9. PRIVACY EVALUATION
###############################################################################
def evaluate_privacy(real, synth):
    vec = CountVectorizer(ngram_range=(3,5), stop_words='english')
    X = vec.fit_transform(list(real["text"]) + list(synth["text"]))
    Xr = X[:len(real)]
    Xs = X[len(real):]

    max_overlap = float(cosine_similarity(Xs, Xr).max())

    return {"max_ngram_overlap": max_overlap}

privacy = evaluate_privacy(real, synthetic_clean)
json.dump(privacy, open("privacy_report.json","w"), indent=2)


print("DONE ✔ All outputs generated.")
