"""
SmartQBank — Marks-target question recommender — TRAINING SCRIPT (hardened)
==========================================================================
Same pipeline as before, but build_features() now scrubs every question to a
guaranteed clean str and embeds defensively so sentence-transformers 5.x cannot
misclassify any entry as a float.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

EMBED_MODEL = "all-mpnet-base-v2"
IMPORTANT_QUANTILE = 0.70
RANDOM_STATE = 42


def _to_binary(series: pd.Series) -> pd.Series:
    truthy = {"yes", "y", "1", "true", "t"}
    return (series.fillna("").astype(str).str.strip().str.lower()
            .isin(truthy).astype(int))


def _to_marks(series: pd.Series) -> pd.Series:
    nums = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(nums, errors="coerce").fillna(0.0)


def _clean_text(val) -> str:
    """Force any cell into a clean non-empty string for the embedder."""
    if val is None:
        return "n/a"
    if isinstance(val, float) and np.isnan(val):
        return "n/a"
    s = str(val).replace("\x00", " ").strip()
    if s == "" or s.lower() == "nan":
        return "n/a"
    return s


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = ["Subject", "Marks", "Concept", "Question", "Has_Code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["Marks"] = _to_marks(df["Marks"])
    df["Has_Code"] = _to_binary(df["Has_Code"])
    df = df.drop(columns=[c for c in ("Diagram",) if c in df.columns])

    for col in ("Concept", "Question", "Subject"):
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df[(df["Concept"] != "") & (df["Concept"].str.lower() != "nan")]
    df = df[(df["Question"] != "") & (df["Question"].str.lower() != "nan")]
    dropped = before - len(df)
    if dropped:
        print(f"[load] dropped {dropped} rows with blank Concept/Question")

    df = df.reset_index(drop=True)
    print(f"[load] {len(df)} usable questions across "
          f"{df['Subject'].nunique()} subjects, {df['Concept'].nunique()} concepts")
    return df


def add_repeat_count(df: pd.DataFrame) -> pd.DataFrame:
    df["repeat_count"] = (
        df.groupby(["Subject", "Concept"])["Concept"].transform("count").astype(float)
    )
    return df


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0.0


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["importance_score"] = (
        _norm(df["repeat_count"]) + _norm(df["Marks"]) + df["Has_Code"]
    )
    concept_mean = (
        df.groupby(["Subject", "Concept"])["importance_score"].mean().rename("concept_score")
    )
    thresholds = concept_mean.groupby(level="Subject").quantile(IMPORTANT_QUANTILE)
    concept_df = concept_mean.reset_index()
    concept_df["thr"] = concept_df["Subject"].map(thresholds)
    concept_df["label"] = (concept_df["concept_score"] >= concept_df["thr"]).astype(int)
    df = df.merge(concept_df[["Subject", "Concept", "label"]],
                  on=["Subject", "Concept"], how="left")
    # Drop any rows whose label didn't resolve in the merge (avoids NaN->int).
    n_before = len(df)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    if len(df) < n_before:
        print(f"[label] dropped {n_before - len(df)} rows with unresolved label")
    df = df.reset_index(drop=True)
    pos = int(df["label"].sum())
    n_imp = int(concept_df["label"].sum())
    print(f"[label] top-30% concepts marked important: {n_imp}/{len(concept_df)} "
          f"concepts -> {pos}/{len(df)} questions positive ({pos / len(df):.1%}); "
          f"question-level skew is expected (important concepts repeat more)")
    return df


def build_features(df: pd.DataFrame, embedder: SentenceTransformer):
    # Hardened: every item is a guaranteed clean python str.
    texts = [_clean_text(q) for q in df["Question"].tolist()]
    assert all(isinstance(t, str) and t != "" for t in texts), "non-str slipped through"
    print(f"[feat] embedding {len(texts)} questions with {EMBED_MODEL} ...")

    # Embed in explicit small batches so any single odd entry is isolated.
    chunks = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        part = texts[i:i + BATCH]
        vecs = embedder.encode(part, convert_to_numpy=True, show_progress_bar=False)
        chunks.append(np.asarray(vecs, dtype=np.float32))
        if (i // BATCH) % 10 == 0:
            print(f"   ...{min(i + BATCH, len(texts))}/{len(texts)}")
    emb = np.vstack(chunks)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    numeric = df[["Marks", "repeat_count", "Has_Code"]].to_numpy(dtype=float)
    X = np.hstack([emb, numeric])
    y = df["label"].to_numpy(dtype=int)
    print(f"[feat] X shape = {X.shape} (emb {emb.shape[1]} + numeric 3)")
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="questions.csv")
    ap.add_argument("--out", default="recommender_model.pkl")
    args = ap.parse_args()

    df = load_data(args.csv)
    df = add_repeat_count(df)
    df = build_labels(df)

    embedder = SentenceTransformer(EMBED_MODEL)
    X, y = build_features(df, embedder)

    if len(np.unique(y)) < 2:
        raise SystemExit("[error] only one class present — check data/threshold.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = neg / pos if pos else 1.0

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    print("[train] fitting XGBoost ...")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print("\n================ EVALUATION ================")
    print(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, digits=3))
    print("============================================\n")

    bundle = {
        "model": clf,
        "embed_model_name": EMBED_MODEL,
        "numeric_cols": ["Marks", "repeat_count", "Has_Code"],
        "important_quantile": IMPORTANT_QUANTILE,
    }
    joblib.dump(bundle, args.out)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
