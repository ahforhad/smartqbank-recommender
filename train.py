"""
SmartQBank — Marks-target recommender — TRAINING SCRIPT (deploy-ready)
=====================================================================
Bundles everything the lightweight server needs into recommender_model.pkl:
  - the trained XGBoost model is used only to precompute importance
  - precomputed importance score for every question
  - the question rows (concept, marks, has_code, code_snippet, DIFFICULTY)

The deployed server reads the precomputed scores (no sentence-transformers /
torch needed), so it stays small enough for free hosting.

Run:
    python train.py --csv smartqbank_dataset.csv
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

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _to_binary(series: pd.Series) -> pd.Series:
    truthy = {"yes", "y", "1", "true", "t"}
    return (series.fillna("").astype(str).str.strip().str.lower()
            .isin(truthy).astype(int))


def _to_marks(series: pd.Series) -> pd.Series:
    nums = series.astype(str).str.replace(",", ".", regex=False)
    nums = nums.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(nums, errors="coerce").fillna(0.0)


def _clean_text(val) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float) and np.isnan(val):
        return "n/a"
    s = str(val).replace("\x00", " ").strip()
    if s == "" or s.lower() == "nan":
        return "n/a"
    return s


def _clean_code(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).replace("\x00", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s


def _clean_difficulty(val) -> str:
    """Normalize to Easy/Medium/Hard; anything else -> '' (unlabeled)."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in VALID_DIFFICULTIES:
        return s.title()  # Easy / Medium / Hard
    return ""


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

    if "Code_Snippet" in df.columns:
        df["Code_Snippet"] = df["Code_Snippet"].map(_clean_code)
    else:
        df["Code_Snippet"] = ""

    if "Difficulty" in df.columns:
        df["Difficulty"] = df["Difficulty"].map(_clean_difficulty)
    else:
        df["Difficulty"] = ""

    for col in ("Concept", "Question", "Subject"):
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df[(df["Concept"] != "") & (df["Concept"].str.lower() != "nan")]
    df = df[(df["Question"] != "") & (df["Question"].str.lower() != "nan")]
    df = df[(df["Subject"] != "") & (df["Subject"].str.lower() != "nan")]
    dropped = before - len(df)
    if dropped:
        print(f"[load] dropped {dropped} rows with blank Concept/Question/Subject")

    df = df.reset_index(drop=True)
    diff_counts = df["Difficulty"].replace("", "(unlabeled)").value_counts().to_dict()
    print(f"[load] {len(df)} usable questions across "
          f"{df['Subject'].nunique()} subjects, {df['Concept'].nunique()} concepts")
    print(f"[load] difficulty spread: {diff_counts}")
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
    n_before = len(df)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    if len(df) < n_before:
        print(f"[label] dropped {n_before - len(df)} rows with unresolved label")
    df = df.reset_index(drop=True)
    pos = int(df["label"].sum())
    n_imp = int(concept_df["label"].sum())
    print(f"[label] top-30% concepts important: {n_imp}/{len(concept_df)} concepts "
          f"-> {pos}/{len(df)} questions positive ({pos/len(df):.1%})")
    return df


def build_embeddings(df: pd.DataFrame, embedder):
    texts = [_clean_text(q) for q in df["Question"].tolist()]
    print(f"[feat] embedding {len(texts)} questions with {EMBED_MODEL} ...")
    chunks = []
    for i in range(0, len(texts), 32):
        vecs = embedder.encode(texts[i:i + 32], convert_to_numpy=True,
                               show_progress_bar=False)
        chunks.append(np.asarray(vecs, dtype=np.float32))
        if (i // 32) % 10 == 0:
            print(f"   ...{min(i + 32, len(texts))}/{len(texts)}")
    emb = np.vstack(chunks)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="smartqbank_dataset.csv")
    ap.add_argument("--out", default="recommender_model.pkl")
    args = ap.parse_args()

    df = load_data(args.csv)
    df = add_repeat_count(df)
    df = build_labels(df)

    embedder = SentenceTransformer(EMBED_MODEL)
    emb = build_embeddings(df, embedder)
    numeric = df[["Marks", "repeat_count", "Has_Code"]].to_numpy(dtype=float)
    X = np.hstack([emb, numeric])
    y = df["label"].to_numpy(dtype=int)
    print(f"[feat] X shape = {X.shape} (emb {emb.shape[1]} + numeric 3)")

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

    importance_all = clf.predict_proba(X)[:, 1]

    questions_table = df[["Subject", "Concept", "Question", "Marks",
                          "repeat_count", "Has_Code", "Code_Snippet",
                          "Difficulty"]].copy()
    questions_table["importance"] = importance_all

    bundle = {
        "questions": questions_table.to_dict(orient="records"),
        "subjects": sorted(df["Subject"].astype(str).unique().tolist()),
        "difficulties": ["Easy", "Medium", "Hard"],
    }
    joblib.dump(bundle, args.out)
    print(f"[save] wrote {args.out} "
          f"({len(bundle['questions'])} scored questions w/ difficulty, "
          f"ready for free hosting)")


if __name__ == "__main__":
    main()
