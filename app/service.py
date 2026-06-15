"""
SmartQBank — Marks-target question recommender — FASTAPI SERVICE
================================================================
Reads recommender_model.pkl + the exported CSV, scores every question's
importance, and serves greedy topic recommendations toward a target mark.

Each recommended question is now an object:
    {"question": "...", "has_code": true, "code_snippet": "..."}
so questions that have code carry their Code_Snippet along.

Run (Command Prompt):
    set MODEL_PATH=recommender_model.pkl
    set CSV_PATH=smartqbank_dataset - Sheet1.csv.csv
    uvicorn app.service:app --port 8000
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.getenv("MODEL_PATH", "recommender_model.pkl")
CSV_PATH = os.getenv("CSV_PATH", "smartqbank_dataset.csv")
MAX_QUESTIONS_PER_CONCEPT = int(os.getenv("MAX_Q_PER_CONCEPT", "3"))

app = FastAPI(title="SmartQBank Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STATE: dict = {}


def _to_binary(series: pd.Series) -> pd.Series:
    truthy = {"yes", "y", "1", "true", "t"}
    return (
        series.fillna("").astype(str).str.strip().str.lower().isin(truthy).astype(int)
    )


def _to_marks(series: pd.Series) -> pd.Series:
    nums = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
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
    """Code snippet: return clean string, or '' if there is none."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).replace("\x00", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s


def _prepare_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Marks"] = _to_marks(df["Marks"])
    df["Has_Code"] = _to_binary(df["Has_Code"])
    df = df.drop(columns=[c for c in ("Diagram",) if c in df.columns])

    # keep Code_Snippet as clean text (may be empty)
    if "Code_Snippet" in df.columns:
        df["Code_Snippet"] = df["Code_Snippet"].map(_clean_code)
    else:
        df["Code_Snippet"] = ""

    for col in ("Concept", "Question", "Subject"):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["Concept"] != "") & (df["Concept"].str.lower() != "nan")]
    df = df[(df["Question"] != "") & (df["Question"].str.lower() != "nan")]
    df["Subject"] = df["Subject"].astype(str)
    df = df[(df["Subject"].str.strip() != "") & (df["Subject"].str.lower() != "nan")]
    df["repeat_count"] = (
        df.groupby(["Subject", "Concept"])["Concept"].transform("count").astype(float)
    )
    return df.reset_index(drop=True)


def _embed(embedder, questions):
    texts = [_clean_text(q) for q in questions]
    chunks = []
    for i in range(0, len(texts), 32):
        vecs = embedder.encode(
            texts[i : i + 32], convert_to_numpy=True, show_progress_bar=False
        )
        chunks.append(np.asarray(vecs, dtype=np.float32))
    emb = np.vstack(chunks)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


@app.on_event("startup")
def _load():
    bundle = joblib.load(MODEL_PATH)
    embedder = SentenceTransformer(bundle["embed_model_name"])
    df = _prepare_df(CSV_PATH)

    print(f"[startup] embedding {len(df)} questions ...")
    emb = _embed(embedder, df["Question"].tolist())
    numeric = df[bundle["numeric_cols"]].to_numpy(dtype=float)
    X = np.hstack([emb, numeric])
    df["importance"] = bundle["model"].predict_proba(X)[:, 1]

    STATE["df"] = df
    STATE["subjects"] = sorted(set(str(x) for x in df["Subject"].tolist()))
    print(f"[startup] scored {len(df)} questions; subjects: {STATE['subjects']}")


@app.get("/health")
def health():
    return {"status": "ok", "questions_loaded": len(STATE.get("df", []))}


@app.get("/subjects")
def subjects():
    return {"subjects": STATE.get("subjects", [])}


@app.get("/recommend")
def recommend(
    subject: str = Query(..., description="exact subject name"),
    target: float = Query(..., gt=0, description="target marks, e.g. 40"),
):
    df = STATE.get("df")
    if df is None:
        raise HTTPException(503, "model not loaded yet")

    sub = df[df["Subject"].str.lower() == subject.strip().lower()]
    if sub.empty:
        raise HTTPException(
            404, f"subject '{subject}' not found. Available: {STATE['subjects']}"
        )

    topics = []
    for concept, g in sub.groupby("Concept"):
        importance = float(g["importance"].mean())
        avg_marks = float(g["Marks"].mean())
        expected = avg_marks * importance
        top = g.sort_values("importance", ascending=False).head(
            MAX_QUESTIONS_PER_CONCEPT
        )

        questions = []
        for _, row in top.iterrows():
            code = _clean_code(row.get("Code_Snippet", ""))
            has_code = bool(int(row.get("Has_Code", 0))) or (code != "")
            questions.append(
                {
                    "question": str(row["Question"]),
                    "has_code": has_code,
                    "code_snippet": code if code != "" else None,
                }
            )

        topics.append(
            {
                "concept": concept,
                "importance": round(importance, 4),
                "expected_marks": round(expected, 2),
                "has_code": bool(g["Has_Code"].max()),
                "questions": questions,
            }
        )

    topics.sort(key=lambda t: t["importance"], reverse=True)

    chosen, projected = [], 0.0
    for t in topics:
        if projected >= target:
            break
        chosen.append(t)
        projected += t["expected_marks"]

    return {
        "subject": sub["Subject"].iloc[0],
        "target": target,
        "projected_marks": round(projected, 2),
        "target_met": projected >= target,
        "topics": chosen,
    }
