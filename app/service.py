"""
SmartQBank — Marks-target recommender — LIGHTWEIGHT FASTAPI SERVICE
==================================================================
Reads recommender_model.pkl which now contains PRECOMPUTED importance scores
for every question (produced by train.py). This server does NOT load
sentence-transformers or torch, so it stays small enough for free hosting.

Run locally:
    set MODEL_PATH=recommender_model.pkl
    uvicorn app.service:app --port 8000

Endpoints:
    GET /recommend?subject=Data%20Structures&target=40
    GET /subjects
    GET /health
"""

import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "recommender_model.pkl")
MAX_QUESTIONS_PER_CONCEPT = int(os.getenv("MAX_Q_PER_CONCEPT", "3"))

app = FastAPI(title="SmartQBank Recommender (lightweight)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STATE: dict = {}


def _clean_code(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s


def _load_bundle():
    bundle = joblib.load(MODEL_PATH)
    df = pd.DataFrame(bundle["questions"])
    # safety: make sure columns are the right types
    df["Subject"] = df["Subject"].astype(str)
    df["Concept"] = df["Concept"].astype(str)
    df["Question"] = df["Question"].astype(str)
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce").fillna(0.0)
    df["Has_Code"] = pd.to_numeric(df["Has_Code"], errors="coerce").fillna(0).astype(int)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    if "Code_Snippet" not in df.columns:
        df["Code_Snippet"] = ""
    STATE["df"] = df
    STATE["subjects"] = bundle.get(
        "subjects", sorted(df["Subject"].unique().tolist())
    )
    print(f"[startup] loaded {len(df)} precomputed questions; "
          f"subjects: {STATE['subjects']}")


@app.on_event("startup")
def _startup():
    _load_bundle()


@app.get("/health")
def health():
    return {"status": "ok", "questions_loaded": len(STATE.get("df", []))}


@app.get("/subjects")
def subjects():
    return {"subjects": STATE.get("subjects", [])}


@app.get("/reload")
def reload_data():
    """Reload recommender_model.pkl without restarting the server."""
    _load_bundle()
    return {"status": "reloaded", "questions_loaded": len(STATE["df"])}


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
        top = g.sort_values("importance", ascending=False).head(MAX_QUESTIONS_PER_CONCEPT)

        questions = []
        for _, row in top.iterrows():
            code = _clean_code(row.get("Code_Snippet", ""))
            has_code = bool(int(row.get("Has_Code", 0))) or (code != "")
            questions.append({
                "question": str(row["Question"]),
                "has_code": has_code,
                "code_snippet": code if code != "" else None,
            })

        topics.append({
            "concept": concept,
            "importance": round(importance, 4),
            "expected_marks": round(expected, 2),
            "has_code": bool(g["Has_Code"].max()),
            "questions": questions,
        })

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
