"""
SmartQBank — Marks-target recommender — LIGHTWEIGHT FASTAPI SERVICE
==================================================================
Reads recommender_model.pkl (precomputed importance + question rows incl.
Difficulty). No sentence-transformers / torch needed.

NEW: optional `difficulty` filter on /recommend.
  - difficulty=Any (or omitted) -> use all questions (current behavior)
  - difficulty=Easy|Medium|Hard -> rebuild the WHOLE plan from only that
    difficulty's questions. Unlabeled questions are excluded under a specific
    difficulty. Concepts with no questions at that difficulty drop out.

Run locally:
    set MODEL_PATH=recommender_model.pkl
    uvicorn app.service:app --port 8000
"""

import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "recommender_model.pkl")
MAX_QUESTIONS_PER_CONCEPT = int(os.getenv("MAX_Q_PER_CONCEPT", "3"))
# Topics contributing fewer than this many expected marks are considered
# "negligible" and are not padded onto a plan. Keeps the list meaningful and
# avoids a long tail of ~0-mark "Optional" topics when a target can't be met.
MIN_TOPIC_CONTRIB = float(os.getenv("MIN_TOPIC_CONTRIB", "0.5"))

app = FastAPI(title="SmartQBank Recommender (lightweight)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STATE: dict = {}
VALID_DIFF = {"easy", "medium", "hard"}


def _clean_code(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s


def _norm_diff(val) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    return s.title() if s in VALID_DIFF else ""


def _load_bundle():
    bundle = joblib.load(MODEL_PATH)
    df = pd.DataFrame(bundle["questions"])
    df["Subject"] = df["Subject"].astype(str)
    df["Concept"] = df["Concept"].astype(str)
    df["Question"] = df["Question"].astype(str)
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce").fillna(0.0)
    df["Has_Code"] = pd.to_numeric(df["Has_Code"], errors="coerce").fillna(0).astype(int)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    if "Code_Snippet" not in df.columns:
        df["Code_Snippet"] = ""
    if "Difficulty" not in df.columns:
        df["Difficulty"] = ""
    df["Difficulty"] = df["Difficulty"].map(_norm_diff)
    STATE["df"] = df
    STATE["subjects"] = bundle.get(
        "subjects", sorted(df["Subject"].unique().tolist())
    )
    STATE["difficulties"] = bundle.get("difficulties", ["Easy", "Medium", "Hard"])
    print(f"[startup] loaded {len(df)} precomputed questions; "
          f"subjects: {STATE['subjects']}; difficulties: {STATE['difficulties']}")


@app.on_event("startup")
def _startup():
    _load_bundle()


@app.get("/health")
def health():
    return {"status": "ok", "questions_loaded": len(STATE.get("df", []))}


@app.get("/subjects")
def subjects():
    return {"subjects": STATE.get("subjects", [])}


@app.get("/difficulties")
def difficulties():
    return {"difficulties": STATE.get("difficulties", ["Easy", "Medium", "Hard"])}


@app.get("/reload")
def reload_data():
    _load_bundle()
    return {"status": "reloaded", "questions_loaded": len(STATE["df"])}


@app.get("/recommend")
def recommend(
    subject: str = Query(..., description="exact subject name"),
    target: float = Query(..., gt=0, description="target marks, e.g. 40"),
    difficulty: str = Query("Any", description="Any | Easy | Medium | Hard"),
):
    df = STATE.get("df")
    if df is None:
        raise HTTPException(503, "model not loaded yet")

    sub = df[df["Subject"].str.lower() == subject.strip().lower()]
    if sub.empty:
        raise HTTPException(
            404, f"subject '{subject}' not found. Available: {STATE['subjects']}"
        )

    # apply difficulty filter -> rebuild whole plan from that pool
    diff = (difficulty or "Any").strip().lower()
    if diff in VALID_DIFF:
        pool = sub[sub["Difficulty"].str.lower() == diff]
        diff_label = diff.title()
    else:
        pool = sub  # Any (or unknown) -> all questions
        diff_label = "Any"

    if pool.empty:
        # No questions at this difficulty for this subject at all.
        return {
            "subject": sub["Subject"].iloc[0],
            "target": target,
            "difficulty": diff_label,
            "projected_marks": 0.0,
            "target_met": False,
            "topics": [],
        }

    topics = []
    for concept, g in pool.groupby("Concept"):
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
                "difficulty": _norm_diff(row.get("Difficulty", "")) or None,
            })

        topics.append({
            "concept": concept,
            "importance": round(importance, 4),
            "expected_marks": round(expected, 2),
            "has_code": bool(g["Has_Code"].max()),
            "questions": questions,
        })

    topics.sort(key=lambda t: t["importance"], reverse=True)

    # Greedy pick until target reached, but never pad with negligible topics.
    chosen, projected = [], 0.0
    for t in topics:
        if projected >= target:
            break
        # Stop once topics stop meaningfully contributing (avoids ~0-mark tail).
        if t["expected_marks"] < MIN_TOPIC_CONTRIB:
            break
        chosen.append(t)
        projected += t["expected_marks"]

    # max reachable = sum of all meaningful topics at this difficulty
    max_reachable = round(
        sum(t["expected_marks"] for t in topics
            if t["expected_marks"] >= MIN_TOPIC_CONTRIB),
        2,
    )

    target_met = projected >= target

    return {
        "subject": sub["Subject"].iloc[0],
        "target": target,
        "difficulty": diff_label,
        "projected_marks": round(projected, 2),
        "target_met": target_met,
        "max_reachable": max_reachable,
        "topics": chosen,
    }
