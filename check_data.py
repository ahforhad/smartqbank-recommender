# Data health check for SmartQBank dataset.
# Run:  python check_data.py
import pandas as pd

df = pd.read_csv("smartqbank_dataset.csv")
df.columns = [c.strip() for c in df.columns]

print("=" * 50)
print("ROWS:", len(df))
print("COLUMNS:", list(df.columns))
print("=" * 50)

# --- Subjects ---
print("\n[SUBJECTS]")
print(df["Subject"].astype(str).str.strip().value_counts(dropna=False))

# --- Difficulty ---
print("\n[DIFFICULTY] (raw values, to catch typos/spaces)")
print(df["Difficulty"].value_counts(dropna=False))
print("\n[DIFFICULTY] normalized (lower/stripped)")
print(df["Difficulty"].astype(str).str.strip().str.lower().value_counts(dropna=False))

# --- Has_Code ---
print("\n[HAS_CODE] raw values")
print(df["Has_Code"].value_counts(dropna=False))

# --- Marks issues ---
print("\n[MARKS] non-numeric / junk values")
nums = pd.to_numeric(
    df["Marks"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce",
)
junk = df.loc[nums.isna(), "Marks"].value_counts(dropna=False)
print(junk if len(junk) else "  none")
print("blank/NaN marks rows:", int(nums.isna().sum()))

# --- Blank key fields ---
print("\n[BLANKS in key fields]")
for col in ["Subject", "Concept", "Question", "Difficulty"]:
    s = df[col].astype(str).str.strip()
    blank = ((s == "") | (s.str.lower() == "nan")).sum()
    print(f"  {col}: {blank} blank")

# --- Difficulty spread per subject (so no subject is empty for a level) ---
print("\n[DIFFICULTY x SUBJECT] (do all subjects have all 3 levels?)")
piv = (
    df.assign(
        d=df["Difficulty"].astype(str).str.strip().str.title(),
        subj=df["Subject"].astype(str).str.strip(),
    )
    .pivot_table(index="subj", columns="d", aggfunc="size", fill_value=0)
)
print(piv)

# --- Concepts with very few questions (thin for filtering) ---
print("\n[CONCEPTS] total unique:", df["Concept"].astype(str).str.strip().nunique())
print("=" * 50)
print("CHECK COMPLETE")
