import sys, os, warnings
sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

os.makedirs("data/NHNES", exist_ok=True)

print("\n[1/5] Loading raw NHANES CSV files...")
labs  = pd.read_csv("data/NHNES/labs.csv")
demo  = pd.read_csv("data/NHNES/demographic.csv")
exam  = pd.read_csv("data/NHNES/examination.csv")
quest = pd.read_csv("data/NHNES/questionnaire.csv")

print(f"      Labs  : {labs.shape}")
print(f"      Demo  : {demo.shape}")
print(f"      Exam  : {exam.shape}")
print(f"      Quest : {quest.shape}")

print("\n  [INFO] Glucose-related cols in labs :",
      [c for c in labs.columns if "GL" in c.upper() or "GLU" in c.upper()])
print("  [INFO] HbA1c-related cols in labs   :",
      [c for c in labs.columns if "GH" in c.upper() or "HBA" in c.upper()])
print("  [INFO] Insulin-related cols in labs :",
      [c for c in labs.columns if "IN" in c.upper() and "SEQN" not in c.upper()])
print("  [INFO] Diabetes cols in quest       :",
      [c for c in quest.columns if "DIQ" in c.upper()])
print("  [INFO] BP cols in exam              :",
      [c for c in exam.columns if "BPX" in c.upper()])
print("  [INFO] BMI cols in exam             :",
      [c for c in exam.columns if "BMX" in c.upper()])

# =========================================================
# STEP 2 — SELECT BEST AVAILABLE COLUMNS
# =========================================================
print("\n[2/5] Selecting best available columns...")

def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

glucose_col  = pick(labs, ["LBXGLU", "LBXSGL", "LBXGL", "LBXGLT", "ORXGL"])
hba1c_col    = pick(labs, ["LBXGH", "ORXGH", "LBDGH"])
insulin_col  = pick(labs, ["LBXIN", "LBDINSI"])
chol_col     = pick(labs, ["LBXTC", "LBDTCSI"])
hdl_col      = pick(labs, ["LBDHDD", "LBDHDL", "LBDHDDSI"])
trig_col     = pick(labs, ["LBXTR", "LBDTRSI"])
bp_col       = pick(exam, ["BPXSY1", "BPXSY2", "BPXSY3"])
dibp_col     = pick(exam, ["BPXDI1", "BPXDI2"])
bmi_col      = pick(exam, ["BMXBMI"])
waist_col    = pick(exam, ["BMXWAIST"])
age_col      = pick(demo, ["RIDAGEYR"])
gender_col   = pick(demo, ["RIAGENDR"])
diab_col     = pick(quest, ["DIQ010"])

print(f"  Glucose  → {glucose_col}")
print(f"  HbA1c    → {hba1c_col}")
print(f"  Insulin  → {insulin_col}")
print(f"  Chol     → {chol_col}")
print(f"  HDL      → {hdl_col}")
print(f"  Triglyc  → {trig_col}")
print(f"  BP Sys   → {bp_col}")
print(f"  BP Dia   → {dibp_col}")
print(f"  BMI      → {bmi_col}")
print(f"  Waist    → {waist_col}")
print(f"  Age      → {age_col}")
print(f"  Gender   → {gender_col}")
print(f"  Diabetes → {diab_col}")

# =========================================================
# STEP 3 — MERGE
# =========================================================
print("\n[3/5] Merging tables on SEQN...")
df = demo[["SEQN"] + [c for c in [age_col, gender_col] if c]].copy()

exam_cols = [c for c in [bmi_col, bp_col, dibp_col, waist_col] if c]
if exam_cols:
    df = df.merge(exam[["SEQN"] + exam_cols], on="SEQN", how="left")

lab_cols = [c for c in [glucose_col, insulin_col, hba1c_col,
                         chol_col, hdl_col, trig_col] if c]
if lab_cols:
    df = df.merge(labs[["SEQN"] + lab_cols], on="SEQN", how="left")

quest_cols = [c for c in [diab_col] if c]
if quest_cols:
    df = df.merge(quest[["SEQN"] + quest_cols], on="SEQN", how="left")

print(f"  Merged shape: {df.shape}")

# =========================================================
# STEP 4 — RENAME + BUILD OUTCOME
# =========================================================
print("\n[4/5] Renaming and building Outcome...")
rename_map = {}
if age_col:    rename_map[age_col]     = "Age"
if gender_col: rename_map[gender_col]  = "Gender"
if bmi_col:    rename_map[bmi_col]     = "BMI"
if bp_col:     rename_map[bp_col]      = "BloodPressure"
if glucose_col:rename_map[glucose_col] = "Glucose"
if insulin_col:rename_map[insulin_col] = "Insulin"
if hba1c_col:  rename_map[hba1c_col]   = "HbA1c"
if chol_col:   rename_map[chol_col]    = "TotalCholesterol"
if hdl_col:    rename_map[hdl_col]     = "HDL"
if trig_col:   rename_map[trig_col]    = "Triglycerides"
if diab_col:   rename_map[diab_col]    = "_diab_raw"
df = df.rename(columns=rename_map)

# ── Outcome from DIQ010 ──
if "_diab_raw" in df.columns:
    df["Outcome"] = df["_diab_raw"].map({1: 1, 2: 0})
else:
    df["Outcome"] = np.nan

# ── HbA1c fallback ──
if "HbA1c" in df.columns:
    hba1c_num = pd.to_numeric(df["HbA1c"], errors="coerce")
    mask = df["Outcome"].isna() & hba1c_num.notna()
    df.loc[mask, "Outcome"] = (hba1c_num[mask] >= 6.5).astype(float)

# ── Glucose fallback ──
if "Glucose" in df.columns:
    gluc_num = pd.to_numeric(df["Glucose"], errors="coerce")
    mask = df["Outcome"].isna() & gluc_num.notna()
    df.loc[mask, "Outcome"] = (gluc_num[mask] >= 126).astype(float)

df["Outcome"] = df["Outcome"].fillna(0).astype(int)

# ── Add Pima-compatible columns ──
df["Pregnancies"]              = 0
df["SkinThickness"]            = 0
# Map HbA1c → DiabetesPedigreeFunction slot
# (NHANES has no DPF; HbA1c is far more informative)
if "HbA1c" in df.columns:
    df["DiabetesPedigreeFunction"] = pd.to_numeric(df["HbA1c"], errors="coerce")
else:
    df["DiabetesPedigreeFunction"] = 0.0

if "Glucose" not in df.columns:
    df["Glucose"] = np.nan
if "Insulin" not in df.columns:
    df["Insulin"] = 0.0
if "HbA1c" not in df.columns:
    df["HbA1c"] = 0.0

# ── Convert to numeric ──
FINAL_COLS = ["Age", "Gender", "BMI", "BloodPressure", "Glucose",
              "Insulin", "HbA1c", "DiabetesPedigreeFunction",
              "Pregnancies", "SkinThickness", "Outcome"]
for c in FINAL_COLS:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ── Unit conversion: mmol/L → mg/dL ──
if "Glucose" in df.columns:
    valid = df["Glucose"].dropna()
    if len(valid) > 0 and valid.median() < 30:
        print("  [INFO] Glucose appears mmol/L — converting ×18 to mg/dL")
        df["Glucose"] = df["Glucose"] * 18.0

# ── Filter valid rows ──
df = df.dropna(subset=["BMI", "Age"])
df = df[df["BMI"] > 0]
df = df[df["Age"] >= 18]

has_glucose = df["Glucose"].notna() & (df["Glucose"] > 0)
has_hba1c   = df["HbA1c"].notna()   & (df["HbA1c"]   > 0)
df = df[has_glucose | has_hba1c].copy()

# ── Fill NaN with median ──
for c in FINAL_COLS:
    if df[c].isnull().any():
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0)

df = df[FINAL_COLS].reset_index(drop=True)

# =========================================================
# STEP 5 — SAVE
# =========================================================
print(f"\n[5/5] Saving rebuilt NHANES...")
print(f"  Final shape           : {df.shape}")
print(f"  Diabetic prevalence   : {df['Outcome'].mean():.1%}")
print(f"  Glucose range (mg/dL) : {df['Glucose'].min():.1f} – {df['Glucose'].max():.1f}")
print(f"  Glucose median        : {df['Glucose'].median():.1f}")
print(f"  HbA1c > 0 rows        : {(df['HbA1c'] > 0).sum()}")
print(f"  DPF (HbA1c) median    : {df['DiabetesPedigreeFunction'].median():.2f}")
print(f"  Insulin > 0 rows      : {(df['Insulin'] > 0).sum()}")
print(f"  Outcome distribution  : {df['Outcome'].value_counts().to_dict()}")

df.to_csv("data/NHNES/nhanes_diabetes_processed.csv", index=False)
print("  Saved: data/NHNES/nhanes_diabetes_processed.csv")
print("\n" + "="*55)
print("  Done. Next: python retrain_combined.py")
print("="*55)
