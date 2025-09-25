# train_startup_simple.py
# Minimal, robust model + skops save

import re, json, sys
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import skops.io as skio
import sklearn

print("TRAIN python:", sys.executable)
print("TRAIN sklearn:", sklearn.__version__)

CSV = "startup data.csv"      # adjust path if needed
OUT = Path("startup_model"); OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)

# Target: success = status != 'closed'
y = (df["status"].astype(str).str.lower().str.strip() != "closed").astype(int)

# --- Whitelist a small, stable set of features ---
keep = [
    # basics
    "state_code", "category_code",
    # funding signals
    "has_angel", "has_VC", "has_roundA", "has_roundB", "funding_rounds",
    # milestones & timing (coarsened later)
    "milestones", "age_first_funding_year",
    # network & visibility
    "relationships", "avg_participants", "is_top500",
]
X = df[[c for c in keep if c in df.columns]].copy()

# Coarsen a few continuous vars into small buckets (robust for a tiny demo)
def bucketize(s, bins):
    return pd.cut(s, bins=bins, include_lowest=True).astype(str)

if "funding_rounds" in X:
    X["funding_rounds_b"] = bucketize(X["funding_rounds"], bins=[-1,0,1,3,10])
if "milestones" in X:
    X["milestones_b"] = bucketize(X["milestones"], bins=[-1,0,1,3,10])
if "age_first_funding_year" in X:
    # lower = earlier funding
    X["age_first_funding_b"] = bucketize(X["age_first_funding_year"], bins=[-1,0,1,3,30])
if "relationships" in X:
    X["relationships_b"] = bucketize(X["relationships"], bins=[-1,0,2,5,20])
if "avg_participants" in X:
    X["avg_participants_b"] = bucketize(X["avg_participants"], bins=[-1,0,2,5,20])

# Final feature set (categoricals only → simple OneHot; ints treated as cats)
feat = []
for c in ["state_code","category_code","has_angel","has_VC","has_roundA","has_roundB","is_top500",
          "funding_rounds_b","milestones_b","age_first_funding_b","relationships_b","avg_participants_b"]:
    if c in X.columns: feat.append(c)
X = X[feat]

# Train/holdout
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess: impute + one-hot (dense)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

pre = ColumnTransformer([("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                                 ("ohe", ohe)]), feat)])

pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
pipe.fit(Xtr, ytr)

proba = pipe.predict_proba(Xte)[:,1]
pred = (proba >= 0.5).astype(int)
holdout = {
    "roc_auc": float(roc_auc_score(yte, proba)),
    "accuracy": float(accuracy_score(yte, pred)),
    "f1": float(f1_score(yte, pred)),
}
print("Holdout:", holdout)

# Save model (skops) + a tiny UI schema
skio.dump(pipe, OUT/"startup_success_model.skops")

schema = {
  "model_name": "logreg_simple",
  "holdout": holdout,
  # UI option sets (short, positive phrasing)
  "ui": {
    "business_types": ["Tech / Software", "Web / Mobile", "E-commerce", "Biotech", "Consulting", "Other"],
    "locations": ["Silicon Valley (CA)", "New York (NY)", "Boston (MA)", "Texas (TX)", "Other"],
    "funding_options": ["Angel", "VC", "Series A", "Series B", "Bootstrapped"],
    "strategy": ["🚀 Big swing early", "💰 Funding first"],
    "network": [
      ("Independent operators", "low"),   # weak network → framed as autonomy
      ("Balanced connectors", "mid"),
      ("Well-connected insiders", "high")
    ],
    "visibility": [
      ("Under the radar", "low"),
      ("Top 500 buzz", "high")
    ],
  }
}
Path(OUT/"feature_schema.json").write_text(json.dumps(schema, indent=2))
print("Saved:", OUT/"startup_success_model.skops", "and feature_schema.json")

