#!/usr/bin/env python3
# spotify_app.py — Simple, educational 3-class demo with Small/Medium/Large per model.
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json, os
import numpy as np
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(page_title="Model Spotify Popularity", page_icon="🎧", layout="centered")

APP_DIR = Path(__file__).parent
ARTIFACT_DIR = Path(os.environ.get("SPOTIFY_ARTIFACT_DIR", APP_DIR / "artifacts" / "spotify_v2")).resolve()
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

ASSETS_DIR = APP_DIR / "assets"
FAMILY_DIR = ASSETS_DIR / "family"
PRESET_DIR = ASSETS_DIR / "preset"

CLASS_NAMES = ["Low", "Medium", "High"]

# Optional explicit map for the family icons and generic size art
IMAGE_MAP = {
    "family": {
        "Logistic regression": str(FAMILY_DIR / "logreg.png"),
        "Decision tree":       str(FAMILY_DIR / "tree.png"),
        "Neural network":      str(FAMILY_DIR / "nn.png"),
    },
    "preset": {
        "Small":  str(PRESET_DIR / "small.png"),
        "Medium": str(PRESET_DIR / "medium.png"),
        "Large":  str(PRESET_DIR / "large.png"),
    }
}

# -------------------- Utilities --------------------
def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        st.error(f"Manifest not found at {MANIFEST_PATH}. Run the trainer to create it.")
        st.stop()
    return json.loads(MANIFEST_PATH.read_text())

def _img_for(group: str, label: str, family: Optional[str] = None) -> Optional[Path]:
    """
    Prefer family-specific size art like preset/logreg_small.png,
    then fall back to generic small.png / medium.png / large.png,
    then try a literal filename match (e.g., 'logreg.png').
    """
    # 1) Family-specific for size cards
    if group == "preset" and family:
        base = PRESET_DIR / f"{family}_{label.lower()}"
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = base.with_suffix(ext)
            if p.exists():
                return p

    # 2) Mapped generic
    mapped = IMAGE_MAP.get(group, {}).get(label)
    if mapped and Path(mapped).exists():
        return Path(mapped)

    # 3) Literal filename (fallback)
    base = FAMILY_DIR if group == "family" else PRESET_DIR
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = base / f"{label.lower()}{ext}"
        if p.exists():
            return p
    return None

def draw_card(group: str, label: str, sub: Optional[str], key: str, family: Optional[str] = None, emoji="🎛️") -> bool:
    img = _img_for(group, label, family=family)
    with st.container():
        if img:
            st.image(str(img), use_container_width=True)
        else:
            st.markdown(f"<div style='font-size:64px;text-align:center'>{emoji}</div>", unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        if sub:
            st.caption(sub)
        return st.button("Choose", key=key, use_container_width=True)

def result_fx_once(ok: bool) -> None:
    if st.session_state.get("fx_done", False):
        return
    try:
        _ = st.balloons() if ok else st.snow()
    except Exception:
        pass
    st.session_state["fx_done"] = True

def friendly_model_name(fam: str, size: str) -> str:
    if fam == "logreg":
        return {"Small": "Logistic — Top 10 features (with interactions)",
                "Medium": "Logistic — L1 selects best subset",
                "Large": "Logistic — All features + interactions"}[size]
    if fam == "tree":
        return {"Small": "Decision Tree — Heavy pruning",
                "Medium": "Decision Tree — CV-pruned (optimal α)",
                "Large": "Decision Tree — Unpruned"}[size]
    if fam == "mlp":
        return {"Small": "Neural Net — Shallow",
                "Medium": "Neural Net — Best (tuned)",
                "Large": "Neural Net — Deep"}[size]
    return f"{fam} — {size}"

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1100px; padding-top: 3.0rem !important; padding-bottom: 2rem;
      }
      .hero-h1 { font-size: 46px; font-weight: 800; line-height: 1.08; margin: 0 0 10px; }
      .hero-sub { font-size: 18px; opacity: 0.9; margin-bottom: 16px; }
      .edu-box { background: #0b13241a; border: 1px solid #0b132433; border-radius: 12px; padding: 14px 16px; }
      .smallnote { font-size: 12px; opacity: 0.7; }
      .tight ul { margin-top: 0.25rem; }
      .tight li { margin-bottom: 0.35rem; }
      .lined { border-top: 1px dashed #ddd; margin: 10px 0 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- App state --------------------
steps = ["landing", "data", "family", "size", "result"]
if "step" not in st.session_state: st.session_state.step = 0
st.session_state.setdefault("family", None)  # "logreg" | "tree" | "mlp"
st.session_state.setdefault("size", None)    # "Small" | "Medium" | "Large"

def goto(name: str) -> None:
    st.session_state.step = steps.index(name)

# -------------------- Panes --------------------
def pane_landing():
    col1, col2 = st.columns([1.15, 2])
    with col1:
        st.markdown("<div style='font-size:140px; line-height:1; text-align:center'>🎧</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            "<div class='hero-h1'>Data Scientists Build Models to Determine Which Factors Drive Predictions</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-sub'>We split tracks into <b>High / Medium / Low</b> popularity and train different "
            "models to predict the category.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Explore the data →", use_container_width=True):
            goto("data"); st.experimental_rerun()

def pane_data(manifest: Dict[str, Any]):
    if "pop3" not in manifest.get("tasks", {}):
        st.error("Manifest missing task 'pop3'. Please run the trainer.")
        st.stop()
    info = manifest["tasks"]["pop3"]["dataset_info"]
    rows = info.get("rows", {})
    nums = info.get("numeric_features", [])
    cats = info.get("categorical_features", [])

    st.subheader("The data we’ll model")
    st.markdown(
        f"- **Goal**: Predict a track’s popularity bucket — **Low / Medium / High** — from audio & metadata.\n"
        f"- **Total tracks**: {rows.get('total','?')} (train: {rows.get('train','?')}, "
        f"val: {rows.get('val','?')}, test: {rows.get('test','?')})\n"
    )
    with st.expander("What are the inputs?", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numeric (examples):**")
            st.markdown(
                "- danceability, energy, loudness\n"
                "- speechiness, acousticness, instrumentalness\n"
                "- liveness, valence, tempo, duration_ms"
            )
        with c2:
            st.markdown("**Categorical (examples):**")
            st.markdown(
                "- explicit flag (yes/no)\n"
                "- musical key, mode\n"
                "- time signature\n"
                "- track_genre"
            )
        st.caption(
            "We also create squared terms and interactions for numeric features to capture non-linear relationships."
        )

    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("⬅️ Back"):
        goto("landing"); st.experimental_rerun()
    if c3.button("Pick a model →"):
        goto("family"); st.experimental_rerun()

def pane_family():
    st.subheader("Pick a model family")
    left, right = st.columns([1.1, 1])

    # LEFT: stacked option cards
    with left:
        # Logistic
        if draw_card("family", "Logistic regression", "Probability model", key="fam_logreg"):
            st.session_state.family = "logreg"; st.session_state.size = None
            st.session_state["fx_done"] = False
            goto("size"); st.experimental_rerun()
        # Tree
        if draw_card("family", "Decision tree", "If–then rules", key="fam_tree"):
            st.session_state.family = "tree"; st.session_state.size = None
            st.session_state["fx_done"] = False
            goto("size"); st.experimental_rerun()
        # NN
        if draw_card("family", "Neural network", "Learns patterns", key="fam_mlp"):
            st.session_state.family = "mlp"; st.session_state.size = None
            st.session_state["fx_done"] = False
            goto("size"); st.experimental_rerun()

    # RIGHT: compact explainer table
    with right:
        st.markdown("**What’s the difference?**")
        st.markdown(
            """
| Model | What it is | Where it shines | Where it can struggle |
|---|---|---|---|
| **Logistic regression** | Uses weighted inputs to estimate class probabilities. | Clear, explainable, fast. | Misses complex, non-linear patterns. |
| **Decision tree** | A flow of IF–THEN rules that split the data. | Intuitive, handles mixed data well. | Can overfit if grown too deep. |
| **Neural network** | Layers of simple units that learn patterns. | Captures complex relationships. | Needs tuning; may overfit if too large. |
            """,
            unsafe_allow_html=True,
        )
        st.caption("Pick one to choose a model size next (Small / Medium / Large).")

    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("⬅️ Data"):
        goto("data"); st.experimental_rerun()

def pane_size(manifest: Dict[str, Any]):
    fam = st.session_state.get("family")
    if fam not in ("logreg", "tree", "mlp"):
        goto("family"); st.experimental_rerun(); return

    st.subheader("Choose model complexity")
    left, right = st.columns([1.1, 1])

    # LEFT: Small / Medium / Large cards (family-specific art if present)
    with left:
        if draw_card("preset", "Small", _size_subtitle(fam, "Small"), key="size_small", family=fam):
            st.session_state.size = "Small"; st.session_state["fx_done"] = False
            goto("result"); st.experimental_rerun()
        if draw_card("preset", "Medium", _size_subtitle(fam, "Medium"), key="size_medium", family=fam):
            st.session_state.size = "Medium"; st.session_state["fx_done"] = False
            goto("result"); st.experimental_rerun()
        if draw_card("preset", "Large", _size_subtitle(fam, "Large"), key="size_large", family=fam):
            st.session_state.size = "Large"; st.session_state["fx_done"] = False
            goto("result"); st.experimental_rerun()

    # RIGHT: short “how to think about size” box
    with right:
        st.markdown("**How to think about ‘Small / Medium / Large’**")
        st.markdown(
            _size_explainer_md(fam),
            unsafe_allow_html=True,
        )
        st.caption("Tip: Medium is often the sweet spot — flexible enough to learn patterns, "
                   "but regularized to avoid memorizing noise.")

    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("⬅️ Pick a family"):
        goto("family"); st.experimental_rerun()

def _size_subtitle(fam: str, size: str) -> str:
    if fam == "logreg":
        return {"Small": "Top 10 features (with interactions)",
                "Medium": "L1 chooses the best subset",
                "Large": "All features + interactions"}[size]
    if fam == "tree":
        return {"Small": "Heavy pruning",
                "Medium": "CV-selected pruning (optimal α)",
                "Large": "Unpruned"}[size]
    if fam == "mlp":
        return {"Small": "Shallow network",
                "Medium": "Best tuned network",
                "Large": "Deeper network"}[size]
    return ""

def _size_explainer_md(fam: str) -> str:
    if fam == "logreg":
        return (
            "- **Small:** Keep only the top 10 most informative features (after creating squares & interactions).\n"
            "- **Medium:** Let **L1** regularization *select* a sparse, optimal subset automatically.\n"
            "- **Large:** Use **all** features (with squares & interactions) with L2.\n"
        )
    if fam == "tree":
        return (
            "- **Small:** A **heavily pruned** tree (fewer branches) → simple rules, less overfitting.\n"
            "- **Medium:** Pruning strength (α) chosen by **cross-validation**.\n"
            "- **Large:** **Unpruned** tree (can fit complex patterns but may overfit).\n"
        )
    if fam == "mlp":
        return (
            "- **Small:** A **shallow** network → quick, stable, less capacity.\n"
            "- **Medium:** The **best** tuned network (depth/width/reg picked on validation).\n"
            "- **Large:** A **deeper** network for more capacity (with early stopping to protect against overfit).\n"
        )
    return ""

def pane_result(manifest: Dict[str, Any]):
    fam = st.session_state.get("family")
    size = st.session_state.get("size")
    if fam not in ("logreg", "tree", "mlp") or size not in ("Small", "Medium", "Large"):
        goto("family"); st.experimental_rerun(); return

    task = manifest["tasks"]["pop3"]
    task_dir = Path(task["path"])
    key_map = {
        "logreg": {"Small": "logreg_small", "Medium": "logreg_medium", "Large": "logreg_large"},
        "tree":   {"Small": "tree_small",   "Medium": "tree_medium",   "Large": "tree_large"},
        "mlp":    {"Small": "mlp_small",    "Medium": "mlp_medium",    "Large": "mlp_large"},
    }
    model_key = key_map[fam][size]
    metrics_path = task_dir / model_key / "metrics.json"
    if not metrics_path.exists():
        st.error(f"Metrics not found for {model_key}. Run the trainer.")
        st.stop()

    metrics = json.loads(metrics_path.read_text())
    st.subheader("Out-of-sample results")
    st.markdown(f"**{friendly_model_name(fam, size)}**  \n*Task:* `pop3` (High / Medium / Low)")

    # Balloons/snow only once
    result_fx_once(metrics.get("accuracy", 0.0) >= 0.55)

    # Show concise, friendly metrics
    c1, c2 = st.columns(2)
    c1.metric("Accuracy (test)", f"{metrics.get('accuracy', 0)*100:.1f}%")
    c2.metric("Macro-F1 (test)", f"{metrics.get('macro_f1', 0):.3f}")

    st.markdown("<div class='edu-box'>", unsafe_allow_html=True)
    st.markdown("**Why Medium is often best**")
    st.markdown(
        """
- **Small** models can **underfit**: they’re simple and may miss patterns.
- **Large** models can **overfit**: they memorize noise and don’t generalize well.
- **Medium** models strike a **bias–variance balance** — enough flexibility to learn,
  plus regularization/pruning/early stopping to avoid overfitting.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='lined'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("⬅️ Change size"):
        st.session_state["fx_done"] = False
        goto("size"); st.experimental_rerun()
    if c2.button("↩︎ Pick a family"):
        st.session_state["fx_done"] = False
        goto("family"); st.experimental_rerun()
    if c3.button("🔁 Start over"):
        for k in list(st.session_state.keys()):
            if k not in ("step",):
                del st.session_state[k]
        st.session_state.step = 0
        st.experimental_rerun()

# -------------------- Router --------------------
manifest = load_manifest()
pane = steps[st.session_state.step]

if pane == "landing":
    pane_landing()
elif pane == "data":
    pane_data(manifest)
elif pane == "family":
    pane_family()
elif pane == "size":
    pane_size(manifest)
else:
    pane_result(manifest)
