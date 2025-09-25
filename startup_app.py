# startup_app.py — clean baseline with landing page and working image cards

import sys, json
from pathlib import Path
import pandas as pd
import streamlit as st
import skops.io as skio
import sklearn

# -------------------- Config --------------------
st.set_page_config(page_title="Create a Startup", page_icon="🚀", layout="centered")
DEBUG = False

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "startup_model"
MODEL_PATH = MODEL_DIR / "startup_success_model.skops"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

ASSETS_DIR = APP_DIR / "assets" 
HERO_DIR = ASSETS_DIR / "hero"

# Hard map images you’ve created
IMAGE_MAP = {
    "biz": {
        "Tech / Software": str(ASSETS_DIR / "biz" / "tech_software.png"),
        "Web / Mobile":    str(ASSETS_DIR / "biz" / "web_mobile.png"),
        "E-Commerce":    str(ASSETS_DIR / "biz" / "ecommerce.png"),
        "Biotech":    str(ASSETS_DIR / "biz" / "biotech.png"),
        "Consulting":    str(ASSETS_DIR / "biz" / "consulting.png"),
        "Other":    str(ASSETS_DIR / "biz" / "other.png"),
    },
    "network": {
        "Well-connected insiders": str(ASSETS_DIR / "network" / "well-connected_insiders.png"),  # ← add this
    }
}

# -------------------- Config --------------------
st.set_page_config(page_title="Create a Startup", page_icon="🚀", layout="centered")



from pathlib import Path
import json
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "startup_model"

def _load_schema():
    candidates = [
        APP_DIR / "feature_schema.json",
        MODEL_DIR / "feature_schema.json",
        Path("feature_schema.json")
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def render_landing():
    meta = _load_schema()
    holdout = meta.get("holdout", {})
    roc_auc = holdout.get("roc_auc")
    acc = holdout.get("accuracy")
    f1 = holdout.get("f1")

    # LANDING-ONLY width bump (removed on next step because app reruns)
    st.markdown(
        """
        <style>
          .block-container { max-width: 1200px; padding-top: 3.5rem; padding-bottom: 2rem; }
          @media (min-width: 1600px) { .block-container { max-width: 1400px; } }
          .hero-h1 { font-size: 54px; font-weight: 800; line-height: 1.05; margin: 0 0 12px; }
          .hero-sub { font-size: 18px; opacity: 0.9; margin-bottom: 16px; }
          .big-button button { font-size: 20px !important; padding: 12px 0 !important; width: 100% !important; }
          .hero-emoji { font-size: 160px; line-height: 1; text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Keep your image/icon on the left
    try:
        col1, col2 = st.columns([2, 2], gap="large")
    except TypeError:
        col1, col2 = st.columns([2, 2])

    with col1:
        img = hero_image_path()
        if img:
            st.image(str(img), use_container_width=True)
        else:
            st.markdown("<div style='font-size:96px'>🚀</div>", unsafe_allow_html=True)

    with col2:
        # NEW headline + subtitle
        st.markdown(
            "<div class='hero-h1'>Data Scientists Build Models to Determine Which Factors Drive Predictions</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-sub'>Try to find the right factors to build a successful start-up.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='big-button'>", unsafe_allow_html=True)
        if st.button("Start modeling →", use_container_width=True):
            st.session_state.step = 1  # goes to "type"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Counterfactual helpers ----------
def predict_from_choices(biz, loc, fund_list, strat, net_label, vis_label, apply_adj=True):
    # Map labels → codes
    net_code = dict(ui["network"]).get(net_label, "mid")       # low/mid/high
    vis_code = dict(ui["visibility"]).get(vis_label, "low")    # low/high

    # Build feature row (same logic as your result page)
    cat_map = {
        "Tech / Software": "software", "Web / Mobile": "web",
        "E-commerce": "ecommerce", "E-Commerce": "ecommerce",  # support both spellings
        "Biotech": "biotech", "Consulting": "consulting", "Other": "other"
    }
    feat = {}
    feat["category_code"] = cat_map.get(biz, "other")

    if "CA" in loc: feat["state_code"] = "CA"
    elif "New York" in loc: feat["state_code"] = "NY"
    elif "Boston" in loc: feat["state_code"] = "MA"
    elif "Texas" in loc: feat["state_code"] = "TX"
    else: feat["state_code"] = "OT"

    sel = set(fund_list or [])
    feat["has_angel"]  = int("Angel" in sel)
    feat["has_VC"]     = int("VC" in sel)
    feat["has_roundA"] = int("Series A" in sel)
    feat["has_roundB"] = int("Series B" in sel)
    rounds = feat["has_angel"] + feat["has_VC"] + feat["has_roundA"] + feat["has_roundB"]
    if "Bootstrapped" in sel:
        rounds = max(rounds - 1, 0)
    feat["funding_rounds_b"] = pd.cut([rounds], bins=[-1, 0, 1, 3, 10], include_lowest=True).astype(str)[0]

    if "Big swing" in strat:
        feat["milestones_b"] = "(0, 1]"
        feat["age_first_funding_b"] = "(1, 3]"
    else:
        feat["milestones_b"] = "(1, 3]"
        feat["age_first_funding_b"] = "(0, 1]"

    bucket = {"low": "(-1, 0]", "mid": "(0, 2]", "high": "(2, 5]"}
    feat["relationships_b"]    = bucket[net_code]
    feat["avg_participants_b"] = bucket[net_code]
    feat["is_top500"] = int(vis_code == "high")

    cols = ["state_code", "category_code", "has_angel", "has_VC", "has_roundA", "has_roundB",
            "funding_rounds_b", "milestones_b", "age_first_funding_b", "relationships_b",
            "avg_participants_b", "is_top500"]
    x = pd.DataFrame([{c: feat.get(c, 0) for c in cols}], columns=cols)

    base_p = float(pipe.predict_proba(x)[:, 1][0])

    if apply_adj:
        adj = 0.0
        if vis_code == "high" and "Big swing" in strat:      adj += 0.03
        if vis_code == "high" and "Funding first" in strat:  adj -= 0.02
        if net_code == "high" and ("Bootstrapped" in sel):   adj -= 0.02
        if net_code == "low"  and ("Bootstrapped" in sel):   adj += 0.01
        base_p += adj

    return min(max(base_p, 0.01), 0.99)


# -------------------- Model loading --------------------
if not MODEL_PATH.exists():
    st.error("Model file not found")
    st.stop()

try:
    untrusted = skio.get_untrusted_types(file=MODEL_PATH)
    pipe = skio.load(MODEL_PATH, trusted=untrusted if untrusted else None)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

try:
    schema = json.loads(SCHEMA_PATH.read_text())
except Exception as e:
    st.error(f"Failed to read schema: {e}")
    st.stop()
ui = schema.get("ui", {})

# -------------------- Helpers --------------------
def slugify(label: str) -> str:
    return (
        label.lower()
        .replace("🚀", "").replace("💰", "")
        .replace("/", " ").replace("&", "and")
        .replace("(", "").replace(")", "")
        .replace(",", "").replace("’", "").replace("'", "")
        .strip().replace(" ", "_")
    )

def option_image_path(pane_key: str, option_label: str):
    mapped = IMAGE_MAP.get(pane_key, {}).get(option_label)
    if mapped and Path(mapped).exists():
        return Path(mapped)
    slug = slugify(option_label)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = ASSETS_DIR / pane_key / f"{slug}{ext}"
        if p.exists():
            return p
    return None

def hero_image_path():
    for name in ("landing", "hero", "cover"):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = HERO_DIR / f"{name}{ext}"
            if p.exists():
                return p
    return None

def draw_option_card(pane_key: str, option_label: str, sublabel: str | None, button_key: str) -> bool:
    img_path = option_image_path(pane_key, option_label)
    with st.container():
        if img_path:
            st.image(str(img_path), use_container_width=True)
        st.markdown(f"**{option_label}**")
        if sublabel:
            st.caption(sublabel)
        return st.button("Choose", key=button_key, use_container_width=True)

def grid_choice(pane_key: str, options, state_key: str, advance_fn):
    try:
        cols = st.columns(3, gap="medium")  # gap works on newer Streamlit
    except TypeError:
        cols = st.columns(3)
    for idx, (label, sub) in enumerate(options):
        with cols[idx % 3]:
            if draw_option_card(pane_key, label, sub, button_key=f"{pane_key}_{idx}"):
                st.session_state[state_key] = label
                advance_fn()
                st.rerun()


# -------------------- State --------------------
steps = ["landing", "type", "loc", "funding", "strategy", "network", "visibility", "result"]
if "step" not in st.session_state: st.session_state.step = 0
def nxt():  st.session_state.step = min(st.session_state.step + 1, len(steps)-1)
def back(): st.session_state.step = max(st.session_state.step - 1, 0)

for k in ["biz","loc","fund","strat","net_label","vis_label","fund_set"]:
    st.session_state.setdefault(k, None)
if st.session_state.fund_set is None:
    st.session_state.fund_set = set()

pane = steps[st.session_state.step]

# -------------------- Panes --------------------
if pane == "landing":
    # Use the modeling-forward landing you pasted above
    render_landing()
    st.stop()  # prevent the rest of the panes from rendering on first load


    with col1:
        img = hero_image_path()
        if img:
            st.image(str(img), use_container_width=True)
        else:
            # fallback emoji if no image
            st.markdown("<div style='font-size:80px'>🚀</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div style='font-size:42px; font-weight:800; margin-bottom:12px;'>"
            "Create a Startup</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:18px; line-height:1.4; margin-bottom:24px;'>"
            "Make a few decisions and see your survival probability "
            "based on a trained model.</div>",
            unsafe_allow_html=True,
        )
        # bigger CTA button (using HTML for styling)
        st.markdown(
            """
            <style>
            .big-button button {
                font-size: 20px !important;
                padding-top: 12px !important;
                padding-bottom: 12px !important;
                width: 100% !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.container().button("👉 Try it now", key="hero_start"):
            nxt()
            st.rerun()



elif pane == "type":
    st.subheader("What kind of startup are you?")
    opts = [(o, None) for o in ui["business_types"]]
    grid_choice("biz", opts, "biz", nxt)
    if st.button("⬅️ Back"): back()

elif pane == "loc":
    st.subheader("Where will you launch?")
    opts = [(o, None) for o in ui["locations"]]
    grid_choice("loc", opts, "loc", nxt)
    if st.button("⬅️ Back"): back()

elif pane == "funding":
    st.subheader("How will you fund the journey? Choose all ways you raise money.")
    fund_opts = ui["funding_options"]

    # 3-column grid (with gap on newer Streamlit)
    try:
        cols = st.columns(3, gap="medium")
    except TypeError:
        cols = st.columns(3)

    for i, label in enumerate(fund_opts):
        with cols[i % 3]:
            is_on = label in st.session_state.fund_set
            sub = "Selected" if is_on else None
            if draw_option_card("funding", label, sub, button_key=f"fund_{i}"):
                if is_on:
                    st.session_state.fund_set.remove(label)
                else:
                    st.session_state.fund_set.add(label)
                st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("⬅️ Back"):
        back()
    if c2.button("Next ➡️"):
        st.session_state.fund = sorted(list(st.session_state.fund_set))
        nxt()
        st.rerun()


elif pane == "strategy":
    st.subheader("Pick your early strategy")
    opts = [(o, None) for o in ui["strategy"]]
    grid_choice("strategy", opts, "strat", nxt)
    if st.button("⬅️ Back"): back()

elif pane == "network":
    st.subheader("What’s your team’s network like?")
    labels = [lbl for (lbl, _) in ui["network"]]
    subs = {"Independent operators": "Lean & focused",
            "Balanced connectors": "A few right doors",
            "Well-connected insiders": "Warm intros"}
    opts = [(lbl, subs.get(lbl)) for lbl in labels]
    grid_choice("network", opts, "net_label", nxt)
    if st.button("⬅️ Back"): back()

elif pane == "visibility":
    st.subheader("How visible are you?")
    labels = [lbl for (lbl, _) in ui["visibility"]]
    subs = {"Under the radar": "Quiet execution", "Top 500 buzz": "Momentum & press"}
    opts = [(lbl, subs.get(lbl)) for lbl in labels]
    grid_choice("visibility", opts, "vis_label", nxt)
    if st.button("⬅️ Back"): back()

else:
    # ========= Results page with What-if panel + one-time confetti/snow =========

    # --- Gather current baseline choices from session ---
    biz        = st.session_state.get("biz", "Other")
    loc        = st.session_state.get("loc", "Other")
    fund       = st.session_state.get("fund", sorted(list(st.session_state.fund_set)))
    strat      = st.session_state.get("strat", ui["strategy"][0])
    net_label  = st.session_state.get("net_label", "Balanced connectors")
    vis_label  = st.session_state.get("vis_label", "Under the radar")

    # --- Compute baseline probability (PURE MODEL) ---
    p = predict_from_choices(biz, loc, fund, strat, net_label, vis_label)
    net_code = dict(ui["network"]).get(net_label, "mid")     # low/mid/high
    vis_code = dict(ui["visibility"]).get(vis_label, "low")  # low/high

    # --- Layout: left = outcome + narrative; right = What-if controls ---
    try:
        left, right = st.columns([2, 1], gap="large")
    except TypeError:
        left, right = st.columns([2, 1])

    with left:
        st.subheader("Your outcome")

        outcome_good = p >= 0.50
        headline = "🎉 Congratulations!" if outcome_good else "Nice try!"
        color = "#16a34a" if outcome_good else "#dc2626"

        st.markdown(
            f"<div style='font-size:36px; font-weight:800; color:{color}; margin:4px 0 8px;'>{headline}</div>",
            unsafe_allow_html=True,
        )

        # --- Confetti / snow: fire only the first time results are shown ---
        if not st.session_state.get("result_fx_done", False):
            try:
                if outcome_good:
                    st.balloons()
                else:
                    st.snow()
            except Exception:
                pass
            st.session_state["result_fx_done"] = True

        # --- Optional splash images (if present) ---
        if outcome_good and (ASSETS_DIR / "celebrate.png").exists():
            st.image(str(ASSETS_DIR / "celebrate.png"), use_container_width=True)
        elif (not outcome_good) and (ASSETS_DIR / "try_again.png").exists():
            st.image(str(ASSETS_DIR / "try_again.png"), use_container_width=True)

        # --- Probability metric + brief narrative ---
        st.metric("Predicted survival probability", f"{p*100:.1f}%")

        bullets = []
        bullets.append("Fast milestones." if "Big swing" in strat else "Runway first.")
        bullets.append({"low": "Independent & lean.", "mid": "Balanced connections.", "high": "Insider access."}[net_code])
        bullets.append("Buzz + momentum." if vis_code == "high" else "Quiet execution.")
        st.markdown("\n".join(f"- {b}" for b in bullets))

        # --- Reset CTA ---
        if st.button("Start over 🔁", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k not in ("step", "fund_set"):
                    del st.session_state[k]
            st.session_state.step = 0
            st.session_state.fund_set = set()
            st.session_state["result_fx_done"] = False  # explicit reset for next run
            st.rerun()

    with right:
        st.subheader("What if?")

        # Helper to get safe index for selectboxes
        def _idx(lst, val, default=0):
            try:
                return lst.index(val)
            except Exception:
                return default

        # --- Controls pre-filled with current baseline ---
        cf_biz  = st.selectbox("Business type", ui["business_types"], index=_idx(ui["business_types"], biz), key="cf_biz")
        cf_loc  = st.selectbox("Location", ui["locations"], index=_idx(ui["locations"], loc), key="cf_loc")
        cf_fund = st.multiselect("Funding", ui["funding_options"], default=fund, key="cf_fund")
        cf_strat = st.selectbox("Strategy", ui["strategy"], index=_idx(ui["strategy"], strat), key="cf_strat")

        cf_net_label = st.selectbox(
            "Network",
            [lbl for (lbl, _) in ui["network"]],
            index=_idx([lbl for (lbl, _) in ui["network"]], net_label),
            key="cf_net"
        )
        cf_vis_label = st.selectbox(
            "Visibility",
            [lbl for (lbl, _) in ui["visibility"]],
            index=_idx([lbl for (lbl, _) in ui["visibility"]], vis_label),
            key="cf_vis"
        )

        # --- Counterfactual probability (PURE MODEL) ---
        cf_p = predict_from_choices(
            cf_biz, cf_loc, cf_fund, cf_strat, cf_net_label, cf_vis_label
        )

        st.metric(
            "Counterfactual probability",
            f"{cf_p*100:.1f}%",
            delta=f"{(cf_p - p)*100:.1f} pp"
        )

        # --- Apply CF back to baseline ---
        if st.button("Apply to baseline", use_container_width=True):
            st.session_state.update({
                "biz": cf_biz,
                "loc": cf_loc,
                "fund": sorted(list(cf_fund)),
                "strat": cf_strat,
                "net_label": cf_net_label,
                "vis_label": cf_vis_label,
            })
            st.session_state.fund_set = set(cf_fund)
            st.rerun()
