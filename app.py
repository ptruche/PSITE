# app.py — Pediatric Surgery Board Prep (CSV-only TrueLearn-style)

import streamlit as st
import pandas as pd
import numpy as np
import os, json
from datetime import datetime
from typing import List

# ---------------------------
# PAGE SETUP & STYLE
# ---------------------------
st.set_page_config(page_title="Pediatric Surgery Board Prep", page_icon="🩺", layout="wide")

STYLE = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 950px;}
h1.title {font-weight:800; letter-spacing:-0.02em; margin-bottom:.25rem;}
.caption {color:#6b7280; margin-bottom:1.25rem;}
.card {background:#fff; border:1px solid rgba(17,24,39,.06); border-radius:16px;
       padding:1.25rem 1.25rem; box-shadow:0 2px 8px rgba(2,6,23,.05);}
.card-dim {background:#f8fafc}
.chip {display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.78rem;
       font-weight:600; background:#eef2ff; color:#1f2937; border:1px solid rgba(59,130,246,.25);
       margin-right:.35rem;}
.stButton>button {border-radius:12px !important; font-weight:600 !important;}
.stRadio [role='radiogroup'] label {padding:.35rem .55rem; border-radius:10px;}
.stRadio [role='radiogroup'] label:hover {background:#f3f4f6;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ---------------------------
# CONSTANTS
# ---------------------------
DATA_DIR = "data"
REQUIRED = ["Question","A","B","C","D","E","Correct","Explanation","Reference","Category","Difficulty"]
PROGRESS_FILE = "progress.json"

# ---------------------------
# HELPERS (CSV ONLY)
# ---------------------------
def _normalize_str(s: str) -> str:
    return str(s).replace("’", "'").replace("–", "-").replace("\u00A0", " ").strip()

def list_csv_files() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([f for f in os.listdir(DATA_DIR)
                   if f.lower().endswith(".csv") and not f.startswith(".")])

def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV (UTF-8-sig tolerant) and validate required columns."""
    p = os.path.join(DATA_DIR, path)
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for c in ["Question","A","B","C","D","E","Explanation","Reference","Category","Difficulty"]:
        df[c] = df[c].astype(str).map(_normalize_str)
    df["Correct"] = df["Correct"].astype(str).str.strip().str.upper()
    df["__sourcefile__"] = path
    return df

def combine_selected(files: List[str]) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            frames.append(load_csv(f))
        except Exception as e:
            st.error(f"❌ Failed to load {f}: {e}")
    if not frames:
        return pd.DataFrame(columns=REQUIRED)
    return pd.concat(frames, ignore_index=True)

def init_state():
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("indices", [])
    ss.setdefault("i", 0)
    ss.setdefault("answers", {})
    ss.setdefault("show_expl", False)
    ss.setdefault("filters", {"Category": [], "Difficulty": []})
    ss.setdefault("shuffle", True)
    ss.setdefault("review_mode", False)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filt = st.session_state.filters
    mask = pd.Series(True, index=df.index)
    for key in ["Category","Difficulty"]:
        vals = filt.get(key) or []
        if vals:
            mask &= df[key].isin(vals)
    return df[mask].reset_index(drop=True)

def rebuild_indices():
    sub = apply_filters(st.session_state.df)
    idxs = list(range(len(sub)))
    if st.session_state.shuffle:
        np.random.default_rng().shuffle(idxs)
    st.session_state.indices = idxs
    st.session_state.i = 0
    st.session_state.answers = {}
    st.session_state.show_expl = False

def persist_progress():
    try:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "answers": st.session_state.answers,
            "filters": st.session_state.filters,
            "shuffle": st.session_state.shuffle,
        }
        with open(PROGRESS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

# ---------------------------
# SIDEBAR
# ---------------------------
init_state()
st.sidebar.title("⚙️ Controls")

files = list_csv_files()
if not files:
    st.sidebar.error("No CSV files found in `data/`.\nAdd files with headers:\n" + ", ".join(REQUIRED))
    st.stop()

st.sidebar.caption("Question sets in repository:")
st.sidebar.write(files)

use_all = st.sidebar.toggle("Use ALL sets", value=True)
selected = files if use_all else st.sidebar.multiselect("Choose set(s)", files, default=files[:1])

if st.sidebar.button("Load / Reload", type="primary", use_container_width=True):
    st.session_state.df = combine_selected(selected)
    if not st.session_state.df.empty:
        rebuild_indices(); persist_progress()

# Auto-load
if st.session_state.df is None:
    st.session_state.df = combine_selected(selected)
    if not st.session_state.df.empty and not st.session_state.indices:
        rebuild_indices()

if st.session_state.df is None or st.session_state.df.empty:
    st.info("Add CSVs to `data/` and click **Load / Reload** to begin.")
    st.stop()

df = st.session_state.df
cats = sorted([c for c in df["Category"].dropna().unique().tolist() if c])
diffs = sorted([d for d in df["Difficulty"].dropna().unique().tolist() if d])

with st.sidebar.expander("Filters", expanded=False):
    st.session_state.filters["Category"] = st.multiselect("Category", cats, default=st.session_state.filters.get("Category", []))
    st.session_state.filters["Difficulty"] = st.multiselect("Difficulty", diffs, default=st.session_state.filters.get("Difficulty", []))
    st.session_state.shuffle = st.toggle("Shuffle questions", value=st.session_state.shuffle)
    c1, c2 = st.columns(2)
    if c1.button("Apply", use_container_width=True):
        rebuild_indices(); persist_progress()
    if c2.button("Reset", use_container_width=True):
        st.session_state.filters = {"Category": [], "Difficulty": []}
        rebuild_indices(); persist_progress()

st.sidebar.markdown("---")
st.session_state.review_mode = st.sidebar.toggle("📖 Review Mode (no grading)", value=st.session_state.review_mode)

# ---------------------------
# MAIN UI
# ---------------------------
st.markdown("<h1 class='title'>Pediatric Surgery Board Prep</h1>", unsafe_allow_html=True)
st.caption("TrueLearn-style question bank • Clean, focused interface")

sub = apply_filters(df)
if len(sub) == 0:
    st.warning("No questions match current filters.")
    st.stop()

st.session_state.i = int(np.clip(st.session_state.i, 0, len(st.session_state.indices)-1))
local_idx = st.session_state.indices[st.session_state.i]
q = sub.loc[local_idx]

answered = len(st.session_state.answers)
total = len(st.session_state.indices)
pct = int((answered / total) * 100) if total else 0
st.progress(min(pct/100, 1.0), text=f"Progress: {answered}/{total} answered • {pct}%")

chips = []
if q["Category"]: chips.append(f"<span class='chip'>Category: {q['Category']}</span>")
if q["Difficulty"]: chips.append(f"<span class='chip'>Difficulty: {q['Difficulty']}</span>")
if "__sourcefile__" in q and q["__sourcefile__"]: chips.append(f"<span class='chip'>Set: {q['__sourcefile__']}</span>")
if chips: st.markdown(" ".join(chips), unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(f"### Q{st.session_state.i+1}")
st.write(q["Question"])

choices = {"A": q["A"], "B": q["B"], "C": q["C"], "D": q["D"], "E": q["E"]}

if st.session_state.review_mode:
    st.info("Review Mode ON — submissions are not graded.")
    picked_letter = None
else:
    previous = st.session_state.answers.get(local_idx, {}).get("choice")
    opt_list = [f"{k}. {v}" for k,v in choices.items()]
    default_index = list(choices.keys()).index(previous) if previous in choices else 0
    pick = st.radio("Choose one:", opt_list, index=default_index)
    picked_letter = pick.split(".")[0]

col = st.columns([1,1,1,6])
if col[0].button("✅ Submit", disabled=st.session_state.review_mode):
    is_correct = (picked_letter == q["Correct"])
    st.session_state.answers[local_idx] = {
        "choice": picked_letter,
        "is_correct": bool(is_correct),
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.show_expl = True
    if is_correct: st.success("Correct ✅")
    else: st.error(f"Incorrect ❌ • Correct: **{q['Correct']}**")
    persist_progress()

if col[1].button("👁 Show explanation"): st.session_state.show_expl = True
if col[2].button("🙈 Hide explanation"): st.session_state.show_expl = False

if st.session_state.show_expl or st.session_state.review_mode:
    st.markdown("<div class='card card-dim'>", unsafe_allow_html=True)
    st.markdown("**Explanation**")
    st.write(q["Explanation"])
    if str(q["Reference"]).lower().startswith(("http://","https://")):
        st.markdown(f"[Reference]({q['Reference']})")
    st.markdown("</div>", unsafe_allow_html=True)

nav = st.columns([1,1,6])
if nav[0].button("⬅️ Previous", disabled=st.session_state.i==0):
    st.session_state.i = max(0, st.session_state.i - 1)
    st.session_state.show_expl = False
if nav[1].button("Next ➡️", disabled=st.session_state.i >= len(st.session_state.indices)-1):
    st.session_state.i = min(len(st.session_state.indices)-1, st.session_state.i + 1)
    st.session_state.show_expl = False

st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📊 Performance by Category"):
    if st.session_state.answers:
        rows = [(sub.loc[idx, "Category"], 1 if rec.get("is_correct") else 0)
                for idx, rec in st.session_state.answers.items() if idx < len(sub)]
        if rows:
            perf = pd.DataFrame(rows, columns=["Category","Correct"]).groupby("Category")\
                .agg(Attempts=("Correct","count"), Correct=("Correct","sum"))
            perf["Accuracy"] = (perf["Correct"] / perf["Attempts"] * 100).round(1)
            st.dataframe(perf, use_container_width=True)
    else:
        st.write("No graded attempts yet.")
