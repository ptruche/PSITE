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
.chip {dis

