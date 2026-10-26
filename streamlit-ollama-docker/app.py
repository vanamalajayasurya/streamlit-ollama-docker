import hashlib
import hmac
import html
import inspect
import io
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache

import altair as alt
import ollama
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from urllib.parse import quote

APP_NAME = "TalentMatch AI"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DB_PATH = os.getenv("DB_PATH", "/data/recruit.db")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
client = ollama.Client(host=OLLAMA_HOST, timeout=600)


def _wide_kwargs() -> dict:
    """Full-width buttons: new Streamlit uses width='stretch', old uses use_container_width."""
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


WIDE = _wide_kwargs()

JD_CHARS = 3000       # job description characters sent to the model
RESUME_CHARS = 5000   # resume characters sent to the model
STATUSES = ["New", "Shortlisted", "Interviewing", "Rejected"]

WINE = "#7F011F"
SAND = "#F5EBD0"
BAND_COLORS = {"Strong fit": "#2f7d4f", "Good fit": WINE, "Partial fit": "#d49a2a", "Low fit": "#a89b86"}
STATUS_COLORS = {"New": "#a89b86", "Shortlisted": "#2f7d4f", "Interviewing": "#d49a2a", "Rejected": WINE}

THEME = {
    "base": "light", "primaryColor": "#7F011F", "backgroundColor": "#F5EBD0",
    "secondaryBackgroundColor": "#EFE1BF", "textColor": "#2a0a12", "font": "sans serif",
}


def _ensure_theme():
    """Streamlit only reads .streamlit/config.toml, so create it if missing (applies from the next restart)."""
    try:
        path = os.path.join(".streamlit", "config.toml")
        if not os.path.exists(path):
            os.makedirs(".streamlit", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("[theme]\n" + "".join(f'{k} = "{v}"\n' for k, v in THEME.items()))
    except OSError:
        pass
    try:  # also applies to new browser sessions right away
        for k, v in THEME.items():
            st._config.set_option(f"theme.{k}", v)
    except Exception:
        pass


_ensure_theme()
st.set_page_config(page_title=APP_NAME, page_icon="🍷", layout="wide")

# ---------------------------------------------------------------- design
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;600;700&display=swap');
:root {
  --wine: #7F011F; --wine-2: #A3153A; --wine-dark: #4d0013;
  --sand: #F5EBD0; --sand-2: #EFE1BF; --card: #FBF6E8; --line: #e0d2ae;
  --ink: #2a0a12; --muted: #7a6a5c;
  --strong: #2f7d4f; --good: #7F011F; --partial: #d49a2a; --low: #a89b86;
}
html, body, .stApp { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(1200px 500px at 90% -10%, #fbf3dc 0%, var(--sand) 60%); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 6rem; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; letter-spacing: -.02em; color: var(--wine); }

/* sidebar */
section[data-testid="stSidebar"] { background: var(--sand-2); border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: 12px; font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800; font-size: 1.35rem; letter-spacing: -.02em; color: var(--wine); margin-bottom: .2rem; }
.brand em { font-style: normal; background: var(--wine); color: var(--sand); border-radius: 8px; padding: 0 7px; margin-left: 4px; font-size: 1rem; }
.brand-sub { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; }
.side-label { font-size: .8rem; font-weight: 700; color: var(--wine); margin: 1.1rem 0 .3rem; }
.who { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; font-size: .88rem; margin: .6rem 0 1rem; }
.who b { color: var(--wine); }

/* the match mark: two circles that slide together */
.mark { position: relative; width: 52px; height: 32px; flex: none; }
.mark i { position: absolute; top: 0; width: 32px; height: 32px; border-radius: 50%; border: 3px solid var(--wine); box-sizing: border-box; }
.mark i:nth-child(1) { left: 0; }
.mark i:nth-child(2) { right: 0; background: rgba(127,1,31,.25); }
.mark.light i { border-color: var(--sand); }
.mark.light i:nth-child(2) { background: rgba(245,235,208,.35); }

/* page heading */
.page-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1.05;
  letter-spacing: -.03em; color: var(--wine); margin: 0 0 .4rem; }
.page-sub { color: var(--muted); max-width: 62ch; margin-bottom: 1.6rem; line-height: 1.55; }
.section-title { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--wine); margin-bottom: .2rem; }
.chart-title { font-weight: 700; color: var(--ink); margin: 0 0 .2rem; }
.muted { color: var(--muted); }
.step { font-weight: 700; color: var(--wine); margin: 0 0 .4rem; }

/* chat */
[data-testid="stChatMessage"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; }
[data-testid="stChatInput"] { border-radius: 14px; border: 1px solid var(--line); background: var(--card); }
.empty { text-align: center; margin: 18vh auto 0; }
.empty .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.3rem; font-weight: 800; color: var(--wine); letter-spacing: -.02em; }
.empty .small { color: var(--muted); margin-top: .5rem; }

/* inputs */
.stTextInput input, .stTextArea textarea { background: #fffdf6 !important; color: var(--ink) !important; }
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
  background: #fffdf6 !important; border-color: var(--line) !important; border-radius: 10px !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within { border-color: var(--wine) !important; box-shadow: 0 0 0 2px rgba(127,1,31,.15); }
[data-testid="stFileUploader"] section { background: var(--card); border: 1.5px dashed #c9b88c; border-radius: 12px; }
button[data-baseweb="tab"] { font-weight: 700; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--wine); }
div[data-baseweb="tab-highlight"] { background: var(--wine) !important; }
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; }
/* (old Streamlit wraps every block in stVerticalBlockBorderWrapper: never style it) */
[data-testid="stForm"] { border: 0; padding: 0; }

/* buttons */
button[data-testid^="stBaseButton-primary"], .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--wine) 0%, var(--wine-2) 100%); color: var(--sand); border: 0;
  border-radius: 12px; font-weight: 700; padding: .55rem 1.2rem; box-shadow: 0 6px 16px rgba(127,1,31,.28);
  transition: transform .15s, box-shadow .15s; }
button[data-testid^="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover {
  transform: translateY(-1px); box-shadow: 0 10px 22px rgba(127,1,31,.36); color: #fff; }
button[data-testid^="stBaseButton-primary"]:active { transform: translateY(0); }
button[data-testid^="stBaseButton-secondary"], .stButton > button[kind="secondary"], .stDownloadButton > button {
  background: transparent; color: var(--wine); border: 1.5px solid var(--wine); border-radius: 12px;
  font-weight: 700; transition: background .15s, color .15s; }
button[data-testid^="stBaseButton-secondary"]:hover, .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {
  background: var(--wine); color: var(--sand); border-color: var(--wine); }
button[data-testid^="stBaseButton"] p { color: inherit; }
button:focus-visible { outline: 2px solid var(--wine) !important; outline-offset: 2px; }

/* stat tiles */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: .4rem 0 1.4rem; }
.stat { background: var(--wine); color: var(--sand); border-radius: 16px; padding: 16px 20px; }
.stat:nth-child(even) { background: var(--card); color: var(--wine); border: 1px solid var(--line); }
.stat b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.3rem; font-weight: 800; line-height: 1.05; letter-spacing: -.02em; }
.stat span { opacity: .85; font-size: .85rem; }
@media (max-width: 700px) { .stats { grid-template-columns: repeat(2, 1fr); } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 22px; margin-bottom: 14px; }

/* candidate scorecard */
.cand { --c: var(--low); display: grid; grid-template-columns: 44px 1fr 130px; gap: 16px;
  background: var(--card); border: 1px solid var(--line); border-left: 6px solid var(--c);
  border-radius: 14px; padding: 16px 20px; margin-top: 14px; }
.cand.strong { --c: var(--strong); } .cand.good { --c: var(--good); }
.cand.partial { --c: var(--partial); } .cand.low { --c: var(--low); }
.cand .rank { color: var(--muted); font-weight: 700; padding-top: 4px; }
.cand .cname { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.3rem; font-weight: 800; letter-spacing: -.01em; }
.cand .badge { font-family: 'DM Sans', sans-serif; font-size: .75rem; font-weight: 700; color: var(--c);
  border: 1.5px solid var(--c); border-radius: 999px; padding: 2px 10px; margin-left: 8px; vertical-align: middle; }
.cand .cmeta { color: var(--muted); font-size: .85rem; margin: 2px 0 8px; }
.cand .csum { font-size: .92rem; line-height: 1.5; margin-top: 8px; max-width: 70ch; }
.cand .cscore { text-align: right; }
.cand .num { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.8rem; font-weight: 800; line-height: 1; color: var(--c); font-variant-numeric: tabular-nums; }
.cand .bar { height: 7px; background: #e6d9b8; border-radius: 4px; margin: 8px 0 6px; overflow: hidden; }
.cand .bar i { display: block; height: 100%; background: var(--c); animation: grow .9s cubic-bezier(.2,.8,.2,1) both; }
.cand .cscore small { color: var(--muted); font-size: .75rem; }
@keyframes grow { from { width: 0; } }
@media (max-width: 700px) { .cand { grid-template-columns: 1fr; } .cand .cscore { text-align: left; } }
.chip { display: inline-block; font-size: .78rem; padding: 2px 10px; border-radius: 999px; margin: 0 6px 6px 0; border: 1px solid var(--line); }
.chip.ok { color: var(--strong); border-color: rgba(47,125,79,.45); background: rgba(47,125,79,.09); }
.chip.no { color: var(--wine-2); border-color: rgba(127,1,31,.4); background: rgba(127,1,31,.07); }
.chip.plain { color: var(--ink); background: var(--sand-2); }
.qhead { font-weight: 700; color: var(--wine); margin: .9rem 0 .2rem; }

/* resume score ring (fills in when it appears) */
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
.ring { width: 190px; height: 190px; border-radius: 50%; margin: 6px auto; display: grid; place-items: center;
  background: conic-gradient(var(--c, #7F011F) calc(var(--p) * 1%), #e6d9b8 0); animation: ringfill 1.2s cubic-bezier(.2,.8,.2,1) both; }
@keyframes ringfill { from { --p: 0; } }
.ring-in { width: 150px; height: 150px; border-radius: 50%; background: var(--sand);
  display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring-in span { font-family: 'Bricolage Grotesque', sans-serif; font-size: 3.2rem; font-weight: 800; line-height: 1; color: var(--wine); }
.ring-in small { color: var(--muted); margin-top: 6px; }

/* splash: the one orchestrated moment */
.splash { position: fixed; inset: 0; z-index: 999999; background: var(--wine); color: var(--sand);
  display: flex; align-items: center; justify-content: center; animation: splashout .7s ease 3.0s forwards; }
.splash-glow { position: absolute; width: 60vmax; height: 60vmax; border-radius: 50%;
  background: radial-gradient(circle, rgba(245,235,208,.16) 0%, rgba(245,235,208,0) 65%); animation: breathe 3s ease-in-out infinite; }
.splash-inner { position: relative; text-align: center; }
.splash .mark { width: 96px; height: 58px; margin: 0 auto 26px; }
.splash .mark i { width: 58px; height: 58px; border-width: 4px; }
.splash .mark i:nth-child(1) { animation: joinL 1.2s .2s cubic-bezier(.2,.8,.2,1) both; }
.splash .mark i:nth-child(2) { animation: joinR 1.2s .2s cubic-bezier(.2,.8,.2,1) both; }
@keyframes joinL { from { transform: translateX(-90px); opacity: 0; } }
@keyframes joinR { from { transform: translateX(90px); opacity: 0; } }
.splash-word { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: clamp(2.8rem, 9vw, 5.6rem);
  letter-spacing: -.04em; line-height: 1; white-space: nowrap; }
.splash-word span { display: inline-block; animation: letter .6s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(1.0s + var(--i) * .06s); }
.splash-word em { display: inline-block; font-style: normal; background: var(--sand); color: var(--wine); border-radius: 14px;
  padding: 0 .18em; margin-left: .2em; animation: pop .6s cubic-bezier(.3,1.6,.5,1) 1.8s both; }
@keyframes letter { from { transform: translateY(40px); opacity: 0; } }
@keyframes pop { from { transform: scale(.2); opacity: 0; } }
.splash-tag { margin-top: 18px; opacity: 0; font-size: 1.05rem; animation: fadein .6s ease 2.2s forwards; }
.splash-load { width: 220px; height: 4px; border-radius: 4px; background: rgba(245,235,208,.22); margin: 26px auto 0; overflow: hidden; }
.splash-load b { display: block; height: 100%; background: var(--sand); animation: grow 3s ease-out both; }
@keyframes fadein { to { opacity: .85; } }
@keyframes splashout { to { opacity: 0; visibility: hidden; transform: scale(1.04); } }
@keyframes breathe { 50% { transform: scale(1.12); } }

/* login */
.hero { background: linear-gradient(160deg, var(--wine) 0%, var(--wine-dark) 100%); color: var(--sand);
  border-radius: 24px; padding: 40px 38px; min-height: 540px; display: flex; flex-direction: column; justify-content: space-between; }
.hero h1 { color: var(--sand); font-size: 3.3rem; line-height: 1; letter-spacing: -.04em; margin: 26px 0 14px; }
.hero p { max-width: 36ch; line-height: 1.55; opacity: .88; }
.hero ul { padding-left: 1.1rem; line-height: 1.9; opacity: .92; }
.minibars { display: flex; align-items: flex-end; gap: 8px; height: 70px; margin-top: 18px; }
.minibars i { width: 16px; border-radius: 5px 5px 0 0; background: var(--sand); opacity: .85;
  height: calc(var(--h) * 1%); transform-origin: bottom; animation: rise 1.4s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--n) * .12s); }
@keyframes rise { from { transform: scaleY(0); } }
.auth-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.1rem; color: var(--wine); letter-spacing: -.03em; margin: 14px 0 2px; }
.auth-sub { color: var(--muted); margin-bottom: 1rem; }
.or { display: flex; align-items: center; gap: 12px; color: var(--muted); font-size: .85rem; margin: .9rem 0 .6rem; }
.or::before, .or::after { content: ""; flex: 1; height: 1px; background: var(--line); }

/* readable on any browser theme: text, labels, inputs, menus, chat bar */
:root, .stApp { color-scheme: light; }
[data-testid="stMarkdownContainer"] { color: var(--ink); }
button [data-testid="stMarkdownContainer"], button p { color: inherit !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--ink) !important; }
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p, [data-testid="stToggle"] label p { color: var(--ink) !important; }
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { background: #fffdf6 !important; border: 2px solid var(--wine) !important; }
[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) > div:first-child { background: var(--wine) !important; }
[data-testid="stToggle"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-child { background: var(--wine) !important; }
[data-testid="stSlider"] div[role="slider"] { background-color: var(--wine) !important; }
[data-testid="stSliderThumbValue"] { color: var(--wine) !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color: #9a8b7a !important; opacity: 1 !important; -webkit-text-fill-color: #9a8b7a !important; }
textarea:disabled { -webkit-text-fill-color: var(--ink) !important; opacity: 1 !important; }
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div { border: 1px solid var(--line) !important; }
div[data-baseweb="base-input"] { background: #fffdf6 !important; border: 0 !important; }
div[data-baseweb="input"] button { background: transparent !important; color: var(--wine) !important; }
div[data-baseweb="input"] svg, div[data-baseweb="select"] svg { fill: var(--wine) !important; }
div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="menu"] { background: #fffdf6 !important; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * { color: var(--ink) !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background: var(--sand-2) !important; }
button[data-baseweb="tab"] { color: var(--muted); }
button[data-baseweb="tab"] p { color: inherit !important; }
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--ink); }
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] div[data-baseweb] { background: var(--card) !important; }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; background: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--wine) !important; color: var(--sand) !important; border-radius: 10px; }

@media (prefers-reduced-motion: reduce) {
  .splash, .splash * , .ring, .cand .bar i, .minibars i { animation-duration: .01s !important; animation-delay: 0s !important; }
}
/* ================= REDESIGN v2: top bar, hero banners, round cards ================= */
:root { --shadow: 0 10px 32px rgba(77,0,19,.10); --paper: #FFFDF6; }
@keyframes rise2 { from { opacity: 0; transform: translateY(14px); } }
.stApp { background: #F5EBD0 !important; }
.block-container { max-width: 1180px; padding-top: 4rem; }
section[data-testid="stSidebar"] { background: #EFE1BF; border-right: 0; }
.side-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.3rem; color: var(--wine); margin: .3rem 0 .2rem; }

/* text and form controls: always dark on light, whatever the browser theme */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--ink) !important; }
div[data-baseweb="select"] > div { background: var(--paper) !important; border: 1.5px solid var(--line) !important; border-radius: 12px !important; }
div[data-baseweb="select"] div, div[data-baseweb="select"] span, div[data-baseweb="select"] input { color: var(--ink) !important; }
div[data-baseweb="select"] svg { fill: var(--wine) !important; }
div[data-baseweb="popover"] ul { background: var(--paper) !important; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * { color: var(--ink) !important; }
div[data-baseweb="popover"] li:hover { background: #f3e7c6 !important; }
[data-testid="stSlider"] div[role="slider"] { background-color: var(--wine) !important; }

/* top navigation: pill tabs */
div[data-testid="stRadio"],
div[data-testid="stRadio"] > div,
div[data-testid="stRadioGroup"],
div[data-testid="stRadioGroup"] > div,
div[role="radiogroup"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
div[data-testid="stRadio"] {
  padding: 0 !important;
}
div[role="radiogroup"] { gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
div[role="radiogroup"] label[data-baseweb="radio"] { background: transparent; border: 1.5px solid rgba(127,1,31,.28);
  border-radius: 999px; padding: 7px 20px; margin: 0; cursor: pointer; transition: background .18s, transform .18s, box-shadow .18s; }
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display: none !important; }
div[role="radiogroup"] label[data-baseweb="radio"] p { color: var(--wine) !important; font-weight: 700; font-size: .92rem; margin: 0; }
div[role="radiogroup"] label[data-baseweb="radio"]:hover { background: rgba(127,1,31,.09); transform: translateY(-1px); }
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) { background: var(--wine); border-color: var(--wine); box-shadow: 0 8px 18px rgba(127,1,31,.30); }
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p { color: var(--sand) !important; }
@media (max-width: 800px) { div[role="radiogroup"] { justify-content: flex-start; } }

/* hero banner at the top of every page */
.hero2 { position: relative; overflow: hidden; background: linear-gradient(135deg, #8a0222 0%, #4d0013 100%);
  color: var(--sand); border-radius: 28px; padding: 34px 40px; margin: 1rem 0 1.7rem; box-shadow: var(--shadow); animation: rise2 .6s both; }
.hero2::before, .hero2::after { content: ""; position: absolute; border-radius: 50%; width: 200px; height: 200px; }
.hero2::before { right: 110px; top: -50px; border: 3px solid rgba(245,235,208,.32); }
.hero2::after { right: 12px; top: 26px; background: rgba(245,235,208,.12); border: 3px solid rgba(245,235,208,.20); }
.hero2 .eyebrow { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 700; opacity: .7; }
.hero2 .hero-title { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.7rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.05; margin: 8px 0 10px; }
.hero2 .hero-sub { max-width: 56ch; line-height: 1.55; opacity: .9; position: relative; z-index: 1; }

/* cards */
[class*="st-key-card_"] { background: var(--paper); border: 1px solid var(--line); border-radius: 22px; box-shadow: var(--shadow);
  padding: 1.1rem 0 1.2rem; overflow: hidden; }
[class*="st-key-card_"] .chart-title { padding: 0 1.4rem; }
.card { background: var(--paper); border-radius: 22px; box-shadow: var(--shadow); }
[data-testid="stExpander"] { background: var(--paper); border-radius: 18px; margin-top: 8px; }
.step { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.3rem; font-weight: 800; color: var(--wine); margin-bottom: .6rem; }
.section-title { font-size: 1.8rem; }

/* buttons: pills */
button[data-testid^="stBaseButton-primary"], .stButton > button[kind="primary"] { border-radius: 999px; padding: .65rem 1.7rem; }
button[data-testid^="stBaseButton-secondary"], .stButton > button[kind="secondary"], .stDownloadButton > button { border-radius: 999px; }

/* stat tiles */
.stats { gap: 14px; }
.stat, .stat:nth-child(even) { background: var(--paper); color: var(--wine); border: 1px solid var(--line); border-radius: 22px;
  box-shadow: var(--shadow); padding: 20px 24px; animation: rise2 .6s both; }
.stat:first-child { background: linear-gradient(135deg, #8a0222, #4d0013); color: var(--sand); border: 0; }
.stat b { font-size: 2.6rem; }

/* candidate card with score ring */
.cand2 { --c: var(--low); display: grid; grid-template-columns: 96px 1fr; gap: 24px; align-items: start;
  background: var(--paper); border: 1px solid var(--line); border-radius: 24px; padding: 22px 26px; margin-top: 16px;
  box-shadow: var(--shadow); animation: rise2 .55s both; }
.cand2.strong { --c: var(--strong); } .cand2.good { --c: var(--good); } .cand2.partial { --c: var(--partial); } .cand2.low { --c: var(--low); }
.c2ring { width: 96px; height: 96px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), #eadfc3 0); animation: ringfill 1.2s cubic-bezier(.2,.8,.2,1) both; }
.c2in { width: 74px; height: 74px; border-radius: 50%; background: var(--paper); display: flex; flex-direction: column; align-items: center; justify-content: center; }
.c2in b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.8rem; font-weight: 800; line-height: 1; color: var(--c); }
.c2in small { font-size: .65rem; color: var(--muted); }
.c2top { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
.c2rank { background: var(--sand-2); color: var(--wine); font-weight: 700; font-size: .78rem; border-radius: 999px; padding: 2px 10px; }
.c2name { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.4rem; font-weight: 800; letter-spacing: -.01em; }
.c2top .badge { margin-left: 0; font-size: .75rem; font-weight: 700; color: var(--c); border: 1.5px solid var(--c); border-radius: 999px; padding: 2px 10px; }
.c2sub { color: var(--muted); font-size: .78rem; margin-top: 8px; }
@media (max-width: 700px) { .cand2 { grid-template-columns: 1fr; } }

/* chat bubbles */
[data-testid="stChatMessage"] { border-radius: 22px; box-shadow: var(--shadow); animation: rise2 .4s both; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: var(--wine); border-color: var(--wine); margin-left: 16%; flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * { color: var(--sand) !important; }
[data-testid="stChatInput"] { border-radius: 999px; box-shadow: var(--shadow); }
.empty .big { font-size: 2.6rem; }
/* newer Streamlit radio markup (react-aria) */
label[data-testid="stRadioOption"] { background: transparent; border: 1.5px solid rgba(127,1,31,.28); border-radius: 999px;
  padding: 7px 20px; margin: 0; cursor: pointer; transition: background .18s, transform .18s, box-shadow .18s; }
label[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
label[data-testid="stRadioOption"] p { color: var(--wine) !important; font-weight: 700; font-size: .92rem; margin: 0; }
label[data-testid="stRadioOption"]:hover { background: rgba(127,1,31,.09); transform: translateY(-1px); }
label[data-testid="stRadioOption"][data-selected="true"], label[data-testid="stRadioOption"]:has(input:checked) {
  background: var(--wine); border-color: var(--wine); box-shadow: 0 8px 18px rgba(127,1,31,.30); }
label[data-testid="stRadioOption"][data-selected="true"] p, label[data-testid="stRadioOption"]:has(input:checked) p { color: var(--sand) !important; }
div[data-testid="stRadioGroup"] { gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
@media (max-width: 800px) { div[data-testid="stRadioGroup"] { justify-content: flex-start; } }
/* user chat bubble: wine, on the right */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { background: var(--wine); border-color: var(--wine);
  width: fit-content; max-width: 82%; margin-left: auto; flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) * { color: var(--sand) !important; }
label[data-testid="stRadioOption"] { padding: 7px 17px; }
div[role="radiogroup"] label[data-baseweb="radio"] > div { margin-left: 0 !important; }
div[role="radiogroup"] label[data-baseweb="radio"] { padding: 7px 17px; }
/* values for rings / splash letters (inline style attributes are stripped by older Streamlit) */
.sc0{--p:0}.sc1{--p:1}.sc2{--p:2}.sc3{--p:3}.sc4{--p:4}.sc5{--p:5}.sc6{--p:6}.sc7{--p:7}.sc8{--p:8}.sc9{--p:9}.sc10{--p:10}.sc11{--p:11}.sc12{--p:12}.sc13{--p:13}.sc14{--p:14}.sc15{--p:15}.sc16{--p:16}.sc17{--p:17}.sc18{--p:18}.sc19{--p:19}.sc20{--p:20}.sc21{--p:21}.sc22{--p:22}.sc23{--p:23}.sc24{--p:24}.sc25{--p:25}.sc26{--p:26}.sc27{--p:27}.sc28{--p:28}.sc29{--p:29}.sc30{--p:30}.sc31{--p:31}.sc32{--p:32}.sc33{--p:33}.sc34{--p:34}.sc35{--p:35}.sc36{--p:36}.sc37{--p:37}.sc38{--p:38}.sc39{--p:39}.sc40{--p:40}.sc41{--p:41}.sc42{--p:42}.sc43{--p:43}.sc44{--p:44}.sc45{--p:45}.sc46{--p:46}.sc47{--p:47}.sc48{--p:48}.sc49{--p:49}.sc50{--p:50}.sc51{--p:51}.sc52{--p:52}.sc53{--p:53}.sc54{--p:54}.sc55{--p:55}.sc56{--p:56}.sc57{--p:57}.sc58{--p:58}.sc59{--p:59}.sc60{--p:60}.sc61{--p:61}.sc62{--p:62}.sc63{--p:63}.sc64{--p:64}.sc65{--p:65}.sc66{--p:66}.sc67{--p:67}.sc68{--p:68}.sc69{--p:69}.sc70{--p:70}.sc71{--p:71}.sc72{--p:72}.sc73{--p:73}.sc74{--p:74}.sc75{--p:75}.sc76{--p:76}.sc77{--p:77}.sc78{--p:78}.sc79{--p:79}.sc80{--p:80}.sc81{--p:81}.sc82{--p:82}.sc83{--p:83}.sc84{--p:84}.sc85{--p:85}.sc86{--p:86}.sc87{--p:87}.sc88{--p:88}.sc89{--p:89}.sc90{--p:90}.sc91{--p:91}.sc92{--p:92}.sc93{--p:93}.sc94{--p:94}.sc95{--p:95}.sc96{--p:96}.sc97{--p:97}.sc98{--p:98}.sc99{--p:99}.sc100{--p:100}.li0{--i:0}.li1{--i:1}.li2{--i:2}.li3{--i:3}.li4{--i:4}.li5{--i:5}.li6{--i:6}.li7{--i:7}.li8{--i:8}.li9{--i:9}.li10{--i:10}.li11{--i:11}.li12{--i:12}.li13{--i:13}.li14{--i:14}.li15{--i:15}
.rc-excellent{--c:#2f7d4f}.rc-good{--c:#7F011F}.rc-needs{--c:#d49a2a}.rc-weak{--c:#a89b86}
.spacer{height:1.4rem}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
esc = html.escape


_card_count = 0


def card():
    """A rounded card. Uses a keyed container (class st-key-card_N) so the styling hits only cards,
    in old and new Streamlit alike (old versions wrap *every* block in a border wrapper)."""
    global _card_count
    _card_count += 1
    try:
        return st.container(key=f"card_{_card_count}")
    except TypeError:  # very old Streamlit without container keys
        return st.container(border=True)


@st.cache_data(ttl=30)
def get_models():
    try:
        resp = client.list()
        models = resp["models"] if isinstance(resp, dict) else resp.models
        names = [m["model"] if isinstance(m, dict) else m.model for m in models]
        return names or []
    except Exception:
        return []


def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _run_js(js: str):
    """Run a tiny script in a zero-height iframe (works on old and new Streamlit)."""
    html_doc = f"<script>{js}</script>"
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=1)
    else:
        components.html(html_doc, height=0)


def speak(text: str, rate: float = 1.0):
    """Read text aloud using the browser's built-in voice (no extra library)."""
    clean = re.sub(r"https?://\S+", "", text)
    clean = re.sub(r"[*#`_>|~\[\]]", "", clean)
    safe = json.dumps(clean).replace("</", "<\\/")
    _run_js(
        "const s = window.parent.speechSynthesis; s.cancel();"
        f"const u = new SpeechSynthesisUtterance({safe}); u.rate = {rate}; s.speak(u);"
    )


def stop_speaking():
    _run_js("window.parent.speechSynthesis.cancel();")


# <rules>
SKILL_WORDS = [
    "python", "java", "javascript", "typescript", "react", "node", "flask", "django",
    "fastapi", "sql", "mysql", "postgresql", "sqlite", "mongodb", "pandas", "numpy",
    "machine learning", "deep learning", "nlp", "power bi", "tableau", "excel", "git",
    "docker", "kubernetes", "aws", "azure", "html", "css", "rest", "api", "figma",
    "tensorflow", "pytorch", "scikit-learn", "c++", "linux", "streamlit", "llm",
]
ALIASES = {
    "javascript": ["js", "node.js", "nodejs"],
    "postgresql": ["postgres"],
    "machine learning": ["ml"],
    "kubernetes": ["k8s"],
    "node": ["node.js", "nodejs"],
    "scikit-learn": ["sklearn"],
    "power bi": ["powerbi"],
    "rest": ["restful", "rest api"],
}
ACTION_VERBS = [
    "built", "developed", "designed", "implemented", "created", "led", "improved",
    "optimized", "analyzed", "deployed", "automated", "reduced", "increased", "managed",
    "launched", "integrated", "trained", "delivered",
]


def has_skill(skill: str, text_lower: str) -> bool:
    s = skill.lower().strip()
    variants = [s] + ALIASES.get(s, [])
    return any(
        re.search(rf"(?<![a-z0-9+#]){re.escape(v)}(?![a-z0-9+#])", text_lower) for v in variants if v
    )


def rule_score(text: str) -> dict:
    t = text.lower()
    words = len(text.split())
    tips = []

    # Contact (20)
    contact = 0
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        contact += 8
    else:
        tips.append("Add a professional email address.")
    if re.search(r"\+?\d[\d\s-]{8,}", text):
        contact += 4
    else:
        tips.append("Add a phone number.")
    if "linkedin" in t:
        contact += 4
    else:
        tips.append("Add your LinkedIn link.")
    if "github" in t or "portfolio" in t:
        contact += 4
    else:
        tips.append("Add your GitHub or portfolio link.")

    # Sections (25)
    sections = {
        "Education": ["education"],
        "Experience": ["experience", "internship"],
        "Projects": ["project"],
        "Skills": ["skills"],
        "Certifications": ["certif", "achievement"],
    }
    sec = 0
    for name, keys in sections.items():
        if any(k in t for k in keys):
            sec += 5
        else:
            tips.append(f"Add a clear '{name}' section.")

    # Skills (20)
    found = [w for w in SKILL_WORDS if has_skill(w, t)]
    skills = min(20, len(found) * 2)
    if skills < 14:
        tips.append("List more relevant tools and technologies in your skills.")

    # Impact (20): numbers + action verbs
    nums = [
        n for n in re.findall(r"\b\d+(?:\.\d+)?%?", text)
        if not (len(n) == 4 and n[:2] in ("19", "20"))
    ]
    verbs = [v for v in ACTION_VERBS if re.search(rf"\b{v}\b", t)]
    impact = min(10, len(nums) * 2) + min(10, len(verbs) * 2)
    if len(nums) < 4:
        tips.append("Add measurable results (numbers, percentages) to your bullet points.")
    if len(verbs) < 4:
        tips.append("Start bullet points with strong action verbs (built, optimized, led).")

    # Length (15)
    if 300 <= words <= 700:
        length = 15
    elif 200 <= words < 300 or 700 < words <= 900:
        length = 10
        tips.append("Aim for 300-700 words so the resume fits one page.")
    else:
        length = 5
        tips.append("Resume length is off. Aim for 300-700 words (one page).")

    parts = {
        "Contact & links": (contact, 20),
        "Key sections": (sec, 25),
        "Skills coverage": (skills, 20),
        "Impact & action verbs": (impact, 20),
        "Length": (length, 15),
    }
    total = sum(v[0] for v in parts.values())
    return {"total": total, "parts": parts, "tips": tips, "skills_found": found}
# </rules>


# ---------------------------------------------------------------- database (SQLite)
@lru_cache(maxsize=1)
def _db_path() -> str:
    try:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        with open(os.path.join(os.path.dirname(DB_PATH) or ".", ".w"), "w"):
            pass
        return DB_PATH
    except OSError:
        return "recruit.db"  # /data not writable: fall back to the app folder


@contextmanager
def db():
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                pw_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                desc_hash TEXT NOT NULL UNIQUE,
                required_skills TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                filename TEXT,
                name TEXT,
                email TEXT,
                score INTEGER NOT NULL DEFAULT 0,
                label TEXT,
                status TEXT NOT NULL DEFAULT 'New',
                data TEXT NOT NULL,
                resume_text TEXT,
                text_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(job_id, text_hash)
            );
            """
        )
        # older databases: add the owner column to jobs
        cols = [r["name"] for r in c.execute("PRAGMA table_info(jobs)").fetchall()]
        if "user_id" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN user_id INTEGER")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---- accounts
def _hash_pw(password: str, salt: str | None = None):
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 150_000).hex()
    return digest, salt


def create_user(name: str, email: str, password: str):
    """Returns (user dict, None) or (None, error message)."""
    email = email.strip().lower()
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            return None, "An account with this email already exists. Try signing in."
        pw_hash, salt = _hash_pw(password)
        cur = c.execute(
            "INSERT INTO users(name, email, pw_hash, salt, created_at) VALUES (?,?,?,?,?)",
            (name.strip(), email, pw_hash, salt, _now()),
        )
        return {"id": cur.lastrowid, "name": name.strip(), "email": email}, None


def check_login(email: str, password: str):
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if not row:
        return None
    if row["pw_hash"] == "!oauth":  # account created through Google/Apple has no password
        return None
    digest, _ = _hash_pw(password, row["salt"])
    if hmac.compare_digest(digest, row["pw_hash"]):
        return {"id": row["id"], "name": row["name"], "email": row["email"]}
    return None


# ---- social sign-in (Google / Apple via Streamlit's built-in OIDC login)
def oauth_ready(provider: str) -> bool:
    """True when this provider is configured in .streamlit/secrets.toml."""
    try:
        cfg = st.secrets["auth"]
        return hasattr(st, "login") and provider in cfg and "redirect_uri" in cfg and "cookie_secret" in cfg
    except Exception:
        return False


def oauth_user():
    """The signed-in Google/Apple identity, or None."""
    try:
        u = getattr(st, "user", None)
        if (u is not None and u.get("is_logged_in") and u.get("email")
                and str(u.get("email_verified", True)).lower() != "false"):
            email = str(u.get("email")).strip().lower()
            return {"name": u.get("name") or email.split("@")[0], "email": email}
    except Exception:
        pass
    return None


def login_with_oauth(name: str, email: str) -> dict:
    """Find the local account for this email, or create one (no password)."""
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row:
            return {"id": row["id"], "name": row["name"], "email": row["email"]}
        cur = c.execute(
            "INSERT INTO users(name, email, pw_hash, salt, created_at) VALUES (?,?,?,?,?)",
            (name.strip(), email, "!oauth", "", _now()),
        )
        return {"id": cur.lastrowid, "name": name.strip(), "email": email}


# ---- jobs and candidates
def save_job(uid: int, title: str, description: str, skills: list) -> int:
    h = hashlib.sha1(f"{uid}|{title}|{description}".encode()).hexdigest()
    with db() as c:
        row = c.execute("SELECT id FROM jobs WHERE desc_hash=?", (h,)).fetchone()
        if row:
            c.execute("UPDATE jobs SET required_skills=? WHERE id=?", (json.dumps(skills), row["id"]))
            return row["id"]
        cur = c.execute(
            "INSERT INTO jobs(title, description, desc_hash, required_skills, created_at, user_id) VALUES (?,?,?,?,?,?)",
            (title, description, h, json.dumps(skills), _now(), uid),
        )
        return cur.lastrowid


def list_jobs(uid: int) -> list:
    with db() as c:
        rows = c.execute(
            "SELECT j.*, (SELECT COUNT(*) FROM candidates WHERE job_id=j.id) AS n "
            "FROM jobs j WHERE j.user_id=? ORDER BY j.id DESC",
            (uid,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_job(job_id: int, uid: int):
    with db() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, uid)).fetchone()
    if not r:
        return None
    job = dict(r)
    job["required_skills"] = json.loads(job["required_skills"])
    return job


def delete_job(job_id: int, uid: int):
    with db() as c:
        if c.execute("SELECT 1 FROM jobs WHERE id=? AND user_id=?", (job_id, uid)).fetchone():
            c.execute("DELETE FROM candidates WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def save_candidate(job_id: int, filename: str, resume_text: str, result: dict) -> int:
    h = hashlib.sha1(resume_text.encode()).hexdigest()
    with db() as c:
        old = c.execute(
            "SELECT id, status FROM candidates WHERE job_id=? AND text_hash=?", (job_id, h)
        ).fetchone()
        status = old["status"] if old else "New"
        if old:
            c.execute("DELETE FROM candidates WHERE id=?", (old["id"],))
        cur = c.execute(
            "INSERT INTO candidates(job_id, filename, name, email, score, label, status, data, "
            "resume_text, text_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, filename, result["name"], result["email"], result["final"], result["label"],
             status, json.dumps(result), resume_text, h, _now()),
        )
        return cur.lastrowid


def list_candidates(job_id: int) -> list:
    with db() as c:
        rows = c.execute(
            "SELECT * FROM candidates WHERE job_id=? ORDER BY score DESC, id ASC", (job_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = json.loads(d["data"])
        out.append(d)
    return out


def all_candidates(uid: int) -> list:
    with db() as c:
        rows = c.execute(
            "SELECT cd.*, j.title AS job_title FROM candidates cd JOIN jobs j ON j.id=cd.job_id "
            "WHERE j.user_id=? ORDER BY cd.score DESC",
            (uid,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = json.loads(d["data"])
        out.append(d)
    return out


def set_status(cid: int, status: str):
    with db() as c:
        c.execute("UPDATE candidates SET status=? WHERE id=?", (status, cid))


def set_questions(cid: int, questions: dict):
    with db() as c:
        row = c.execute("SELECT data FROM candidates WHERE id=?", (cid,)).fetchone()
        if row:
            data = json.loads(row["data"])
            data["questions"] = questions
            c.execute("UPDATE candidates SET data=? WHERE id=?", (json.dumps(data), cid))


def delete_candidate(cid: int):
    with db() as c:
        c.execute("DELETE FROM candidates WHERE id=?", (cid,))


# ---------------------------------------------------------------- LLM (structured JSON)
def _clean_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        v = [x for x in re.split(r"[\n,;]", v)]
    out = []
    for x in v:
        s = str(x.get("name") if isinstance(x, dict) and "name" in x else x).strip(" -•*\t")
        if s and s.lower() not in {i.lower() for i in out}:
            out.append(s)
    return out


class JDProfile(BaseModel):
    title: str = ""
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    min_years: float = 0

    @field_validator("required_skills", "nice_to_have", mode="before")
    @classmethod
    def _lists(cls, v):
        return _clean_list(v)

    @field_validator("min_years", mode="before")
    @classmethod
    def _years(cls, v):
        return _to_float(v)


class Evaluation(BaseModel):
    candidate_name: str = ""
    email: str = ""
    years_experience: float = 0
    education: str = ""
    core_skills: list[str] = Field(default_factory=list)
    fit_score: int = 0
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    @field_validator("core_skills", "strengths", "concerns", mode="before")
    @classmethod
    def _lists(cls, v):
        return _clean_list(v)

    @field_validator("years_experience", mode="before")
    @classmethod
    def _years(cls, v):
        return _to_float(v)

    @field_validator("fit_score", mode="before")
    @classmethod
    def _score(cls, v):
        return int(max(0, min(100, _to_float(v))))


class Questions(BaseModel):
    technical: list[str] = Field(default_factory=list)
    behavioral: list[str] = Field(default_factory=list)
    gap_probing: list[str] = Field(default_factory=list)

    @field_validator("technical", "behavioral", "gap_probing", mode="before")
    @classmethod
    def _lists(cls, v):
        return _clean_list(v)


def _to_float(v) -> float:
    try:
        m = re.search(r"\d+(?:\.\d+)?", str(v))
        return float(m.group()) if m else 0.0
    except Exception:
        return 0.0


def llm_json(model_name: str, system: str, user: str, schema, temperature: float = 0.2):
    """Ask Ollama for JSON that matches a pydantic schema. Retries once."""
    last = None
    for _ in range(2):
        try:
            resp = client.chat(
                model=model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema.model_json_schema(),
                options={"temperature": temperature, "num_ctx": NUM_CTX},
            )
            raw = resp["message"]["content"]
            try:
                return schema.model_validate_json(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.S)
                return schema.model_validate_json(m.group() if m else "{}")
        except Exception as e:  # network, model, or JSON problems
            last = e
    raise RuntimeError(str(last))


def fit_label(score: int):
    if score >= 80:
        return "Strong fit", "strong"
    if score >= 65:
        return "Good fit", "good"
    if score >= 50:
        return "Partial fit", "partial"
    return "Low fit", "low"


def analyze_jd(model_name: str, jd_text: str, title: str, manual_skills: str) -> dict:
    profile = {"title": title.strip(), "required_skills": [], "min_years": 0.0}
    manual = _clean_list(manual_skills)
    keyword_hits = [w for w in SKILL_WORDS if has_skill(w, jd_text.lower())]
    try:
        p = llm_json(
            model_name,
            "You are a recruiter. Extract hiring requirements from a job description. Reply with JSON only.",
            f"JOB DESCRIPTION:\n{jd_text[:JD_CHARS]}\n\n"
            "Return: title (job title), required_skills (the must-have tools, languages and "
            "technologies, max 12, short names like 'Python' or 'SQL'), nice_to_have (max 6), "
            "min_years (minimum years of experience as a number, 0 for freshers).",
            JDProfile,
        )
        profile["title"] = profile["title"] or p.title
        profile["min_years"] = p.min_years
        skills = p.required_skills
    except RuntimeError:
        skills = []
    if len(skills) < 3:
        skills = skills + [k for k in keyword_hits if k.lower() not in {s.lower() for s in skills}]
    profile["required_skills"] = manual or skills[:15]
    profile["title"] = profile["title"] or "Untitled role"
    return profile


def match_skills(required: list, resume_text: str):
    t = resume_text.lower()
    matched = [s for s in required if has_skill(s, t)]
    missing = [s for s in required if s not in matched]
    return matched, missing


def blank_questions() -> dict:
    return {"technical": [], "behavioral": [], "gap_probing": []}


def generate_questions(model_name: str, profile: dict, result: dict) -> dict:
    user = (
        f"ROLE: {profile['title']}\n"
        f"REQUIRED SKILLS: {', '.join(profile['required_skills'])}\n"
        f"CANDIDATE SKILLS: {', '.join(result['core_skills'])}\n"
        f"MISSING SKILLS: {', '.join(result['missing']) or 'none'}\n"
        f"CANDIDATE SUMMARY: {result['summary']}\n"
        f"CONCERNS: {'; '.join(result['concerns']) or 'none'}\n\n"
        "Write interview questions for this candidate. technical: 5 specific questions on the "
        "required skills the candidate claims. behavioral: 3 questions on teamwork, ownership and "
        "problem solving. gap_probing: 3 questions that test the missing skills and concerns. "
        "One sentence per question."
    )
    q = llm_json(
        model_name,
        "You are a senior interviewer. Write sharp, role-specific questions. Reply with JSON only.",
        user,
        Questions,
        temperature=0.4,
    )
    return {"technical": q.technical[:5], "behavioral": q.behavioral[:3], "gap_probing": q.gap_probing[:3]}


def evaluate_candidate(model_name: str, jd_text: str, profile: dict, resume_text: str,
                       filename: str, want_questions: bool) -> dict:
    required = profile["required_skills"]
    matched, missing = match_skills(required, resume_text)
    skill_score = round(100 * len(matched) / len(required)) if required else None

    ev, ai_ok = Evaluation(), True
    try:
        ev = llm_json(
            model_name,
            "You are a strict but fair technical recruiter. Judge the candidate ONLY from the "
            "resume text and never invent facts. Reply with JSON only.",
            f"JOB TITLE: {profile['title']}\n"
            f"REQUIRED SKILLS: {', '.join(required) or 'see description'}\n"
            f"MINIMUM EXPERIENCE: {profile['min_years']} years\n\n"
            f"JOB DESCRIPTION:\n{jd_text[:JD_CHARS]}\n\n"
            f"RESUME:\n{resume_text[:RESUME_CHARS]}\n\n"
            "Return: candidate_name, email, years_experience (number, internships count half), "
            "education (highest degree and college), core_skills (up to 12 skills the candidate "
            "really has), fit_score (0-100 holistic fit for THIS job: 85+ excellent, 70-84 strong, "
            "50-69 partial, below 50 weak; for fresher roles weigh projects and internships), "
            "summary (2 sentences), strengths (3 items), concerns (up to 3 gaps versus the job).",
            Evaluation,
        )
    except RuntimeError:
        ai_ok = False

    first_line = next((l.strip() for l in resume_text.splitlines() if l.strip()), "")
    name = ev.candidate_name.strip()
    if not name or len(name) > 60:
        name = first_line[:40] if 0 < len(first_line) <= 40 else os.path.splitext(filename)[0]
    email = ev.email.strip()
    if "@" not in email:
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text)
        email = m.group() if m else ""

    if ai_ok and skill_score is not None:
        final = round(0.6 * ev.fit_score + 0.4 * skill_score)
    elif ai_ok:
        final = ev.fit_score
    else:
        final = skill_score or 0
    label, _ = fit_label(final)

    result = {
        "name": name, "email": email, "years": ev.years_experience, "education": ev.education,
        "core_skills": ev.core_skills or matched, "summary": ev.summary,
        "strengths": ev.strengths, "concerns": ev.concerns,
        "ai_score": ev.fit_score if ai_ok else None, "skill_score": skill_score,
        "matched": matched, "missing": missing, "final": final, "label": label,
        "ai_ok": ai_ok, "questions": blank_questions(),
    }
    if not ai_ok:
        result["summary"] = "AI review failed for this resume. The score uses skill matching only."
    elif want_questions:
        try:
            result["questions"] = generate_questions(model_name, profile, result)
        except RuntimeError:
            pass
    return result


def interview_kit_md(c: dict, job: dict) -> str:
    d = c["data"]
    q = d.get("questions", {})
    lines = [f"# Interview kit: {c['name']}", f"Role: {job['title']}  ", f"Score: {c['score']}/100 ({c['label']})", ""]
    lines += ["## Summary", d.get("summary", ""), ""]
    for title, key in (("Technical", "technical"), ("Behavioral", "behavioral"), ("Gap probing", "gap_probing")):
        if q.get(key):
            lines += [f"## {title} questions"] + [f"{i}. {x}" for i, x in enumerate(q[key], 1)] + [""]
    if d.get("missing"):
        lines += ["## Skills to verify", ", ".join(d["missing"]), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- charts (Altair)
def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def _style(chart):
    return (
        chart.properties(padding={"left": 18, "right": 18, "top": 6, "bottom": 10}).configure(background="rgba(0,0,0,0)")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#5b4a45", titleColor="#5b4a45", gridColor="#e6d9b8",
                        domainColor="#d8c9a3", tickColor="#d8c9a3", labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor="#5b4a45", titleColor="#5b4a45", orient="bottom", labelFontSize=12)
    )


def chart_scores(rows: list):
    rows = rows[:15]
    df = pd.DataFrame({
        "Candidate": [f"#{i} {r['name'][:22]}" for i, r in enumerate(rows, 1)],
        "Score": [r["score"] for r in rows],
        "Fit": [r["label"] for r in rows],
    })
    scale = alt.Scale(domain=list(BAND_COLORS), range=list(BAND_COLORS.values()))
    base = alt.Chart(df).encode(y=alt.Y("Candidate:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)))
    bars = base.mark_bar(cornerRadiusEnd=6, size=20).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score out of 100"),
        color=alt.Color("Fit:N", scale=scale, legend=alt.Legend(title=None)),
        tooltip=["Candidate", "Score", "Fit"],
    )
    labels = base.mark_text(align="left", dx=5, color="#2a0a12", fontWeight="bold").encode(
        x="Score:Q", text="Score:Q")
    return _style((bars + labels).properties(height=max(240, 34 * len(df) + 130)))


def chart_donut(labels: list, colors: dict):
    df = pd.DataFrame({"Group": labels})
    counts = df["Group"].value_counts().reset_index()
    counts.columns = ["Group", "Count"]
    scale = alt.Scale(domain=list(colors), range=list(colors.values()))
    ch = alt.Chart(counts).mark_arc(innerRadius=56, outerRadius=105, stroke=SAND, strokeWidth=3).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Group:N", scale=scale, legend=alt.Legend(title=None)),
        tooltip=["Group", "Count"],
    ).properties(height=340)
    return _style(ch)


def chart_gaps(rows: list):
    counts = {}
    for r in rows:
        for s in r["data"].get("missing", []):
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    df = pd.DataFrame({"Skill": list(counts), "Candidates missing it": list(counts.values())})
    df = df.sort_values("Candidates missing it", ascending=False).head(10)
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=6, color=WINE, size=20).encode(
        x=alt.X("Skill:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Candidates missing it:Q", axis=alt.Axis(tickMinStep=1)),
        tooltip=["Skill", "Candidates missing it"],
    ).properties(height=240)
    return _style(ch)


def chart_distribution(rows: list):
    names = [f"{i * 10}-{i * 10 + 9}" for i in range(9)] + ["90-100"]
    counts = {n: 0 for n in names}
    for r in rows:
        counts[names[min(r["score"] // 10, 9)]] += 1
    df = pd.DataFrame({"Score range": names, "Candidates": [counts[n] for n in names]})
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=6, color=WINE).encode(
        x=alt.X("Score range:N", sort=names, title="Score range", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Candidates:Q", axis=alt.Axis(tickMinStep=1)),
        tooltip=["Score range", "Candidates"],
    ).properties(height=270)
    return _style(ch)


def chart_jobs(rows: list):
    agg = {}
    for r in rows:
        agg.setdefault(r["job_title"], []).append(r["score"])
    df = pd.DataFrame({
        "Job": list(agg),
        "Average score": [round(sum(v) / len(v)) for v in agg.values()],
        "Candidates": [len(v) for v in agg.values()],
    })
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=6, color=WINE, size=22).encode(
        y=alt.Y("Job:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Average score:Q", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Job", "Average score", "Candidates"],
    ).properties(height=max(220, 40 * len(df) + 110))
    return _style(ch)


def chart_resume_parts(parts: dict):
    df = pd.DataFrame({
        "Section": list(parts),
        "Percent": [round(100 * got / mx) for got, mx in parts.values()],
        "Points": [f"{got}/{mx}" for got, mx in parts.values()],
    })
    base = alt.Chart(df).encode(y=alt.Y("Section:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)))
    bars = base.mark_bar(cornerRadiusEnd=6, color=WINE, size=18).encode(
        x=alt.X("Percent:Q", scale=alt.Scale(domain=[0, 100]), title="Percent of maximum"),
        tooltip=["Section", "Points", "Percent"],
    )
    labels = base.mark_text(align="left", dx=5, color="#2a0a12", fontWeight="bold").encode(
        x="Percent:Q", text="Points:N")
    return _style((bars + labels).properties(height=220))


# ---------------------------------------------------------------- UI pieces
def page_head(title: str, sub: str):
    st.markdown(
        f'<div class="hero2"><div class="eyebrow">TalentMatch AI</div>'
        f'<div class="hero-title">{esc(title)}</div><div class="hero-sub">{esc(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def chips(items, cls="plain", limit=None):
    items = list(items)[:limit] if limit else list(items)
    return "".join(f'<span class="chip {cls}">{esc(str(i))}</span>' for i in items)


def candidate_card(c: dict, rank: int) -> str:
    d = c["data"]
    _, band = fit_label(c["score"])
    meta = " &nbsp;|&nbsp; ".join(
        esc(x) for x in [c.get("email") or "", f"{d.get('years', 0):g} yrs" if d.get("years") else "",
                         d.get("education", "")[:60], c.get("status", "")] if x
    )
    sub = []
    if d.get("ai_score") is not None:
        sub.append(f"AI review {d['ai_score']}")
    if d.get("skill_score") is not None:
        sub.append(f"Skills {d['skill_score']}")
    return (
        f'<div class="cand2 {band}">'
        f'<div class="c2ring sc{c["score"]}"><div class="c2in"><b>{c["score"]}</b><small>/100</small></div></div>'
        f'<div class="c2body"><div class="c2top"><span class="c2rank">#{rank}</span>'
        f'<span class="c2name">{esc(c["name"])}</span><span class="badge">{esc(c["label"])}</span></div>'
        f'<div class="cmeta">{meta}</div>'
        f'{chips(d.get("matched", []), "ok", 8)}{chips(d.get("missing", []), "no", 6)}'
        f'<div class="csum">{esc(d.get("summary", ""))}</div>'
        f'<div class="c2sub">{esc(" | ".join(sub))}</div></div></div>'
    )


def stat_tiles(rows: list):
    scores = [r["score"] for r in rows]
    tiles = [
        (len(rows), "Candidates screened"),
        (round(sum(scores) / len(scores)) if scores else 0, "Average score"),
        (max(scores) if scores else 0, "Top score"),
        (sum(1 for s in scores if s >= 65), "Good fit or better"),
    ]
    st.markdown(
        '<div class="stats">' + "".join(f"<div class='stat'><b>{v}</b><span>{k}</span></div>" for v, k in tiles) + "</div>",
        unsafe_allow_html=True,
    )


def render_insights(rows: list):
    st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        with card():
            st.markdown('<div class="chart-title">Score by candidate</div>', unsafe_allow_html=True)
            show_chart(chart_scores(rows))
    with c2:
        with card():
            st.markdown('<div class="chart-title">Fit breakdown</div>', unsafe_allow_html=True)
            show_chart(chart_donut([r["label"] for r in rows], BAND_COLORS))
    gaps = chart_gaps(rows)
    if gaps is not None:
        with card():
            st.markdown('<div class="chart-title">Most common missing skills</div>', unsafe_allow_html=True)
            show_chart(gaps)


def _on_status(cid: int):
    set_status(cid, st.session_state[f"status_{cid}"])


def _on_delete(cid: int):
    delete_candidate(cid)


def render_candidates(rows: list, job: dict, model_name: str):
    for rank, c in enumerate(rows, 1):
        cid, d = c["id"], c["data"]
        st.markdown(candidate_card(c, rank), unsafe_allow_html=True)
        with st.expander("Details, interview questions and status"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Strengths**")
                for s in d.get("strengths") or ["Not available"]:
                    st.markdown(f"- {s}")
                st.markdown("**Concerns**")
                for s in d.get("concerns") or ["None flagged"]:
                    st.markdown(f"- {s}")
            with right:
                st.markdown("**Skills found in the resume**")
                st.markdown(chips(d.get("core_skills", []), "plain") or "None detected", unsafe_allow_html=True)
                st.markdown("**Required skills missing**")
                st.markdown(chips(d.get("missing", []), "no") or "None missing", unsafe_allow_html=True)

            q = d.get("questions") or blank_questions()
            st.markdown("---")
            if any(q.values()):
                for title, key in (("Technical questions", "technical"), ("Behavioral questions", "behavioral"),
                                   ("Gap-probing questions", "gap_probing")):
                    if q.get(key):
                        st.markdown(f'<div class="qhead">{title}</div>', unsafe_allow_html=True)
                        for i, item in enumerate(q[key], 1):
                            st.markdown(f"{i}. {item}")
            else:
                st.markdown('<span class="muted">No interview questions yet.</span>', unsafe_allow_html=True)

            a, b, cdl, e = st.columns([1.2, 1.2, 1.2, 1])
            a.selectbox("Status", STATUSES, index=STATUSES.index(c["status"]) if c["status"] in STATUSES else 0,
                        key=f"status_{cid}", on_change=_on_status, args=(cid,))
            if b.button("Generate questions" if not any(q.values()) else "Regenerate questions", key=f"q_{cid}"):
                with st.spinner("Writing questions..."):
                    try:
                        newq = generate_questions(model_name, {"title": job["title"], "required_skills": job["required_skills"]}, d)
                        set_questions(cid, newq)
                        st.rerun()
                    except RuntimeError as err:
                        st.error(f"Could not generate questions: {err}")
            cdl.download_button("Download kit", interview_kit_md(c, job).encode("utf-8"),
                                f"interview_kit_{re.sub(r'[^A-Za-z0-9]+', '_', c['name'])}.md",
                                "text/markdown", key=f"dl_{cid}")
            e.button("Delete", key=f"del_{cid}", on_click=_on_delete, args=(cid,))


def export_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Rank": i, "Name": r["name"], "Email": r["email"], "Score": r["score"], "Fit": r["label"],
            "Status": r["status"], "Matched skills": ", ".join(r["data"].get("matched", [])),
            "Missing skills": ", ".join(r["data"].get("missing", [])), "File": r["filename"],
        }
        for i, r in enumerate(rows, 1)
    ])


def render_score(res: dict):
    total = res["total"]
    label = (
        "Excellent" if total >= 85 else "Good" if total >= 70
        else "Needs work" if total >= 50 else "Weak"
    )
    ring_class = {"Excellent": "rc-excellent", "Good": "rc-good", "Needs work": "rc-needs", "Weak": "rc-weak"}[label]
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown(
            f'<div class="ring sc{int(total)} {ring_class}"><div class="ring-in">'
            f"<span>{total}</span><small>out of 100 | {label}</small></div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        with card():
            st.markdown('<div class="chart-title">Score by section</div>', unsafe_allow_html=True)
            show_chart(chart_resume_parts(res["parts"]))
    if res["tips"]:
        st.markdown('<div class="card"><b>Quick fixes</b></div>', unsafe_allow_html=True)
        for tip in res["tips"]:
            st.markdown(f"- {tip}")


# ---------------------------------------------------------------- splash + login
def splash_html() -> str:
    letters = "".join(f'<span class="li{i}">{ch}</span>' for i, ch in enumerate("TalentMatch"))
    return (
        '<div class="splash"><div class="splash-glow"></div><div class="splash-inner">'
        '<div class="mark light"><i></i><i></i></div>'
        f'<div class="splash-word">{letters}<em>AI</em></div>'
        '<div class="splash-tag">Hire smarter. Match faster.</div>'
        '<div class="splash-load"><b></b></div></div></div>'
    )


GOOGLE_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'>"
    "<path fill='#EA4335' d='M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z'/>"
    "<path fill='#4285F4' d='M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z'/>"
    "<path fill='#FBBC05' d='M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z'/>"
    "<path fill='#34A853' d='M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z'/>"
    "</svg>"
)
APPLE_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='#7F011F' d='M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3.675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701'/></svg>"
)


def social_css() -> str:
    """Puts the Google / Apple logos on the four social buttons (Streamlit adds .st-key-<key> classes)."""
    rules = ""
    for provider, svg in (("google", GOOGLE_SVG), ("apple", APPLE_SVG)):
        uri = "data:image/svg+xml," + quote(svg)
        for key in ("si", "su"):
            sel = f".st-key-{key}_{provider} button"
            rules += (
                f'{sel}, {sel}:hover {{ background-image: url("{uri}") !important; background-repeat: no-repeat !important; '
                f"background-position: 18px center !important; background-size: 20px 20px !important; }}\n"
                f"{sel}:hover {{ background-color: rgba(127,1,31,.07) !important; color: var(--wine) !important; }}\n"
            )
    return f"<style>{rules}</style>"


def social_buttons(action: str, key: str):
    verb = "Sign in" if action == "in" else "Sign up"
    for provider, label in (("google", "Google"), ("apple", "Apple")):
        if st.button(f"{verb} with {label}", key=f"{key}_{provider}", **WIDE):
            if not oauth_ready(provider):
                st.warning(
                    f"{label} sign-in is not set up yet. Add your {label} keys to .streamlit/secrets.toml "
                    "(see secrets.toml.example), restart the app, or use email below."
                )
            else:
                try:
                    st.login(provider)
                except Exception as err:  # e.g. Authlib missing or a bad secrets.toml
                    st.error(f"Could not start {label} sign-in: {err}")
    st.markdown('<div class="or">or</div>', unsafe_allow_html=True)


def render_auth():
    # no sidebar on the login screen
    st.markdown(
        "<style>section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
        "[data-testid='collapsedControl'] { display: none !important; }</style>",
        unsafe_allow_html=True,
    )
    st.markdown(social_css(), unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        bars = "".join(f'<i style="--h:{h};--n:{n}"></i>' for n, h in enumerate([35, 55, 45, 75, 62, 90, 80]))
        st.markdown(
            '<div class="hero"><div>'
            '<div class="mark light"><i></i><i></i></div>'
            f"<h1>{APP_NAME}</h1>"
            "<p>Upload resumes, compare them with the job, and see who fits best in seconds.</p>"
            "<ul><li>Ranked shortlist for every job</li><li>Matched and missing skills per candidate</li>"
            "<li>Interview questions written for you</li></ul></div>"
            f'<div class="minibars">{bars}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="auth-title">Welcome</div>'
            '<div class="auth-sub">Sign in to your account, or create one to get started.</div>',
            unsafe_allow_html=True,
        )
        tab_in, tab_up = st.tabs(["Sign in", "Sign up"])
        with tab_in:
            social_buttons("in", "si")
            st.markdown('<div class="step">Sign in with email</div>', unsafe_allow_html=True)
            with st.form("signin_form"):
                email = st.text_input("Email", placeholder="you@company.com", key="si_email")
                pw = st.text_input("Password", type="password", key="si_pw")
                go = st.form_submit_button("Sign in", type="primary", **WIDE)
            if go:
                user = check_login(email, pw) if email and pw else None
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Email or password is not correct. Check them and try again.")
        with tab_up:
            social_buttons("up", "su")
            st.markdown('<div class="step">Sign up with email</div>', unsafe_allow_html=True)
            with st.form("signup_form"):
                name = st.text_input("Full name", key="su_name")
                email = st.text_input("Work email", placeholder="you@company.com", key="su_email")
                pw = st.text_input("Password", type="password", help="At least 6 characters.", key="su_pw")
                pw2 = st.text_input("Confirm password", type="password", key="su_pw2")
                go = st.form_submit_button("Create account", type="primary", **WIDE)
            if go:
                if len(name.strip()) < 2:
                    st.error("Enter your full name.")
                elif not re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.-]+", email.strip()):
                    st.error("Enter a valid email address.")
                elif len(pw) < 6:
                    st.error("Use a password with at least 6 characters.")
                elif pw != pw2:
                    st.error("The two passwords do not match.")
                else:
                    user, err = create_user(name, email, pw)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.user = user
                        st.rerun()


# ---------------------------------------------------------------- state
init_db()
st.session_state.setdefault("user", None)
st.session_state.setdefault("splash_done", False)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("resume_result", None)
st.session_state.setdefault("resume_feedback", "")
st.session_state.setdefault("active_job", None)

if not st.session_state.splash_done:
    holder = st.empty()
    holder.markdown(splash_html(), unsafe_allow_html=True)
    time.sleep(3.6)
    holder.empty()
    st.session_state.splash_done = True

# No sign-in: the app opens straight after the splash animation.
# Everything is stored under one local user (re-uses the oldest existing account's data, if any).
USER = {"id": 1, "name": "Recruiter", "email": ""}
with db() as _c:
    _row = _c.execute("SELECT MIN(id) AS m FROM users").fetchone()
    if _row and _row["m"]:
        USER["id"] = _row["m"]
UID = USER["id"]

MODES = ["Screen candidates", "Candidates", "Dashboard", "Chat", "Resume score"]

# ---------------------------------------------------------------- sidebar
try:
    nav_l, nav_r = st.columns([1, 3.4], gap="large", vertical_alignment="center")
except TypeError:  # older Streamlit
    nav_l, nav_r = st.columns([1, 3.4], gap="large")
with nav_l:
    st.markdown(
        '<div class="brand"><div class="mark"><i></i><i></i></div><span>TalentMatch<em>AI</em></span></div>',
        unsafe_allow_html=True,
    )
with nav_r:
    mode = st.radio(
        "Mode",
        MODES,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="top_mode",
    )

with st.sidebar:
    st.markdown('<div class="side-title">Settings</div>', unsafe_allow_html=True)

    models = get_models()
    if not models:
        st.warning(
            f"No model available yet at {OLLAMA_HOST}. If the app just started, the model "
            f"({DEFAULT_MODEL}) may still be downloading. Refresh in a minute."
        )
        st.stop()

    st.markdown('<div class="side-label">Model</div>', unsafe_allow_html=True)
    default_idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    model = st.selectbox("Model", models, index=default_idx, label_visibility="collapsed")
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)

    pdf = None
    speak_on, speech_rate = False, 1.0
    if mode in ("Chat", "Resume score"):
        st.markdown('<div class="side-label">Document</div>', unsafe_allow_html=True)
        pdf = st.file_uploader("Upload resume / PDF", type="pdf", label_visibility="collapsed")

        st.markdown('<div class="side-label">Voice</div>', unsafe_allow_html=True)
        speak_on = st.toggle("Speak replies", value=False)
        speech_rate = st.slider("Voice speed", 0.7, 1.5, 1.0, 0.1)
        v1, v2 = st.columns(2)
        last_reply = next(
            (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            "",
        )
        if v1.button("Replay", **WIDE) and last_reply:
            speak(last_reply, speech_rate)
        if v2.button("Stop", **WIDE):
            stop_speaking()

    if mode == "Chat":
        st.markdown('<div class="side-label">Chat</div>', unsafe_allow_html=True)
        if st.button("Clear chat", **WIDE):
            st.session_state.messages = []
            st.rerun()
        if st.session_state.messages:
            df = pd.DataFrame(st.session_state.messages)
            st.download_button(
                "Download chat (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                "chat_history.csv",
                "text/csv",
                **WIDE,
            )

resume_text = ""
if pdf:
    resume_text = read_pdf(pdf)

# ---------------------------------------------------------------- screen candidates
if mode == "Screen candidates":
    page_head(
        "Screen candidates",
        "Paste a job description, upload resumes, and get a ranked shortlist with the skills "
        "each person has, the skills they lack, and interview questions to ask.",
    )

    col_jd, col_cv = st.columns(2, gap="large")
    with col_jd:
        st.markdown('<div class="step">1. The job</div>', unsafe_allow_html=True)
        job_title = st.text_input("Job title", placeholder="e.g. Data Analyst (fresher)")
        jd_pdf = st.file_uploader("Job description PDF (optional)", type="pdf", key="jd_pdf")
        jd_typed = st.text_area("Or paste the job description", height=190,
                                placeholder="Responsibilities, required skills, experience...")
        must_have = st.text_input(
            "Must-have skills (optional)",
            placeholder="Python, SQL, Power BI",
            help="Comma separated. Leave empty and the AI reads them from the job description.",
        )
    with col_cv:
        st.markdown('<div class="step">2. The resumes</div>', unsafe_allow_html=True)
        cv_files = st.file_uploader("Resume PDFs", type="pdf", accept_multiple_files=True, key="cv_files")
        want_q = st.toggle("Write interview questions for each candidate", value=True,
                           help="Adds one extra AI call per resume. Turn off for faster screening.")
        st.caption("Scoring blends the AI's review (60%) with how many required skills appear in the resume (40%).")
        if cv_files:
            st.caption(f"{len(cv_files)} resume(s) ready.")

    run = st.button("Screen resumes", type="primary")

    if run:
        jd_text = jd_typed.strip()
        if jd_pdf is not None:
            try:
                jd_text = (read_pdf(io.BytesIO(jd_pdf.getvalue())).strip() + "\n" + jd_text).strip()
            except Exception:
                st.error("Could not read the job description PDF.")
        if len(jd_text) < 40:
            st.error("Add a job description (paste it or upload a PDF) with at least a few lines.")
        elif not cv_files:
            st.error("Upload at least one resume PDF.")
        else:
            with st.status("Screening resumes...", expanded=True) as status:
                status.write("Reading the job description")
                profile = analyze_jd(model, jd_text, job_title, must_have)
                job_id = save_job(UID, profile["title"], jd_text, profile["required_skills"])
                status.write(f"Role: {profile['title']}. Required skills: "
                             f"{', '.join(profile['required_skills']) or 'none detected'}")
                bar = st.progress(0.0)
                done, skipped = 0, []
                for i, f in enumerate(cv_files):
                    status.write(f"Reviewing {f.name} ({i + 1} of {len(cv_files)})")
                    try:
                        text = read_pdf(io.BytesIO(f.getvalue())).strip()
                    except Exception:
                        text = ""
                    if len(text) < 80:
                        skipped.append(f.name)
                    else:
                        res = evaluate_candidate(model, jd_text, profile, text, f.name, want_q)
                        save_candidate(job_id, f.name, text, res)
                        done += 1
                    bar.progress((i + 1) / len(cv_files))
                status.update(label=f"Screened {done} resume(s)", state="complete", expanded=False)
            if skipped:
                st.warning("No readable text in: " + ", ".join(skipped) + ". Scanned PDFs are not supported.")
            if done:
                st.session_state.active_job = job_id

    job_id = st.session_state.active_job
    job = get_job(job_id, UID) if job_id else None
    if job:
        rows = list_candidates(job["id"])
        if rows:
            st.markdown("---")
            st.markdown(f'<div class="section-title">Shortlist for {esc(job["title"])}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="muted">Required skills: {esc(", ".join(job["required_skills"]) or "none detected")}</div>',
                unsafe_allow_html=True,
            )
            stat_tiles(rows)
            render_insights(rows)
            st.download_button("Download ranking (CSV)", export_df(rows).to_csv(index=False).encode("utf-8"),
                               "ranking.csv", "text/csv")
            render_candidates(rows, job, model)

# ---------------------------------------------------------------- candidates database
elif mode == "Candidates":
    page_head("Candidates", "Every screened candidate is saved here, grouped by job. Filter, change status, or export.")
    jobs = list_jobs(UID)
    if not jobs:
        st.markdown(
            '<div class="card">No candidates yet. Open <b>Screen candidates</b>, add a job and upload resumes '
            "to build your first shortlist.</div>",
            unsafe_allow_html=True,
        )
    else:
        labels = {j["id"]: f"{j['title']} ({j['n']} candidates, {j['created_at']})" for j in jobs}
        ids = [j["id"] for j in jobs]
        start = ids.index(st.session_state.active_job) if st.session_state.active_job in ids else 0
        job_id = st.selectbox("Job", ids, index=start, format_func=lambda i: labels[i])
        job = get_job(job_id, UID)
        rows = list_candidates(job_id)

        f1, f2, f3 = st.columns([1, 1, 1])
        min_score = f1.slider("Minimum score", 0, 100, 0)
        statuses = f2.multiselect("Status", STATUSES, default=STATUSES)
        query = f3.text_input("Search name or skill")
        shown = [
            r for r in rows
            if r["score"] >= min_score and r["status"] in statuses
            and (not query or query.lower() in (r["name"] + " " + " ".join(r["data"].get("core_skills", []))).lower())
        ]
        stat_tiles(rows)
        if shown:
            render_insights(shown)
            st.download_button("Download ranking (CSV)", export_df(shown).to_csv(index=False).encode("utf-8"),
                               "ranking.csv", "text/csv")
            render_candidates(shown, job, model)
        else:
            st.info("No candidates match these filters.")

        with st.expander("Job details"):
            st.markdown(f"**Required skills:** {', '.join(job['required_skills']) or 'none'}")
            st.text_area("Job description", job["description"], height=200, disabled=True, key=f"jd_view_{job_id}")
            sure = st.checkbox("I want to delete this job and all its candidates", key=f"sure_{job_id}")
            if st.button("Delete job", disabled=not sure, key=f"deljob_{job_id}"):
                delete_job(job_id, UID)
                if st.session_state.active_job == job_id:
                    st.session_state.active_job = None
                st.rerun()

# ---------------------------------------------------------------- dashboard
elif mode == "Dashboard":
    page_head("Dashboard", "A quick look at your whole hiring pipeline across every job.")
    allrows = all_candidates(UID)
    if not allrows:
        st.markdown(
            '<div class="card">Nothing to chart yet. Screen some resumes and your numbers will show up here.</div>',
            unsafe_allow_html=True,
        )
    else:
        stat_tiles(allrows)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with card():
                st.markdown('<div class="chart-title">Pipeline by status</div>', unsafe_allow_html=True)
                show_chart(chart_donut([r["status"] for r in allrows], STATUS_COLORS))
        with c2:
            with card():
                st.markdown('<div class="chart-title">Fit breakdown</div>', unsafe_allow_html=True)
                show_chart(chart_donut([r["label"] for r in allrows], BAND_COLORS))
        with card():
            st.markdown('<div class="chart-title">How scores are spread</div>', unsafe_allow_html=True)
            show_chart(chart_distribution(allrows))
        with card():
            st.markdown('<div class="chart-title">Average score by job</div>', unsafe_allow_html=True)
            show_chart(chart_jobs(allrows))
        gaps = chart_gaps(allrows)
        if gaps is not None:
            with card():
                st.markdown('<div class="chart-title">Most common missing skills</div>', unsafe_allow_html=True)
                show_chart(gaps)

# ---------------------------------------------------------------- chat mode
elif mode == "Chat":
    context = resume_text[:8000]  # keep prompt small for small models
    if pdf:
        st.sidebar.success(f"PDF loaded ({len(resume_text)} chars)")

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    empty_slot = st.empty()
    if not st.session_state.messages:
        empty_slot.markdown(
            '<div class="empty"><div class="big">What can I help you with?</div>'
            '<div class="small">Ask anything, upload a PDF to chat with it, '
            "or switch to Screen candidates.</div></div>",
            unsafe_allow_html=True,
        )

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🍷"):
            st.markdown(m["content"])

    if prompt := st.chat_input("Message your assistant..."):
        empty_slot.empty()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        history = list(st.session_state.messages)
        if context:
            history.insert(0, {
                "role": "system",
                "content": f"Answer using this document when relevant:\n\n{context}",
            })

        with st.chat_message("assistant", avatar="🍷"):
            def stream():
                for chunk in client.chat(
                    model=model,
                    messages=history,
                    stream=True,
                    options={"temperature": temperature},
                ):
                    yield chunk["message"]["content"]

            reply = st.write_stream(stream())
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if speak_on:
            speak(reply, speech_rate)

# ---------------------------------------------------------------- resume score
else:
    page_head(
        "Resume score",
        "Check how complete and strong a resume is, without a job description. "
        "Upload the PDF in the sidebar, then press the button.",
    )

    if not resume_text.strip():
        st.markdown(
            '<div class="card">No resume loaded yet. Upload a PDF from the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        role = st.text_input("Target role (optional)", placeholder="e.g. Data Analyst, Frontend Developer")
        if st.button("Score my resume", type="primary"):
            res = rule_score(resume_text)
            st.session_state.resume_result = res
            render_score(res)

            st.markdown("### AI feedback")
            prompt = (
                f"Target role: {role or 'general software / data roles'}\n\n"
                f"Resume:\n{resume_text[:6000]}\n\n"
                "Give: 1) Top 3 strengths 2) Top 5 specific improvements "
                "3) A one-line verdict. Short bullet points, no preamble."
            )

            def fb_stream():
                for chunk in client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict but helpful resume reviewer."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                    options={"temperature": min(temperature, 0.5)},
                ):
                    yield chunk["message"]["content"]

            feedback = st.write_stream(fb_stream())
            st.session_state.resume_feedback = feedback
            if speak_on:
                speak(
                    f"Your resume score is {res['total']} out of 100. {feedback}",
                    speech_rate,
                )
        elif st.session_state.resume_result:
            render_score(st.session_state.resume_result)
            if st.session_state.resume_feedback:
                st.markdown("### AI feedback")
                st.markdown(st.session_state.resume_feedback)