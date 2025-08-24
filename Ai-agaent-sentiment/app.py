import os
import time
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline

# =========================
# --------- CONFIG --------
# =========================

st.set_page_config(
    page_title="Support Sentiment Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Minimal CSS polish (chips, header, cards) ---
st.markdown("""
<style>
.big-number{font-size:2.2rem;font-weight:700;margin:0}
.stat-card{background:#0b1220;border:1px solid #1f2b4a;border-radius:16px;padding:16px}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:0.85rem;font-weight:600}
.badge-anger{background:#331515;color:#ff6b6b;border:1px solid #ff6b6b}
.badge-fear{background:#1b2133;color:#f6c177;border:1px solid #f6c177}
.badge-sadness{background:#1b2133;color:#89b4fa;border:1px solid #89b4fa}
.badge-disgust{background:#1b1f17;color:#a6e3a1;border:1px solid #a6e3a1}
.badge-joy{background:#172a1a;color:#a7f3d0;border:1px solid #34d399}
.badge-surprise{background:#1e2235;color:#f5c2e7;border:1px solid #f5c2e7}
.badge-neutral{background:#1f2432;color:#cbd5e1;border:1px solid #475569}
.rule{border-top:1px solid #2a3556;margin:12px 0}
</style>
""", unsafe_allow_html=True)

# =========================
# ------- CACHING ---------
# =========================

@st.cache_resource(show_spinner="Loading emotion model…")
def load_emotion_pipeline():
    # DistilRoBERTa fine-tuned for emotions (multi-class)
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

emotion_pipe = load_emotion_pipeline()

# Labels this model commonly returns
KNOWN_LABELS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]

NEG_SET = {"anger","fear","sadness","disgust"}
POS_SET = {"joy","surprise"}
NEU_SET = {"neutral"}

# =========================
# ---- SESSION STATE ------
# =========================
if "records" not in st.session_state:
    st.session_state.records = []  # list of dicts: {timestamp,user,channel,message,emotion,score}

if "stream_on" not in st.session_state:
    st.session_state.stream_on = False

# =========================
# ------ HELPERS ----------
# =========================

def classify(msg: str) -> Tuple[str, float]:
    if not msg or not msg.strip():
        return "neutral", 0.0
    out = emotion_pipe(msg)[0]
    label = out.get("label","neutral").lower()
    score = float(out.get("score",0.0))
    # Normalize unknown labels
    if label not in KNOWN_LABELS:
        label = "neutral"
    return label, score

def polarity(label: str) -> str:
    if label in NEG_SET: return "negative"
    if label in POS_SET: return "positive"
    return "neutral"

def badge(label: str) -> str:
    return f"""<span class="badge badge-{label}">{label.capitalize()}</span>"""

def add_record(message:str, user:str="user", channel:str="chat", ts:datetime=None):
    if ts is None: ts = datetime.utcnow()
    lbl, score = classify(message)
    st.session_state.records.append({
        "timestamp": ts,
        "user": user,
        "channel": channel,
        "message": message,
        "emotion": lbl,
        "polarity": polarity(lbl),
        "score": score
    })

def to_df() -> pd.DataFrame:
    if not st.session_state.records:
        return pd.DataFrame(columns=["timestamp","user","channel","message","emotion","polarity","score"])
    df = pd.DataFrame(st.session_state.records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def rolling_alert(df: pd.DataFrame, window_n: int, threshold: float) -> Tuple[bool, float]:
    """Return (alert_on, last_negative_ratio) using last N messages."""
    if df.empty: 
        return (False, 0.0)
    last = df.tail(window_n)
    neg_ratio = (last["polarity"] == "negative").mean() if len(last)>0 else 0.0
    return (neg_ratio >= threshold, float(neg_ratio))

# =========================
# ------ SIDEBAR ----------
# =========================

st.sidebar.title("⚙️ Controls")

window_n = st.sidebar.slider("Rolling window (messages)", min_value=5, max_value=50, value=10, step=1)
threshold = st.sidebar.slider("Alert threshold (negative ratio)", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Upload CSV with a 'message' column to bulk analyze.")
uploaded = st.sidebar.file_uploader("Upload chats CSV", type=["csv"])

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        if "message" not in df_up.columns:
            st.sidebar.error("CSV must contain a 'message' column.")
        else:
            with st.spinner("Analyzing uploaded messages…"):
                for _, row in df_up.iterrows():
                    add_record(
                        message=str(row["message"]),
                        user=str(row.get("user","user")),
                        channel=str(row.get("channel","chat")),
                        ts=pd.to_datetime(row.get("timestamp", datetime.utcnow()))
                    )
            st.sidebar.success(f"Added {len(df_up)} messages.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("Clear session data"):
    st.session_state.records = []
    st.sidebar.success("Cleared.")

# =========================
# -------- HEADER ---------
# =========================

left, mid, right = st.columns([0.7,0.15,0.15])
with left:
    st.markdown("## 💬 Support Sentiment Agent")
    st.caption("Real-time emotion tagging of tickets & live chats with spike alerts.")
with mid:
    st.write("")
with right:
    st.write("")

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

# =========================
# ------- LIVE INPUT ------
# =========================

c1, c2 = st.columns([0.65, 0.35])

with c1:
    st.subheader("🟢 Live Monitor")
    msg = st.text_input("Type/paste a customer message and press **Analyze**:", placeholder="e.g., I’m frustrated—my order is still not delivered.")
    col_analyze, col_sim = st.columns([0.3,0.7])
    if col_analyze.button("Analyze"):
        if msg.strip():
            add_record(msg)
            st.success("Analyzed.")
        else:
            st.warning("Please enter a message.")

    # Demo streamer: pick lines from built-in sample if user wants
    def toggle_stream():
        st.session_state.stream_on = not st.session_state.stream_on

    if col_sim.toggle("Simulate stream from sample data", value=False, key="simtoggle"):
        if st.button("Start stream" if not st.session_state.stream_on else "Stop stream", on_click=toggle_stream):
            pass

        if st.session_state.stream_on:
            # tiny in-app sample
            samples = [
                "I have been on hold forever. Horrible service!",
                "Thanks a lot! The agent solved my problem quickly.",
                "I’m confused about the subscription tiers.",
                "This is unacceptable. I need a refund.",
                "Amazing! The new update is great."
            ]
            placeholder = st.empty()
            for s in samples:
                if not st.session_state.stream_on:
                    break
                add_record(s)
                placeholder.info(f"Streaming: {s}")
                time.sleep(1.0)
                st.experimental_rerun()

with c2:
    # ======= KPIs =======
    df = to_df()
    total = len(df)
    neg = int((df["polarity"] == "negative").sum()) if total else 0
    pos = int((df["polarity"] == "positive").sum()) if total else 0
    neu = total - neg - pos

    st.subheader("📈 KPIs")
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown('<div class="stat-card"><div>Messages</div><div class="big-number">{}</div></div>'.format(total), unsafe_allow_html=True)
    with k2: st.markdown('<div class="stat-card"><div>Negative</div><div class="big-number">{}</div></div>'.format(neg), unsafe_allow_html=True)
    with k3: st.markdown('<div class="stat-card"><div>Neutral</div><div class="big-number">{}</div></div>'.format(neu), unsafe_allow_html=True)
    with k4: st.markdown('<div class="stat-card"><div>Positive</div><div class="big-number">{}</div></div>'.format(pos), unsafe_allow_html=True)

    # Alert
    alert_on, ratio = rolling_alert(df, window_n, threshold)
    st.markdown("### 🔔 Alert")
    st.progress(min(1.0, ratio))
    st.caption(f"Negative ratio in last {window_n} messages: **{ratio:.0%}** (threshold {threshold:.0%})")
    if alert_on:
        st.error("⚠️ Negative emotions are spiking. Notify the support lead.")
        # Optional Slack webhook: set SLACK_WEBHOOK env var in deployment
        hook = os.getenv("SLACK_WEBHOOK", "")
        if hook:
            try:
                import json, urllib.request
                payload = {"text": f"ALERT: Negative emotions ratio {ratio:.0%} in last {window_n} messages."}
                req = urllib.request.Request(hook, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
                urllib.request.urlopen(req, timeout=4)
                st.caption("Sent Slack alert.")
            except Exception:
                st.caption("Slack alert failed (check webhook).")

# =========================
# ------ TABLE + CHARTS ---
# =========================

st.markdown("### 🗂️ Labeled Messages")
if df.empty:
    st.info("No messages yet. Analyze one above or upload a CSV from the sidebar.")
else:
    # Recent table
    st.dataframe(df.sort_values("timestamp").tail(200), use_container_width=True)

    # Emotion distribution chart
    st.markdown("#### Emotion Distribution")
    emo_counts = df["emotion"].value_counts().reset_index()
    emo_counts.columns = ["emotion","count"]
    fig = px.bar(emo_counts, x="emotion", y="count", text="count")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Timeline chart
    st.markdown("#### Timeline (Negative vs Others)")
    df_tl = df.copy().sort_values("timestamp")
    df_tl["neg_bin"] = (df_tl["polarity"] == "negative").astype(int)
    df_tl["idx"] = range(1, len(df_tl)+1)
    fig2 = px.line(df_tl, x="idx", y="neg_bin", markers=True)
    fig2.update_yaxes(title="Negative (1) / Other (0)", range=[-0.05,1.05])
    fig2.update_xaxes(title="Message #")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# ----- EXPORT & ABOUT ----
# =========================

colx, coly = st.columns([0.5,0.5])
with colx:
    st.markdown("### ⬇️ Export")
    if not df.empty:
        out_csv = df.sort_values("timestamp").to_csv(index=False).encode("utf-8")
        st.download_button("Download labeled CSV", out_csv, file_name="labeled_chats.csv", mime="text/csv")

with coly:
    st.markdown("### ℹ️ Notes")
    st.write("""
- Emotions: anger, disgust, fear, joy, neutral, sadness, surprise.
- Alert triggers when the **negative ratio** in the last *N* messages ≥ threshold.
- Tweak window/threshold in the sidebar.
- Optional Slack alerts via **SLACK_WEBHOOK** environment variable.
""")
