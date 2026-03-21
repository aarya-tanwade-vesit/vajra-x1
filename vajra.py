"""
vajra.py  --  Vajra-X1 | AI Predictive Maintenance Command Center
=================================================================
Dual-Layer AI:  LOF (early warning) + LightGBM (hard confirmation)
Risk Formula :  risk = min(100, 0.65*LOF + 0.35*LGBM_fail%)
LGBM Override:  if LightGBM confirms failure -> risk = 100 immediately
y_pred        : 1 ONLY when LightGBM predicts a failure class

3 Risk Stages:  NOMINAL (0-40) | WARNING (40-70) | CRITICAL (70+)
Notifications:  Email (Gmail SMTP) + SMS (Twilio REST)  on CRITICAL entry
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import io
import smtplib
import threading
import requests
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vajra-X1 | Predictive Maintenance AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #080C12; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0D1117 0%,#080C12 100%);
    border-right: 1px solid #1C2333;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    color: #00FFCC !important;
    font-weight: 600;
}
div[data-testid="stMetricLabel"] {
    color: #8B949E !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #0D1117;
    border-bottom: 1px solid #1C2333; padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #8B949E;
    border-radius: 6px 6px 0 0; padding: 0.6rem 1.4rem;
    font-size: 0.85rem; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #161B22 !important;
    color: #00FFCC !important;
    border-bottom: 2px solid #00FFCC !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#00FFCC,#0096FF) !important;
    border-radius: 4px;
}
hr { border-color: #1C2333; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Startup checks -- clear errors instead of Python tracebacks
# ---------------------------------------------------------------------------
import os, pathlib

_MODEL_FILES = [
    "deployed_models/vajra_champion_lgbm.pkl",
    "deployed_models/vajra_anomaly_detector.pkl",
    "deployed_models/vajra_calibrator.pkl",
    "deployed_models/sensor_scaler.pkl",
    "deployed_models/label_decoder.pkl",
]
_SCENARIO_FILES = [
    "scenario_osf.csv", "scenario_hdf.csv",
    "scenario_pwf.csv", "scenario_twf.csv",
    "scenario_mixed.csv"
]

_missing = [f for f in _MODEL_FILES + _SCENARIO_FILES if not pathlib.Path(f).exists()]
if _missing:
    st.error("**Missing required files:**")
    for m in _missing:
        st.code(m)
    st.markdown("""
**If running locally:** make sure you cloned the full repo including `deployed_models/` and the scenario CSVs.

**If models are missing:** run the training notebook to regenerate the `.pkl` files.

**If scenario CSVs are missing:** run `python generate_scenarios.py` (requires `sensors_data.csv` in the same folder).
    """)
    st.stop()

# ---------------------------------------------------------------------------
# Load models  (cached -- runs only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    lgbm = joblib.load('deployed_models/vajra_champion_lgbm.pkl')
    lof  = joblib.load('deployed_models/vajra_anomaly_detector.pkl')
    cal  = joblib.load('deployed_models/vajra_calibrator.pkl')
    sc   = joblib.load('deployed_models/sensor_scaler.pkl')
    le   = joblib.load('deployed_models/label_decoder.pkl')
    safe = list(le.classes_).index('No Failure')
    return lgbm, lof, cal, sc, le, safe

lgbm_model, lof_model, calibrator, scaler, label_enc, safe_idx = load_models()

FEATURES = [
    'Air_temperature', 'Process_temperature', 'Rotational_speed',
    'Torque', 'Tool_wear', 'Temperature_Difference',
    'Power_Generated', 'Overstrain_Metric', 'Type_L', 'Type_M'
]

def engineer(row):
    d = row.copy()
    d['Temperature_Difference'] = d['Process_temperature'] - d['Air_temperature']
    d['Power_Generated']        = d['Rotational_speed'] * d['Torque']
    d['Overstrain_Metric']      = d['Tool_wear'] * d['Torque']
    d['Type_L'] = (d['Machine_Type'] == 'L (Low)').astype(int)
    d['Type_M'] = (d['Machine_Type'] == 'M (Medium)').astype(int)
    return d[FEATURES]

def lof_to_pct(raw):
    """LOF decision_function >= 0.5 -> 0% risk, <= -2.0 -> 100% risk.
    The new LOF model outputs values from ~1.0 down to -13.0 instead of just +/- 0.3.
    """
    return float(max(0.0, min(100.0, (0.5 - raw) / 2.5 * 100)))

# ---------------------------------------------------------------------------
# Risk computation  (EMA smoothed, Calibrated override)
# ---------------------------------------------------------------------------
def compute_risk(lgbm_fail_prob, is_anomaly, lgbm_confirmed, prev_smooth, calibrator):
    """
    raw = Calibrated LGBM Failure Probability + 15% Anomaly Penalty. Capped at 100%.
    smoothed = asymmetric EMA  (rises fast, falls slow -> no flickering)
    """
    # 1. Base Calibrated Risk
    calibrated_base = calibrator.predict([lgbm_fail_prob])[0] * 100.0
    
    # 2. Add Anomaly Penalty
    raw = calibrated_base + (15.0 if is_anomaly else 0.0)
    
    # 3. Override
    if lgbm_confirmed:
        raw = 100.0
        
    raw = min(100.0, max(0.0, raw))
        
    alpha = 0.35 if raw > prev_smooth else 0.12
    smoothed = round(alpha * raw + (1.0 - alpha) * prev_smooth, 1)
    return round(raw, 1), smoothed

def get_stage(risk):
    if risk < 40:
        return "NOMINAL"
    if risk < 70:
        return "WARNING"
    return "CRITICAL"

# ---------------------------------------------------------------------------
# Time-to-failure estimator
# ---------------------------------------------------------------------------
def estimate_ttf(history, current_risk):
    if len(history) < 5:
        return "..."
    recent = [h['Smoothed_Risk'] for h in history[-15:]]
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent, 1)[0])
    if slope <= 0.1:
        return "Stable"
    ticks = (70 - current_risk) / slope
    if ticks <= 0:
        return "NOW"
    secs = ticks * 0.08
    return f"~{int(secs)}s" if secs < 60 else f"~{int(secs/60)}m {int(secs%60)}s"

# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------
def build_alert_html(stage, risk, failure_mode, machine, timestamp):
    color = "#FF3333" if stage in ("CRITICAL", "FAILURE") else "#FF9933"
    return f"""
<html><body style="font-family:Arial,sans-serif;background:#0E1117;color:#E0E0E0;padding:20px">
<div style="max-width:600px;margin:auto;background:#161B22;border-radius:12px;
            padding:30px;border:2px solid {color}">
  <h1 style="color:{color};margin:0 0 8px">Vajra-X1 Alert</h1>
  <h2 style="color:#FF9933;margin:0 0 20px">{stage}</h2>
  <table style="width:100%;border-collapse:collapse">
    <tr><td style="padding:8px;color:#8B949E">Machine</td>
        <td style="padding:8px;color:#00FFCC"><b>{machine}</b></td></tr>
    <tr><td style="padding:8px;color:#8B949E">Risk Score</td>
        <td style="padding:8px;color:#FF3333"><b>{risk:.1f}%</b></td></tr>
    <tr><td style="padding:8px;color:#8B949E">Failure Mode</td>
        <td style="padding:8px;color:#FF9933"><b>{failure_mode}</b></td></tr>
    <tr><td style="padding:8px;color:#8B949E">Timestamp</td>
        <td style="padding:8px">{timestamp}</td></tr>
  </table>
  <div style="margin-top:20px;padding:15px;background:#0E1117;border-radius:8px;
              border-left:4px solid {color}">
    <b style="color:{color}">Action Required</b>
    <p style="margin:8px 0 0;color:#C0C0C0">
      {"Immediate shutdown and inspection advised. LightGBM has confirmed the failure pattern."
       if stage == "CRITICAL"
       else "Elevated sensor drift detected. Schedule inspection soon."}
    </p>
  </div>
  <p style="margin-top:20px;color:#8B949E;font-size:0.8rem">
    Vajra-X1 AI Predictive Maintenance System
  </p>
</div>
</body></html>
"""

def send_email(sender_gmail, app_password, recipient, subject, html_body,
               chart_bytes=None, csv_bytes=None):
    """
    Send email via Gmail SMTP SSL.
    sender_gmail  : YOUR real Gmail address (e.g. aarya@gmail.com)
    app_password  : 16-char App Password from Google Account > Security
    recipient     : where the alert should arrive (can be same as sender)
    """
    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = sender_gmail
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))

        if chart_bytes:
            part = MIMEBase("image", "png")
            part.set_payload(chart_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            'attachment; filename="vajra_risk_chart.png"')
            msg.attach(part)

        if csv_bytes:
            csv_part = MIMEBase("text", "csv")
            csv_part.set_payload(csv_bytes)
            encoders.encode_base64(csv_part)
            csv_part.add_header("Content-Disposition",
                                'attachment; filename="vajra_session_data.csv"')
            msg.attach(csv_part)

        # Gmail App Passwords can be pasted with spaces -- strip them
        clean_pwd = app_password.replace(" ", "")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as srv:
            srv.login(sender_gmail, clean_pwd)
            srv.sendmail(sender_gmail, recipient, msg.as_string())
        return True, "Sent"

    except smtplib.SMTPAuthenticationError:
        return False, (
            "Authentication failed. Make sure:\n"
            "1. You entered YOUR REAL Gmail (not a fake one)\n"
            "2. You used an APP PASSWORD (Google Account > Security > "
            "2-Step Verification > App Passwords)\n"
            "3. The App Password is 16 characters"
        )
    except smtplib.SMTPConnectError:
        return False, "Cannot connect to Gmail SMTP. Check your internet connection."
    except Exception as e:
        return False, str(e)

def send_sms(sid, token, from_num, to_num, body):
    """Twilio REST API -- no SDK needed."""
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            auth=(sid, token),
            data={"From": from_num, "To": to_num, "Body": body},
            timeout=10,
        )
        ok = resp.status_code == 201
        return ok, "Sent" if ok else resp.json().get("message", "Unknown error")
    except Exception as e:
        return False, str(e)

def generate_chart_png(history):
    hdf = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#161B22")
    ax.set_facecolor("#0E1117")
    ax.plot(hdf['Time_Second'], hdf['Smoothed_Risk'],
            color="#00FFCC", lw=2, label="Risk Score")
    ax.plot(hdf['Time_Second'], hdf['LOF'],
            color="#FF9933", lw=1.5, alpha=0.8, label="LOF Signal")
    ax.plot(hdf['Time_Second'], hdf['LGBM_Fail'],
            color="#FF3366", lw=1.5, alpha=0.8, label="LGBM Fail%")
    ax.axhline(70, color="#FF3333", ls="--", alpha=0.5, label="CRITICAL threshold")
    ax.axhline(40, color="#FFCC00", ls="--", alpha=0.4, label="WARNING threshold")
    ax.set_xlabel("Time (s)", color="#8B949E")
    ax.set_ylabel("Risk %", color="#8B949E")
    ax.set_title("Vajra-X1 Risk Timeline", color="#E0E0E0")
    ax.tick_params(colors="#8B949E")
    ax.spines[:].set_color("#1C2333")
    ax.legend(facecolor="#161B22", edgecolor="#1C2333",
               labelcolor="#E0E0E0", fontsize=8)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, facecolor="#161B22")
    plt.close(fig)
    return buf.getvalue()

def build_report_email(hdf, scenario, alert_log):
    """Rich HTML email with session summary stats and alert history table."""
    max_risk   = hdf['Risk'].max()
    avg_risk   = hdf['Risk'].mean()
    peak_mode  = hdf.loc[hdf['LGBM_Fail'].idxmax(), 'Stage'] if 'Stage' in hdf else 'N/A'
    n_alerts   = len(alert_log)
    duration   = int(hdf['Time_Second'].max() - hdf['Time_Second'].min())
    failures   = int(hdf['Pred_Failure'].sum()) if 'Pred_Failure' in hdf else 0

    alert_rows = "".join(
        f"<tr><td style='padding:6px;color:#8B949E'>{r['Time']}</td>"
        f"<td style='padding:6px'>{r['From']}</td>"
        f"<td style='padding:6px;color:#FF9933'>{r['To']}</td>"
        f"<td style='padding:6px;color:#00FFCC'>{r['Risk']}</td>"
        f"<td style='padding:6px'>{r['Mode']}</td></tr>"
        for r in alert_log
    ) or "<tr><td colspan='5' style='color:#8B949E;padding:8px'>No stage transitions recorded.</td></tr>"

    return f"""
<html><body style="font-family:Arial,sans-serif;background:#0E1117;color:#E0E0E0;padding:20px">
<div style="max-width:680px;margin:auto;background:#161B22;border-radius:12px;
            padding:30px;border:2px solid #00FFCC">
  <h1 style="color:#00FFCC;margin:0 0 4px">Vajra-X1</h1>
  <h2 style="color:#8B949E;margin:0 0 24px;font-weight:400">Session Report: {scenario}</h2>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:24px">
    <div style="background:#0E1117;border-radius:8px;padding:14px;border:1px solid #1C2333">
      <div style="color:#8B949E;font-size:0.75rem;text-transform:uppercase">Peak Risk</div>
      <div style="color:#FF3333;font-size:1.8rem;font-weight:700">{max_risk:.1f}%</div>
    </div>
    <div style="background:#0E1117;border-radius:8px;padding:14px;border:1px solid #1C2333">
      <div style="color:#8B949E;font-size:0.75rem;text-transform:uppercase">Avg Risk</div>
      <div style="color:#FFCC00;font-size:1.8rem;font-weight:700">{avg_risk:.1f}%</div>
    </div>
    <div style="background:#0E1117;border-radius:8px;padding:14px;border:1px solid #1C2333">
      <div style="color:#8B949E;font-size:0.75rem;text-transform:uppercase">Alerts Fired</div>
      <div style="color:#00FFCC;font-size:1.8rem;font-weight:700">{n_alerts}</div>
    </div>
    <div style="background:#0E1117;border-radius:8px;padding:14px;border:1px solid #1C2333">
      <div style="color:#8B949E;font-size:0.75rem;text-transform:uppercase">Failures Detected</div>
      <div style="color:#FF9933;font-size:1.8rem;font-weight:700">{failures}</div>
    </div>
    <div style="background:#0E1117;border-radius:8px;padding:14px;border:1px solid #1C2333">
      <div style="color:#8B949E;font-size:0.75rem;text-transform:uppercase">Duration</div>
      <div style="color:#E0E0E0;font-size:1.8rem;font-weight:700">{duration}s</div>
    </div>
    <div style="background:#0E1117;border-radius:8px;padding:14px;border:1px solid #1C2333">
      <div style="color:#8B949E;font-size:0.75rem;text-transform:uppercase">Peak Stage</div>
      <div style="color:#FF3366;font-size:1.8rem;font-weight:700">{peak_mode[:8]}</div>
    </div>
  </div>

  <h3 style="color:#E0E0E0;border-bottom:1px solid #1C2333;padding-bottom:8px">Stage Transition Log</h3>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
    <tr style="background:#0E1117">
      <th style="padding:8px;text-align:left;color:#8B949E">Time</th>
      <th style="padding:8px;text-align:left;color:#8B949E">From</th>
      <th style="padding:8px;text-align:left;color:#8B949E">To</th>
      <th style="padding:8px;text-align:left;color:#8B949E">Risk</th>
      <th style="padding:8px;text-align:left;color:#8B949E">Triggered By</th>
    </tr>
    {alert_rows}
  </table>

  <p style="margin-top:24px;padding:12px;background:#0E1117;border-radius:8px;
            color:#8B949E;font-size:0.8rem;border-left:3px solid #00FFCC">
    Attachments: Risk chart (PNG) + Full session data (CSV)<br>
    Generated by Vajra-X1 AI Predictive Maintenance System
  </p>
</div>
</body></html>
"""

def notify_background(email_cfg, sms_cfg, subject, html, sms_body,
                      chart_bytes, csv_bytes=None):
    """Fires in a daemon thread -- won't block the simulation."""
    if email_cfg:
        send_email(
            email_cfg["sender"], email_cfg["app_password"],
            email_cfg["recipient"], subject, html, chart_bytes, csv_bytes
        )
    if sms_cfg:
        send_sms(sms_cfg["sid"], sms_cfg["token"],
                 sms_cfg["from_num"], sms_cfg["to_num"], sms_body)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "MIXED -- Unseen Linear Degradation (Demo)": "scenario_mixed.csv",
    "OSF -- Overstrain Failure":       "scenario_osf.csv",
    "HDF -- Heat Dissipation Failure": "scenario_hdf.csv",
    "PWF -- Power Failure":            "scenario_pwf.csv",
    "TWF -- Tool Wear Failure":        "scenario_twf.csv",
}
SCENARIO_DESC = {
    "MIXED -- Unseen Linear Degradation (Demo)": ("Unknown", "A slow, realistic, linear degradation ending in a surprise failure."),
    "OSF -- Overstrain Failure":       ("Screw", "Tool Wear x Torque > 12,000 Nm.min"),
    "HDF -- Heat Dissipation Failure": ("Thermometer", "Temp Diff < 8.6K AND RPM < 1380"),
    "PWF -- Power Failure":            ("Lightning", "Power < 3500W or > 9000W"),
    "TWF -- Tool Wear Failure":        ("Wrench", "Tool wear enters 200-240 min zone"),
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Vajra-X1")
    st.caption("Industrial AI Predictive Maintenance")
    st.divider()

    scenario = st.selectbox("Scenario", list(SCENARIOS.keys()),
                            label_visibility="collapsed")
    _, desc = SCENARIO_DESC[scenario]
    st.caption(desc)
    st.divider()

    speed = st.slider("Sim Speed", 0.02, 0.25, 0.07, 0.01, format="%.2fs/row")
    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")
    st.divider()

    # ── Email Alerts ─────────────────────────────────────────────────────
    with st.expander("Alert Notifications", expanded=False):

        st.markdown("**Email (Gmail Setup)**")
        with st.popover("How to test alerts?"):
            st.markdown("""
**How to get email working in 2 mins:**
1. Open **Google Account > Security**.
2. Turn **ON** 2-Step Verification.
3. Search for **"App passwords"** at the top.
4. Select App: **Mail**, Device: **Other**.
5. Copy the 16-character password it generates.
6. Enter it below and click **Send Test Email**.
            """)

        email_on = st.toggle("Enable Email Alerts", key="email_on")
        my_gmail = st.text_input("YOUR Gmail address",
                                  placeholder="name@gmail.com", key="s_email")
        app_pwd  = st.text_input("App Password (16 chars)",
                                  placeholder="abcd efgh ijkl mnop",
                                  type="password",
                                  help="Your 16-char secret generated in Google settings (not your regular login).",
                                  key="s_pass")
        alert_to = st.text_input("Send alerts TO",
                                  placeholder="judge@company.com",
                                  key="r_email")

        if st.button("Send Test Email", use_container_width=True):
            if not my_gmail or not app_pwd or not alert_to:
                st.error("Fill in all 3 fields first.")
            else:
                with st.spinner("Connecting to Gmail..."):
                    ok, msg = send_email(
                        my_gmail, app_pwd, alert_to,
                        "[Vajra-X1] Test -- Connection OK",
                        build_alert_html("TEST", 0.0, "N/A",
                                         "Test Machine", "Connection check"),
                        chart_bytes=None,
                    )
                if ok:
                    st.success("Email sent! Check your inbox.")
                else:
                    st.error(msg)

        st.divider()
        st.markdown("**SMS (Twilio)**")
        st.caption(
            "Free tier: twilio.com/try-twilio. "
            "You get a phone number + $15 credit."
        )
        sms_on     = st.toggle("Enable SMS Alerts", key="sms_on")
        twilio_sid  = st.text_input("Account SID",  type="password", key="t_sid")
        twilio_tok  = st.text_input("Auth Token",   type="password", key="t_tok")
        twilio_from = st.text_input("From Number",  placeholder="+1415XXXXXXX", key="t_from")
        twilio_to   = st.text_input("To Number",    placeholder="+91XXXXXXXXXX", key="t_to")

    with st.expander("AI Architecture", expanded=False):
        st.markdown("""
**Layer 1: LOF** (65% weight)
Trained on 9,661 healthy records.
Smooth early-warning ramp. Recall: **60%**

**Layer 2: LightGBM** (35% weight)
98% accuracy classifier. Hard failure confirmation.

```
risk = 0.65 x LOF + 0.35 x LGBM_fail%
if LGBM confirms -> risk = 100% (override)
y_pred = 1 only if LGBM classifies failure
```
        """)

# Build config dicts (None if not configured)
email_cfg = None
if st.session_state.get("email_on") and my_gmail and app_pwd and alert_to:
    email_cfg = {"sender": my_gmail, "app_password": app_pwd, "recipient": alert_to}

sms_cfg = None
if st.session_state.get("sms_on") and twilio_sid and twilio_tok and twilio_from and twilio_to:
    sms_cfg = {"sid": twilio_sid, "token": twilio_tok,
               "from_num": twilio_from, "to_num": twilio_to}

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------
col_t, col_badge = st.columns([5, 1])
col_t.markdown("# Vajra-X1 -- Predictive Maintenance AI")
col_t.caption("Dual-layer real-time AI: LOF anomaly detection + LightGBM failure classification")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if email_cfg or sms_cfg:
        st.success("Alerts ON")
    else:
        st.info("AI ONLINE")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_signals, tab_eval, tab_arch, tab_hist = st.tabs([
    "Live Dashboard",
    "Signal Breakdown",
    "System Accuracy",
    "AI Architecture",
    "Alert History",
])

# ── Tab 1: Live Dashboard ──────────────────────────────────────────────────
with tab_live:
    alert_slot = st.empty()
    st.markdown("---")
    kc = st.columns(6)
    kpi_risk  = kc[0].empty()
    kpi_stage = kc[1].empty()
    kpi_lof   = kc[2].empty()
    kpi_lgbm  = kc[3].empty()
    kpi_ttf   = kc[4].empty()
    kpi_wear  = kc[5].empty()
    st.markdown("---")
    row1 = st.columns(2)
    row1[0].markdown("**Risk Score** (LOF + LightGBM)")
    chart_risk = row1[0].empty()
    row1[1].markdown("**Mechanical Load** (Torque & Tool Wear)")
    chart_mech = row1[1].empty()
    row2 = st.columns(2)
    row2[0].markdown("**Thermal Profile**")
    chart_temp = row2[0].empty()
    row2[1].markdown("**Rotational Speed**")
    chart_spd  = row2[1].empty()
    st.markdown("---")
    notify_slot = st.empty()

# ── Tab 2: Signal Breakdown ────────────────────────────────────────────────
with tab_signals:
    sb1, sb2 = st.columns([3, 2])
    with sb1:
        st.markdown("### LightGBM Failure Mode Probabilities")
        lgbm_chart = st.empty()
    with sb2:
        st.markdown("### LOF Anomaly Risk")
        lof_bar    = st.empty()
        st.divider()
        st.markdown("### Formula")
        st.latex(r"R = \min\!\left(100,\ \text{Calibrated LGBM}_{fail\%} + 15\%\ \text{Anomaly Penalty}\right)")
        st.latex(r"\text{Override:}\ \hat{y}_{LGBM}\neq\text{No Failure}\Rightarrow R=100\%")
        sb_w = st.columns(2)
        lof_weight_slot  = sb_w[0].empty()
        lgbm_weight_slot = sb_w[1].empty()

# ── Tab 3: Accuracy ────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### Live Accuracy vs Ground Truth")
    st.caption("y_pred = 1 when LightGBM predicts a failure class")
    ea = st.columns(3)
    eval_prec = ea[0].empty()
    eval_rec  = ea[1].empty()
    eval_f1   = ea[2].empty()
    st.markdown("---")
    ev1, ev2 = st.columns([3, 2])
    ev1.markdown("**Risk vs Ground Truth**")
    eval_tl  = ev1.empty()
    ev2.markdown("**Prediction Log**")
    eval_log = ev2.empty()

# ── Tab 4: Architecture ────────────────────────────────────────────────────
with tab_arch:
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("### Layer 1: LOF")
        with st.container(border=True):
            st.markdown("""
**Local Outlier Factor** (novelty=True)
- Trained on **9,661 healthy** machine records
- Detects sensor density drift from healthy cluster
- Smooth 0-100% continuous risk signal
- Best recall **(60%)** vs Isolation Forest & Autoencoder
""")
        st.info("Weight: **65%** of risk score")

    with a2:
        st.markdown("### Layer 2: LightGBM")
        with st.container(border=True):
            st.markdown("""
**Gradient Boosted Tree Classifier**
- Trained on full AI4I 2020 dataset (10,000 rows)
- **98% accuracy** on failure classification
- Hard prediction drives y_pred (never a threshold)
- If it fires -> **Risk overrides to 100%**
""")
        st.info("Weight: **35%** of risk score")

    with a3:
        st.markdown("### Design Rationale")
        with st.container(border=True):
            st.markdown("""
**LightGBM = Light Switch**
0% until it matches a known pattern. Then 100%.

**LOF = Dimmer Switch**
Rises from the moment sensors drift from the
healthy cluster -- catches failures *minutes*
before LightGBM detects anything.

**Together:** Early Warning + Hard Confirmation.
""")
        st.success("Benchmarked vs Isolation Forest & Autoencoder")

    st.divider()
    st.latex(r"R = \min\!\left(100,\ \text{Calibrated LGBM}_{fail\%} + 15\%\ \text{Anomaly Penalty}\right)")
    st.latex(r"\hat{y} = \mathbb{1}\!\left[\text{LightGBM class} \neq \text{No Failure}\right]")

# ── Tab 5: Alert History ───────────────────────────────────────────────────
with tab_hist:
    st.markdown("### Stage Transition Log")
    st.caption("Records every time the risk stage changes during the simulation")
    hist_table = st.empty()
    st.divider()
    st.markdown("### Download Session Data")
    dl_slot = st.empty()

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
if run_btn:
    csv_file = SCENARIOS[scenario]
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"Missing {csv_file}. Run `python generate_scenarios.py` first.")
        st.stop()

    has_gt  = 'Machine_failure' in data.columns
    n       = len(data)
    history = []
    alert_log = []
    sent_stages = set()

    # chart handles (None = not yet created)
    cr = cm = ct = cs = et = None

    prog = st.sidebar.progress(0, text="Starting...")

    with st.status(f"Simulating: {scenario}", expanded=False) as sim_box:
        for t in range(n):
            row_df   = data.iloc[[t]].copy()
            X        = engineer(row_df)
            X_scaled = scaler.transform(X)

            # LOF Safety Net Anomaly Check
            is_anomaly = (lof_model.predict(X_scaled)[0] == -1)
            raw_lof = float(lof_model.decision_function(X_scaled)[0])
            lof_pct = lof_to_pct(raw_lof) # Kept for UI display in charts

            # LightGBM Classifier
            probs      = lgbm_model.predict_proba(X_scaled)[0]
            pred_idx   = int(np.argmax(probs))
            pred_label = label_enc.inverse_transform([pred_idx])[0]
            lgbm_fail_prob = float(1.0 - probs[safe_idx])
            lgbm_fail  = lgbm_fail_prob * 100 # Kept for UI display

            fail_pairs = [(label_enc.classes_[i], float(p))
                          for i, p in enumerate(probs) if i != safe_idx]
            top_mode   = max(fail_pairs, key=lambda x: x[1])

            # Risk calculation with Calibrator
            lgbm_ok   = (pred_label != 'No Failure')
            prev_s    = history[-1]['Smoothed_Risk'] if history else 0.0
            raw_r, sm = compute_risk(lgbm_fail_prob, is_anomaly, lgbm_ok, prev_s, calibrator)
            stage     = "CRITICAL" if lgbm_ok else get_stage(sm)
            y_pred    = int(lgbm_ok)
            ttf       = estimate_ttf(history, sm)
            ts        = datetime.now().strftime("%H:%M:%S")

            rec = {
                'Time_Second':         int(row_df['Time_Second'].iloc[0]),
                'Risk':                sm,
                'Smoothed_Risk':       sm,
                'Raw_Risk':            raw_r,
                'LOF':                 round(lof_pct, 1),
                'LGBM_Fail':           round(lgbm_fail, 2),
                'Stage':               stage,
                'Pred_Failure':        y_pred,
                'Failure_Mode':        top_mode[0],
                'Machine_failure':     int(row_df['Machine_failure'].iloc[0]) if has_gt else 0,
                'Torque':              float(row_df['Torque'].iloc[0]),
                'Tool_wear':           float(row_df['Tool_wear'].iloc[0]),
                'Process_temperature': float(row_df['Process_temperature'].iloc[0]),
                'Air_temperature':     float(row_df['Air_temperature'].iloc[0]),
                'Rotational_speed':    float(row_df['Rotational_speed'].iloc[0]),
                'Timestamp':           ts,
            }
            history.append(rec)
            hdf    = pd.DataFrame(history)
            last_r = hdf.iloc[[-1]].set_index('Time_Second')

            # Stage transition logging + toast
            prev_stage = history[-2]['Stage'] if len(history) > 1 else "NOMINAL"
            if stage != prev_stage:
                alert_log.append({'Time': ts, 'Step': t,
                                   'From': prev_stage, 'To': stage,
                                   'Risk': f"{sm:.1f}%", 'Mode': top_mode[0]})
                if stage == "CRITICAL":
                    st.toast(f"🚨 CRITICAL -- {top_mode[0]} -- Risk {sm:.1f}%", icon="🚨")

            # Alert box
            STYLE = {"NOMINAL": ("success", "NOMINAL"), "WARNING": ("warning", "WARNING"),
                     "CRITICAL": ("error", "CRITICAL")}
            fn, label = STYLE.get(stage, ("error", "CRITICAL"))
            
            display_mode = "Operations Normal" if stage == "NOMINAL" else top_mode[0]
            
            getattr(alert_slot, fn)(
                f"**{label}** -- {display_mode} -- Risk {sm:.1f}% -- TTF: {ttf}"
            )

            # KPIs
            d_risk = sm - (history[-2]['Smoothed_Risk'] if len(history) > 1 else sm)
            kpi_risk.metric("Risk Score",  f"{sm:.1f}%",
                             delta=f"{d_risk:+.1f}%", delta_color="inverse")
            kpi_stage.metric("Stage",      stage)
            kpi_lof.metric(  "LOF",        f"{lof_pct:.1f}%")
            kpi_lgbm.metric( "LGBM Fail%", f"{lgbm_fail:.2f}%")
            kpi_ttf.metric(  "Est. TTF",   ttf)
            kpi_wear.metric( "Tool Wear",  f"{row_df['Tool_wear'].iloc[0]:.1f} min")

            # Charts (init on t=0, add_rows after)
            if t == 0:
                h0 = hdf.set_index('Time_Second')
                cr = chart_risk.line_chart(h0[['Risk','LOF','LGBM_Fail']],
                                           height=250, color=["#00FFCC","#FF9933","#FF3366"])
                cm = chart_mech.line_chart(h0[['Torque','Tool_wear']],
                                           height=250, color=["#FFCC00","#FF6666"])
                ct = chart_temp.line_chart(h0[['Process_temperature','Air_temperature']],
                                           height=250, color=["#66FF66","#3399FF"])
                cs = chart_spd.line_chart( h0[['Rotational_speed']],
                                           height=250, color=["#AA77FF"])
            else:
                cr.add_rows(last_r[['Risk','LOF','LGBM_Fail']])
                cm.add_rows(last_r[['Torque','Tool_wear']])
                ct.add_rows(last_r[['Process_temperature','Air_temperature']])
                cs.add_rows(last_r[['Rotational_speed']])

            # Signal Breakdown tab
            prob_df = pd.DataFrame({'Mode': label_enc.classes_, 'Prob%': probs*100}
                                   ).sort_values('Prob%', ascending=True)
            with lgbm_chart.container():
                st.bar_chart(prob_df.set_index('Mode'), horizontal=True,
                              color="#00FFCC", height=240)
                best = prob_df[prob_df['Mode'] != 'No Failure'].iloc[-1]
                st.caption(f"Top failure mode: **{best['Mode']}** -- {best['Prob%']:.4f}%")
            lof_bar.progress(int(min(lof_pct, 100)), text=f"LOF Risk: {lof_pct:.1f}%")
            lof_weight_slot.metric("LOF Weight",  "65%")
            lgbm_weight_slot.metric("LGBM Weight", "35%")

            # Accuracy tab
            if has_gt and len(hdf) > 5:
                y_true = hdf['Machine_failure'].astype(int).values
                y_pred_arr = hdf['Pred_Failure'].astype(int).values
                if len(np.unique(y_true)) == 2:
                    eval_prec.metric("Precision",
                        f"{precision_score(y_true, y_pred_arr, zero_division=0):.1%}")
                    eval_rec.metric("Recall",
                        f"{recall_score(y_true, y_pred_arr, zero_division=0):.1%}")
                    eval_f1.metric("F1 Score",
                        f"{f1_score(y_true, y_pred_arr, zero_division=0):.1%}")
                # Scale Machine_failure (0/1) -> (0/100) so it's on the same axis as Risk (%)
                hdf['GT_Failure_Pct'] = hdf['Machine_failure'] * 100
                last_r_gt = hdf.iloc[[-1]].set_index('Time_Second')
                if et is None:
                    et = eval_tl.line_chart(
                        hdf.set_index('Time_Second')[['Risk', 'GT_Failure_Pct']],
                        height=230, color=["#00FFCC", "#FF3366"]
                    )
                else:
                    et.add_rows(last_r_gt[['Risk', 'GT_Failure_Pct']])
                log_cols = ['Time_Second','Risk','LOF','LGBM_Fail','Machine_failure','Pred_Failure','Failure_Mode','Stage']
                eval_log.dataframe(hdf[log_cols].tail(12),
                                    use_container_width=True, hide_index=True)

            # Alert history tab
            if alert_log:
                hist_table.dataframe(pd.DataFrame(alert_log),
                                      use_container_width=True, hide_index=True)

            # Notifications (on CRITICAL first entry, background thread)
            if stage == "CRITICAL" and stage not in sent_stages:
                sent_stages.add(stage)
                if email_cfg or sms_cfg:
                    chart_png = generate_chart_png(history)
                    machine   = "Extruder-Alpha"
                    html_body = build_alert_html(stage, sm, top_mode[0], machine, ts)
                    subject   = f"[Vajra-X1] CRITICAL -- {top_mode[0]} -- Risk {sm:.1f}%"
                    sms_body  = (f"Vajra-X1 CRITICAL: Risk {sm:.1f}%. "
                                 f"Mode: {top_mode[0]}. Machine: {machine}. Time: {ts}.")
                    threading.Thread(
                        target=notify_background,
                        args=(email_cfg, sms_cfg, subject, html_body, sms_body, chart_png),
                        daemon=True,
                    ).start()
                    notify_slot.info("Notifications sent in background.")

            prog.progress((t + 1) / n,
                          text=f"Step {t+1}/{n}  |  Stage: {stage}  |  Risk: {sm:.1f}%")
            time.sleep(speed)

    sim_box.update(label="Simulation Complete", state="complete", expanded=False)
    st.sidebar.success("Done!")

    # Post-sim downloads + report
    chart_png = generate_chart_png(history)
    csv_bytes = hdf.to_csv(index=False).encode()

    with tab_hist:
        st.markdown("**Session complete. Export your data:**")
        dc = st.columns(3)
        dc[0].download_button(
            "Download CSV",
            data=csv_bytes, file_name="vajra_session.csv",
            mime="text/csv", use_container_width=True,
            help="Full session data: risk, LOF, LGBM, sensor readings, stage per row"
        )
        dc[1].download_button(
            "Download Chart (PNG)",
            data=chart_png, file_name="vajra_chart.png",
            mime="image/png", use_container_width=True,
            help="Risk timeline chart as high-resolution PNG"
        )
        if email_cfg:
            if dc[2].button("Email Full Report", use_container_width=True,
                             help="Sends HTML report + chart PNG + CSV data to your email"):
                with st.spinner("Sending report email..."):
                    report_html = build_report_email(hdf, scenario, alert_log)
                    ok, err = send_email(
                        email_cfg["sender"], email_cfg["app_password"],
                        email_cfg["recipient"],
                        f"[Vajra-X1] Full Session Report -- {scenario}",
                        report_html,
                        chart_bytes=chart_png,
                        csv_bytes=csv_bytes,
                    )
                if ok:
                    st.success("Report emailed with chart + CSV attached!")
                else:
                    st.error(f"Email failed: {err}")
        else:
            dc[2].info("Configure email in sidebar to send report")

        st.divider()
        # Quick stats summary
        st.markdown("**Session Summary**")
        ss = st.columns(4)
        ss[0].metric("Peak Risk",   f"{hdf['Risk'].max():.1f}%")
        ss[1].metric("Avg Risk",    f"{hdf['Risk'].mean():.1f}%")
        ss[2].metric("Failures",    str(int(hdf['Pred_Failure'].sum())))
        ss[3].metric("Stage Changes", str(len(alert_log)))


else:
    with tab_live:
        st.info("Select a scenario in the sidebar and click **Run Simulation**.")
        st.markdown("""
| Scenario | Failure Physics | Sensors to Watch |
|---|---|---|
| **OSF** | Tool Wear x Torque > 12,000 Nm.min | Torque, Tool Wear |
| **HDF** | Temp Diff < 8.6K AND RPM < 1380 | Air Temp, Process Temp, RPM |
| **PWF** | Power < 3500W or > 9000W | RPM, Torque |
| **TWF** | Tool wear enters 200-240 min zone | Tool Wear |
        """)
    with tab_hist:
        hist_table.info("Alert history appears here after a simulation runs.")
        dl_slot.info("Download options appear after simulation completes.")