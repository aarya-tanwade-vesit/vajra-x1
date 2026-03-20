import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import os

# ----------------------------------------------------------------------
# 🏗️ CONFIGURATION & CSS (The "SaaS Corporate" Look)
# ----------------------------------------------------------------------
st.set_page_config(page_title="Vajra-X1 Command Center", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stAlert { border-radius: 10px; }
    h1, h2, h3 { color: #00ffcc !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 🧠 BRAIN: LOAD THE VAJRA-X1 CORE v2
# ----------------------------------------------------------------------
@st.cache_resource
def load_vajra_core():
    # Load from the finalized deployment folder
    path = os.path.join(os.path.dirname(__file__), "deployed_models")
    
    # Use the SINGLE champion model we agreed upon!
    champion_lgbm = joblib.load(os.path.join(path, 'vajra_champion_lgbm.pkl'))
    calibrator = joblib.load(os.path.join(path, 'vajra_calibrator.pkl'))
    lof_model = joblib.load(os.path.join(path, 'vajra_anomaly_detector.pkl'))
    scaler = joblib.load(os.path.join(path, 'sensor_scaler.pkl'))
    label_decoder = joblib.load(os.path.join(path, 'label_decoder.pkl'))
    
    return champion_lgbm, calibrator, lof_model, scaler, label_decoder

try:
    champion_lgbm, calibrator, lof, scaler, decoder = load_vajra_core()
    core_loaded = True
except Exception as e:
    st.error(f"Failed to load AI Models. Please ensure you have generated the files in 'deployed_models'. Error: {e}")
    core_loaded = False

# ----------------------------------------------------------------------
# 📊 LOGIC: DUAL-LAYER RISK ENGINE
# ----------------------------------------------------------------------
def predict_unified_risk(sensor_df):
    """Calculates the 0-100% smoothed risk score."""
    # 1. Scale
    scaled_input = scaler.transform(sensor_df)
    
    # 2. Probability of FAILURE from Champion
    no_fail_idx = list(decoder.classes_).index('No Failure')
    probs = champion_lgbm.predict_proba(scaled_input)[0]
    p_healthy = probs[no_fail_idx]
    p_fail_raw = 1.0 - p_healthy
    
    # 3. Calibration
    calibrated_risk = calibrator.predict([p_fail_raw])[0]
    
    # 4. Anomaly Alert (LOF Safety Net)
    is_anomaly = (lof.predict(scaled_input)[0] == -1)
    anomaly_penalty = 0.15 if is_anomaly else 0.0
    
    # Combine!
    final_score = np.clip(calibrated_risk + anomaly_penalty, 0.0, 1.0)
    
    # What type of failure does the champion think it is?
    predicted_type = decoder.inverse_transform([np.argmax(probs)])[0]
    
    return final_score * 100.0, predicted_type, is_anomaly

# ----------------------------------------------------------------------
# 🖥️ UI: THE DASHBOARD HEADER
# ----------------------------------------------------------------------
st.title("🛡️ VAJRA-X1: Industrial AI Command Center")
st.markdown("---")

if core_loaded:
    # 🏛️ Dashboard Layout (Two Columns)
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📡 Real-Time Telemetry")
        t_air = st.slider("Air Temperature [K]", 295.0, 305.0, 300.0)
        t_proc = st.slider("Process Temperature [K]", 305.0, 315.0, 308.0)
        rpm = st.slider("Rotational Speed [rpm]", 1300, 2900, 1500)
        torque = st.slider("Torque [Nm]", 5.0, 75.0, 40.0)
        wear = st.slider("Tool Wear [min]", 0, 250, 50)
        m_type = st.selectbox("Machine Type", ['L', 'M', 'H'])
        
        # Feature Engineering (Physics features!)
        temp_diff = t_proc - t_air
        power = rpm * torque * (2 * np.pi / 60)
        overstrain = wear * torque
        
        type_l = 1 if m_type == 'L' else 0
        type_m = 1 if m_type == 'M' else 0
        
        # Construct DataFrame to match exact training column names
        feature_dict = {
            'Air_temperature': t_air,
            'Process_temperature': t_proc,
            'Rotational_speed': rpm,
            'Torque': torque,
            'Tool_wear': wear,
            'Temperature_Difference': temp_diff,
            'Power_Generated': power,
            'Overstrain_Metric': overstrain,
            'Type_L': type_l,
            'Type_M': type_m
        }
        
        input_df = pd.DataFrame([feature_dict])
        
        # 🏃‍♂️ RUN THE ENGINE
        risk_score, fail_type, anomalous = predict_unified_risk(input_df)

    with col_right:
        # 📐 GAUGE: MISSION CRITICAL RISK
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "🔥 UNIFIED SYSTEM RISK %"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00ffcc" if risk_score < 70 else "#ff3300"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.1)"},
                    {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.1)"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)

        # 🚨 NOTIFICATION CENTER
        if risk_score < 30:
            st.success(f"🟢 **STATUS: OPTIMAL** | System running within safe parameters.")
        elif risk_score < 70:
            st.warning(f"🟡 **STATUS: WARNING** | Higher than normal stress detected!")
        else:
            st.error(f"🔴 **STATUS: CRITICAL** | Impending Failure Detected: **{fail_type}** | Initiate Maintenance!")

        # 🛡️ THE LOF ALERT (The Winning Feature)
        if anomalous:
            st.info("ℹ️ **ABNORMALITY DETECTED:** Unsupervised Sensors report strange patterns. The Safety Net is Active. (+15% Risk)")

    # 📊 THE BOTTOM WIDGETS (SaaS Metrics)
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Financial Exposure", f"₹{int(risk_score * 150)}/hr", delta="High Risk" if risk_score > 70 else "Safe", delta_color="inverse")
    m2.metric("Estimated Tool Life", f"{250 - wear} min", delta="-5 min/hr")
    m3.metric("Mechanical Power", f"{int(power)}W", delta=f"{int((power/120000)*100)}% Load")
