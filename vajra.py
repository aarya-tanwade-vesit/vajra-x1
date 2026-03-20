import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import os

# ----------------------------------------------------------------------
# 🏗️ CONFIGURATION & CSS
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
# 🧠 BRAIN: LOAD THE VAJRA-X1 CORE
# ----------------------------------------------------------------------
@st.cache_resource
def load_vajra_core():
    path = os.path.join(os.path.dirname(__file__), "deployed_models")
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
    st.error(f"Failed to load AI Models. Error: {e}")
    core_loaded = False

# ----------------------------------------------------------------------
# 📊 LOGIC: DUAL-LAYER RISK ENGINE
# ----------------------------------------------------------------------
def predict_unified_risk(sensor_df):
    scaled_input = scaler.transform(sensor_df)
    no_fail_idx = list(decoder.classes_).index('No Failure')
    
    probs = champion_lgbm.predict_proba(scaled_input)[0]
    p_healthy = probs[no_fail_idx]
    p_fail_raw = 1.0 - p_healthy
    
    calibrated_risk = calibrator.predict([p_fail_raw])[0]
    is_anomaly = (lof.predict(scaled_input)[0] == -1)
    anomaly_penalty = 0.15 if is_anomaly else 0.0
    
    final_score = np.clip(calibrated_risk + anomaly_penalty, 0.0, 1.0)
    predicted_type = decoder.inverse_transform([np.argmax(probs)])[0]
    
    return final_score * 100.0, predicted_type, is_anomaly

# ----------------------------------------------------------------------
# 🖥️ UI GENERATOR
# ----------------------------------------------------------------------
def render_dashboard_pane(t_air, t_proc, rpm, torque, wear, m_type):
    # Feature Engineering
    temp_diff = t_proc - t_air
    power = rpm * torque * (2 * np.pi / 60)
    overstrain = wear * torque
    type_l = 1 if m_type == 'L' else 0
    type_m = 1 if m_type == 'M' else 0
    
    feature_dict = {
        'Air_temperature': t_air, 'Process_temperature': t_proc,
        'Rotational_speed': rpm, 'Torque': torque, 'Tool_wear': wear,
        'Temperature_Difference': temp_diff, 'Power_Generated': power,
        'Overstrain_Metric': overstrain, 'Type_L': type_l, 'Type_M': type_m
    }
    input_df = pd.DataFrame([feature_dict])
    risk_score, fail_type, anomalous = predict_unified_risk(input_df)

    col_gauge, col_alerts = st.columns([1, 1])
    
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_score,
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
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)

    with col_alerts:
        st.write("### Diagnostics")
        if risk_score < 30:
            st.success(f"🟢 **STATUS: OPTIMAL** | Operations Normal")
        elif risk_score < 70:
            st.warning(f"🟡 **STATUS: WARNING** | Higher than normal stress detected")
        else:
            st.error(f"🔴 **STATUS: CRITICAL** | Impending Failure: **{fail_type}** | Initiate Maintenance!")

        if anomalous:
            st.info("ℹ️ **ABNORMALITY DETECTED:** Unsupervised Sensors report strange patterns. The Safety Net is Active. (+15% Risk)")
            
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Financial Exposure", f"₹{int(risk_score * 150)}/hr", delta="High Risk" if risk_score > 70 else "Safe", delta_color="inverse")
    m2.metric("Estimated Tool Life", f"{int(max(0, 250 - wear))} min", delta="-5 min/hr")
    m3.metric("Mechanical Power", f"{int(power)}W", delta=f"{int((power/120000)*100)}% Load")

# ----------------------------------------------------------------------
# 🖥️ MAIN LAYOUT
# ----------------------------------------------------------------------
st.title("🛡️ VAJRA-X1: Industrial AI Command Center")

if core_loaded:
    tab_stream, tab_manual = st.tabs(["🏭 Live Factory Stream (Demo)", "🎛️ Manual Override"])

    # --- TAB 1: LIVE STREAMING ---
    with tab_stream:
        st.markdown("### 📡 Continuous CNC Telemetry Stream")
        st.write("Simulating a realistic machine degradation cycle over time.")
        
        start_stream = st.button("▶️ Start Live Stream", use_container_width=True)
        stream_placeholder = st.empty()
        
        if start_stream:
            # Baseline values
            s_air = 298.0
            s_proc = 308.2
            s_rpm = 1530.0
            s_torque = 38.0
            s_wear = 20.0
            
            for step in range(1, 51):
                with stream_placeholder.container():
                    st.write(f"⏱️ **Time Sequence:** T+{step * 3} seconds")
                    
                    # Introduce degradation over time
                    s_proc += np.random.uniform(0.1, 0.4) # Heats up
                    s_torque += np.random.uniform(0.5, 1.5) # Harder to turn
                    s_rpm -= np.random.uniform(5.0, 15.0) # Slows down
                    s_wear += np.random.uniform(2.0, 5.0) # Tool wears out rapidly
                    
                    render_dashboard_pane(s_air, s_proc, s_rpm, s_torque, s_wear, 'H')
                
                time.sleep(0.4) # Speed of the simulation tick
                
            with stream_placeholder.container():
                st.error("⏹️ **SIMULATION ENDED:** Machine reached critical failure status.")

    # --- TAB 2: MANUAL CONTROL ---
    with tab_manual:
        st.markdown("### ⚙️ Interactive Condition Testing")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            t_air = st.slider("Air Temperature [K]", 295.0, 305.0, 300.0)
            t_proc = st.slider("Process Temp [K]", 305.0, 315.0, 308.0)
        with col_s2:
            rpm = st.slider("Rotational Speed [rpm]", 1300, 2900, 1500)
            torque = st.slider("Torque [Nm]", 5.0, 75.0, 40.0)
        with col_s3:
            wear = st.slider("Tool Wear [min]", 0, 250, 50)
            m_type = st.selectbox("Machine Type", ['L', 'M', 'H'])
        
        st.markdown("---")
        render_dashboard_pane(t_air, t_proc, rpm, torque, wear, m_type)
