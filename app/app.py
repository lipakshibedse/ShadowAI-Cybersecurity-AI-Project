# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

st.set_page_config(page_title="Shadow AI Detection", layout="wide")
st.title("🚨 AI Misuse & Shadow AI Detection Dashboard")

# ---------------------------
# Paths (safe)
base_dir = os.path.dirname(os.path.abspath(__file__))        # app/ directory
data_path = os.path.join(base_dir, '..', 'data', 'simulated_shadow_ai_logs.csv')
rf_path = os.path.join(base_dir, '..', 'models', 'rf_model.joblib')
scaler_path = os.path.join(base_dir, '..', 'models', 'scaler.joblib')

# ---------------------------
# Load dataset safely
if not os.path.exists(data_path):
    st.error("❌ Dataset missing: data/simulated_shadow_ai_logs.csv\nPlease generate or place dataset in the data/ folder.")
    st.stop()


# Connect to SQLite database
conn = sqlite3.connect(os.path.join(base_dir, "..", "data", "shadowai.db"))

# Read data directly from the database
df = pd.read_sql_query("SELECT * FROM ai_logs", conn)

conn.close()

st.success("✅ Dataset loaded successfully!")
st.sidebar.header("📁 Dataset Info")
st.sidebar.write("Columns:", list(df.columns))

# Normalize or ensure some expected columns exist for display
# (If your dataset uses different column names, adapt below names accordingly)
# We'll support: department, tool, prompt_length, sensitive_data, usage_time, risk_score
# If they don't exist, create safe defaults:
if 'department' not in df.columns:
    df['department'] = 'Unknown'
if 'tool' not in df.columns:
    df['tool'] = 'Unknown'
if 'prompt_length' not in df.columns:
    df['prompt_length'] = df.get('prompt_len', 0)
if 'sensitive_data' not in df.columns:
    df['sensitive_data'] = df.get('sensitive', 'No')
if 'usage_time' not in df.columns:
    df['usage_time'] = df.get('time', 0)
if 'risk_score' not in df.columns:
    # if no model-derived risk_score present, make a simple synthetic score using heuristics
    # (this is just for visualization; real score should come from model)
    df['risk_score'] = df['prompt_length'].apply(lambda x: min(1.0, (x/400)))  # 0..1

# Sidebar filters
st.sidebar.header("🔍 Filters")
depts = sorted(df['department'].unique().tolist())
tools = sorted(df['tool'].unique().tolist())
sel_dept = st.sidebar.multiselect("Select Department", depts, default=depts)
sel_tool = st.sidebar.multiselect("Select Tool", tools, default=tools)

filtered = df[df['department'].isin(sel_dept) & df['tool'].isin(sel_tool)]

# Top metrics
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.metric("Total records", len(filtered))
with c2:
    avg_risk = filtered['risk_score'].mean() if len(filtered)>0 else 0
    st.metric("Average risk score", f"{avg_risk:.2f}")
with c3:
    sensitive_pct = 0
    if 'sensitive_data' in filtered.columns and len(filtered)>0:
        sensitive_pct = (filtered['sensitive_data'].astype(str).str.lower().isin(['yes','true','1']).sum() / len(filtered)) * 100
    st.metric("Sensitive data %", f"{sensitive_pct:.1f}%")

st.markdown("---")

# Layout: left = table & filters, right = charts & model panel
left, right = st.columns([1.2, 1.8])

with left:
    st.subheader("📋 Filtered Logs (preview)")
    st.dataframe(filtered.head(20), height=320)

    # High-risk rule-based highlight (simple)
   # ============================
# ============================
# 🔥 High-Risk Alert Section
# ============================
st.subheader("🚨 AI-Detected High-Risk Alerts")

# Define risk detection rule
cond = (filtered['risk_score'] >= 0.8)
if 'sensitive_word_count' in filtered.columns:
    cond = cond | (filtered['sensitive_word_count'] > 3)
if 'tool' in filtered.columns:
    cond = cond | filtered['tool'].astype(str).str.lower().isin(['chatgpt', 'copilot', 'bard'])

# Filter data
alerts = filtered[cond]

# Display alerts safely
if len(alerts) > 0:
    st.success(f"✅ {len(alerts)} potential high-risk logs found!")
    cols_to_show = [col for col in ['department', 'tool', 'prompt_length', 'sensitive_word_count', 'risk_score'] if col in alerts.columns]
    st.dataframe(alerts[cols_to_show].head(10))
else:
    st.info("No high-risk alerts detected in the current filter.")


with right:
    st.subheader("📊 Visualizations")

    # Bar: Avg risk by department
    dept_group = filtered.groupby('department')['risk_score'].mean().sort_values(ascending=False)
    st.markdown("**Average risk score by Department**")
    st.bar_chart(dept_group)

    # Pie: sensitive data distribution (if exists)
    st.markdown("**Sensitive data usage**")
    if 'sensitive_data' in filtered.columns:
        sens_counts = filtered['sensitive_data'].astype(str).value_counts()
        fig1, ax1 = plt.subplots(figsize=(4,3))
        ax1.pie(sens_counts, labels=sens_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.write("No sensitive_data column.")

    # Scatter: prompt_length vs risk_score
    st.markdown("**Prompt length vs Risk Score**")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.scatter(filtered['prompt_length'], filtered['risk_score'], alpha=0.6)
    ax2.set_xlabel("prompt_length")
    ax2.set_ylabel("risk_score")
    st.pyplot(fig2)

    # Time trend (if usage_time is numeric)
    st.markdown("**Usage time trend (sample)**")
    if 'usage_time' in filtered.columns:
        try:
            usage_mean = filtered.groupby('usage_time')['risk_score'].mean().sort_index()
            st.line_chart(usage_mean)
        except Exception:
            st.write("Unable to plot usage_time trend.")
    else:
        st.write("No usage_time column.")

st.markdown("---")

# ---------------------------
# Model prediction panel (manual input)
st.subheader("🧠 AI Prediction Panel (Use trained model)")

model_loaded = False
if os.path.exists(rf_path) and os.path.exists(scaler_path):
    try:
        rf_model = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        model_loaded = True
        st.success("Model loaded. You can predict risk for custom inputs.")
    except Exception as e:
        st.error(f"Model files found but failed to load: {e}")
else:
    st.warning("Trained model not found in models/. Prediction panel will be disabled. (Run model_training.py to create model.)")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    dept_in = col1.selectbox("Department", depts)
    tool_in = col2.selectbox("Tool", tools)
    prompt_len_in = col3.number_input("Prompt length (words)", min_value=0, value=50)
    sensitive_in = col1.selectbox("Sensitive data included?", ["No","Yes"])
    usage_time_in = col2.number_input("Usage time (minutes)", min_value=0, value=5)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Prepare feature vector - here we must match what the model expects.
    # For demo, we will assemble simple numeric features used in training:
    # features = ['prompt_length','sensitive','usage_time'] etc.
    if not model_loaded:
        st.error("Model not loaded. Can't predict.")
    else:
        # simple preprocessing compatible with typical training script used earlier
        # sensitive -> numeric
        s_val = 1 if str(sensitive_in).lower() == 'yes' else 0
        # Build vector in same column order used during training.
        # IF your model was trained on one-hot dummies, full integration would need same columns.
        # Here we assume model trained on [prompt_length, sensitive, usage_time] (modify if different).
        X_new = np.array([[prompt_len_in, s_val, usage_time_in]])
        try:
            X_new_scaled = scaler.transform(X_new)
            pred_proba = rf_model.predict_proba(X_new_scaled)[0][1]  # probability of "risky"
            pred_label = "High" if pred_proba >= 0.7 else ("Medium" if pred_proba >= 0.4 else "Low")
            st.success(f"Predicted risk score: {pred_proba:.2f} → {pred_label} risk")
            if pred_proba >= 0.7:
                st.error("🚨 ALERT: Predicted HIGH risk. Take immediate action (review / block).")
        except Exception as e:
            st.error(f"Prediction failed — model/scaler mismatch. Error: {e}")

st.markdown("---")
st.write("💡 Tip: For production, ensure model's feature columns & preprocessing match exactly between training and this app.")
