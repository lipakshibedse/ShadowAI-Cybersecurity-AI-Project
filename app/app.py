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
base_dir = os.path.dirname(os.path.abspath(__file__))

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
# ---------------------------

st.subheader("🧠 AI Prediction Panel (Use trained model)")

model_loaded = False

# ---------------------------
# FIXED feature list (must match training exactly)
# ---------------------------
feature_list = [
    'prompt_length',
    'usage_time',
    'department_Finance',
    'department_HR',
    'department_IT',
    'department_Operations',
    'department_Sales',
    'tool_Bard',
    'tool_ChatGPT',
    'tool_Claude',
    'tool_Copilot',
    'tool_Gemini',
    'sensitive_data_No',
    'sensitive_data_Yes'
]


if os.path.exists(rf_path) and os.path.exists(scaler_path):
    try:
        rf_model = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)

        # Load the required feature list
        #feature_list_path = os.path.join(base_dir, "models", "feature_list.json")

        model_loaded = True
        st.success("Model + Scaler + Feature List loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
else:
    st.warning("Model files missing. Run train_model.py first.")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    dept_in = col1.selectbox("Department", depts)
    tool_in = col2.selectbox("Tool", tools)
    prompt_len_in = col3.number_input("Prompt length (words)", min_value=0, value=50)

    sensitive_in = col1.selectbox("Sensitive data included?", ["No", "Yes"])
    usage_time_in = col2.number_input("Usage time (minutes)", min_value=0, value=5)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    if not model_loaded:
        st.error("Model not loaded.")
    else:
        # 1️⃣ Empty feature row (same as training)
        input_row = {col: 0 for col in feature_list}

        # 2️⃣ Department one-hot
        dept_col = f"department_{dept_in}"
        if dept_col in input_row:
            input_row[dept_col] = 1

        # 3️⃣ Tool one-hot (ALL tools exist)
        for tool in ["Bard", "ChatGPT", "Claude", "Copilot", "Gemini"]:
            col = f"tool_{tool}"
            if col in input_row:
                input_row[col] = 1 if tool == tool_in else 0

        # 4️⃣ Sensitive data one-hot
        if sensitive_in == "Yes":
            input_row["sensitive_data_Yes"] = 1
            input_row["sensitive_data_No"] = 0
        else:
            input_row["sensitive_data_No"] = 1
            input_row["sensitive_data_Yes"] = 0

        # 5️⃣ Numeric values
        input_row["prompt_length"] = prompt_len_in
        input_row["usage_time"] = usage_time_in

        # 6️⃣ DataFrame + SAME ORDER
        X_new = pd.DataFrame([input_row])
        X_new = X_new[feature_list]   # 🔥 THIS LINE IS VERY IMPORTANT

        try:
            X_new_scaled = scaler.transform(X_new)
            pred_proba = rf_model.predict_proba(X_new_scaled)[0][1]

            if pred_proba < 0.3:
                level = "Low"
            elif pred_proba < 0.6:
                level = "Medium"
            else:
                level = "High"

            st.success(f"Predicted Risk Score: {pred_proba:.2f} → {level} Risk")

            if level == "High":
                st.error("🚨 HIGH RISK — Immediate Review Needed")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
