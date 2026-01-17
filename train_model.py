import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
import joblib
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "data/simulated_shadow_ai_logs.csv"

if not os.path.exists(DATA_PATH):
    raise Exception(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Clean Columns & Target
# -----------------------------
target_col = "risk_score"   # MUST match CSV

if target_col not in df.columns:
    raise Exception(f"Dataset must contain '{target_col}' column")

# Ensure numeric conversion
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

# Remove NaN rows
df = df.dropna()

# -----------------------------
# 3. Features (everything except target)
# -----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert continuous risk score into binary class
# 0 = Low risk, 1 = High risk
y = (y >= 0.6).astype(int)


# Convert categorical columns → numeric
X = pd.get_dummies(X)

# -----------------------------
# 4. Scale Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_scaled, y)

# -----------------------------
# 6. Ensure models folder exists
# -----------------------------
if not os.path.exists("models"):
    os.makedirs("models")

# -----------------------------
# 7. Save Model & Scaler
# -----------------------------
joblib.dump(model, "models/rf_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\nTraining completed successfully!")
print("Saved: models/rf_model.joblib")
print("Saved: models/scaler.joblib")
print("Columns used for training:", list(X.columns))
print("Training feature order:")
print(X.columns.tolist())

