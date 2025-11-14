import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Step 1: Load dataset
data_path = "data/simulated_shadow_ai_logs.csv"
df = pd.read_csv(data_path)

# Step 2: Encode categorical columns
df["sensitive_data"] = df["sensitive_data"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, columns=["department", "tool"], drop_first=True)

# Step 3: Features and labels
X = df.drop("sensitive_data", axis=1)
y = df["sensitive_data"]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the model
rf = RandomForestClassifier(n_estimators=120, random_state=42)
rf.fit(X_train_scaled, y_train)

# Step 7: Evaluate
y_pred = rf.predict(X_test_scaled)
print("✅ Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\n✅ Files saved:")
print("→ models/rf_model.joblib")
print("→ models/scaler.joblib")
