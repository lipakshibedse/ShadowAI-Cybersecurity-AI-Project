import pandas as pd
import numpy as np
import random
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Departments and tools
departments = ["Finance", "HR", "IT", "Sales", "Operations"]
tools = ["ChatGPT", "Bard", "Claude", "Copilot", "Gemini"]

# Simulated dataset
records = []
for _ in range(1200):
    dept = random.choice(departments)
    tool = random.choice(tools)
    prompt_length = random.randint(20, 400)
    sensitive = random.choice(["Yes", "No"])
    usage_time = random.randint(1, 60)
    risk_score = random.uniform(0, 1)
    records.append([dept, tool, prompt_length, sensitive, usage_time, risk_score])

df = pd.DataFrame(records, columns=[
    "department", "tool", "prompt_length", "sensitive_data", "usage_time", "risk_score"
])

file_path = "data/simulated_shadow_ai_logs.csv"
df.to_csv(file_path, index=False)

print(f"✅ Dataset generated and saved as: {file_path}")
print(f"Total records: {len(df)}")
