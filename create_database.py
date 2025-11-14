import pandas as pd
import sqlite3

# Step 1: Load the CSV file
df = pd.read_csv("data/simulated_shadow_ai_logs.csv")  

# Step 2: Create an SQLite database (if it doesn’t already exist)
conn = sqlite3.connect("data/shadowai.db")

# Step 3: Create a table and insert the data
df.to_sql("ai_logs", conn, if_exists="replace", index=False)

# Step 4: Close the connection
conn.close()

print("✅ Database created successfully: shadowai.db")
