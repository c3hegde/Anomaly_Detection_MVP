import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

n = 2000
departments = ['HR', 'Engineering', 'Marketing', 'Finance']
categories = ['Travel', 'Meals', 'Supplies', 'Software', 'Other']

df = pd.DataFrame({
    "employee_id": np.random.randint(1000, 1100, size=n),
    "department": np.random.choice(departments, size=n),
    "category": np.random.choice(categories, size=n),
    "amount": np.random.normal(loc=200, scale=50, size=n).round(2),
    "timestamp": pd.date_range("2023-01-01", periods=n, freq="H"),
})

# Inject fraud/anomaly
fraud_indices = random.sample(range(n), 50)
df.loc[fraud_indices, "amount"] *= 5
df["is_fraud"] = 0
df.loc[fraud_indices, "is_fraud"] = 1

df.to_csv("data/raw/expenses.csv", index=False)
