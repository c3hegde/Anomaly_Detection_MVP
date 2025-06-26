import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def train_isolation_forest(data_path="data/raw/expenses.csv", model_path="models/isolation_forest.pkl"):
    df = pd.read_csv(data_path)

    # Select numerical features for training
    features = ["amount"]  # Expand this based on your data
    df = df.dropna(subset=features)
    
    X = df[features]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # Predict anomaly scores
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["anomaly_flag"] = model.predict(X_scaled)  # -1 for anomaly

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, "models/scaler.pkl")
    df.to_csv("data/processed/expenses_scored.csv", index=False)

    print("Isolation Forest trained and predictions saved.")

if __name__ == "__main__":
    train_isolation_forest()
