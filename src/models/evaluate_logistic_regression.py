import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def evaluate(data_path="data/processed/expenses_scored.csv"):
    df = pd.read_csv(data_path)

    features = ["amount"]
    X = df[features]
    y_true = df["is_fraud"]

    model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler_logreg.pkl")

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print(" Evaluation Report (Logistic Regression):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
