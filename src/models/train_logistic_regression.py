import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def train_logistic_model(data_path="data/processed/expenses_scored.csv", model_path="models/logistic_regression.pkl"):
    df = pd.read_csv(data_path)

    if "is_fraud" not in df.columns:
        raise ValueError(" 'is_fraud' column not found in the dataset.")

    # Define features and target
    features = ["amount"]  # You can expand this based on what's available
    X = df[features]
    y = df["is_fraud"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, "models/scaler_logreg.pkl")

    print(" Logistic Regression model trained and saved.")
    return model, scaler

if __name__ == "__main__":
    train_logistic_model()
