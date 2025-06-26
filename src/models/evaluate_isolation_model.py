import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sklearn.metrics import classification_report

def evaluate(path="data/processed/expenses_scored.csv"):
    df = pd.read_csv(path)
    if "is_fraud" not in df.columns:
        print("No ground truth labels found.")
        return

    # Convert Isolation Forest prediction to binary (1 = fraud)
    df["predicted_fraud"] = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)
    print(" Evaluation Report (Unsupervised):")
    print(classification_report(df["is_fraud"], df["predicted_fraud"]))

if __name__ == "__main__":
    evaluate()
