import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def train_with_tracking(data_path="data/raw/expenses.csv"):
    df = pd.read_csv(data_path)

    # Separate target
    y = df["is_fraud"]
    X = df.drop("is_fraud", axis=1)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    # Full model pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Set experiment and start MLflow run
    mlflow.set_experiment("ExpenseAnomalyDetection")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        f1_score = report["1"]["f1-score"]

        # Log params and metrics
        mlflow.log_param("model_type", "LogisticRegression + OHE")
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)

        # Log the full pipeline
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="ExpenseAnomalyModel"
        )

        print("Model trained and logged to MLflow successfully.")

if __name__ == "__main__":
    train_with_tracking()