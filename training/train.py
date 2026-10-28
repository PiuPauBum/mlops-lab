import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("incident-classifier-baseline")

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

params = {
    "model": "LogisticRegression",
    "max_iter": 1000,
    "random_state": 42,
}

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=params["max_iter"], random_state=params["random_state"])),
    ]
)

with mlflow.start_run():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(y_test, predictions, average="weighted"),
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.set_tags(
        {
            "dataset": "sklearn_wine",
            "stage": "baseline",
            "owner": "nikita",
        }
    )

    signature = infer_signature(X_test, predictions)
    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_test[:3],
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Metrics: {metrics}")
