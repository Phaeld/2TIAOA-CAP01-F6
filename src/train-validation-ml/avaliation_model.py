"""
Project: Multi-Agent Cardiac Risk Prediction System
AUTOR: Raphael da Silva RM561452
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "risk_model.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"


def evaluate_model():

    # Load the trained model artifact
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]
    target = artifact["target"]

    # Load the reserved test dataset
    df = pd.read_csv(TEST_DATA_PATH)

    X_test = df[features]
    y_test = df[target]

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Display model performance
    print("\nMODEL EVALUATION RESULTS")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return model, features


def predict_new_case(model, features):

    # Simulated case provided in the assignment
    new_case = pd.DataFrame([{
        "middle_ages": 62,
        "system_load": 92,
        "waiting_time": 75,
        "average_hr": 118,
        "average_spo2": 89,
        "systolic_bp": 100,
        "temperature": 37.5,
        "event_density": 7,
        "events_24h": 5,
        "resource_availability": 25
    }])

    # Ensure the correct feature order
    new_case = new_case[features]

    # Predict the target class
    predicted_class = int(model.predict(new_case)[0])

    # Estimate the positive-class probability
    class_index = list(model.classes_).index(1)

    risk_probability = float(
        model.predict_proba(new_case)[0][class_index]
    )

    print("\nNEW CASE PREDICTION")
    print(f"Predicted class: {predicted_class}")
    print(f"Risk probability: {risk_probability:.4f}")
    print(f"Risk percentage: {risk_probability * 100:.2f}%")


def main():

    model, features = evaluate_model()
    predict_new_case(model, features)


if __name__ == "__main__":
    main()
