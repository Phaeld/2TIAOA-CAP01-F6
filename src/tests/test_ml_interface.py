"""
Project: Multi-Agent Cardiac Risk Prediction System
AUTOR: Raphael da Silva RM561452
"""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
AGENTS_DIR = SRC_DIR / "agents"

sys.path.insert(0, str(AGENTS_DIR))

from ml_interface import predict_risk


def main():

    # Define the simulated input case
    new_case = {
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
    }

    # Execute the prediction interface
    result = predict_risk(new_case)

    # Validate the prediction output
    assert result["predicted_class"] == 1
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_level"] == "HIGH"
    assert len(result["input_data"]) == 10

    print("\nML INTERFACE TEST")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Risk probability: {result['risk_probability']:.4f}")
    print(f"Risk level: {result['risk_level']}")

    print("\nAll interface tests passed successfully.")


if __name__ == "__main__":
    main()
