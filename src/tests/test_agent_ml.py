"""
Project: Multi-Agent Cardiac Risk Prediction System
AUTOR: Raphael da Silva RM561452
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = PROJECT_ROOT / "agents"

sys.path.insert(0, str(AGENTS_DIR))

from agent_ml import MLPredictionAgent


def main():

    # Initialize the prediction agent
    agent = MLPredictionAgent()

    # Simulated case from the assignment
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

    # Execute the prediction
    result = agent.predict(new_case)

    # Validate the response structure
    assert result["predicted_class"] in [0, 1]
    assert 0 <= result["risk_probability"] <= 1
    assert result["risk_level"] in ["LOW", "HIGH"]

    # Display the prediction result
    print("\nMACHINE LEARNING AGENT TEST")
    print(json.dumps(result, indent=4))

    print("\nAgent test completed successfully.")


if __name__ == "__main__":
    main()
