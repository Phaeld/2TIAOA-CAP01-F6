"""
Project: Multi-Agent Cardiac Risk Prediction System
AUTOR: Raphael da Silva RM561452
"""

import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
AGENTS_DIR = SRC_DIR / "agents"

sys.path.insert(0, str(AGENTS_DIR))

from agent_ml import MLPredictionAgent
from protocol_agent import ProtocolAgent


def main():

    # Initialize both agents
    ml_agent = MLPredictionAgent()
    protocol_agent = ProtocolAgent()

    # Define the simulated test case
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

    # Execute the ML prediction agent
    ml_result = ml_agent.predict(new_case)

    # Forward the ML result to the protocol agent
    protocol_result = protocol_agent.analyze(ml_result)

    # Validate the integration output
    assert protocol_result["predicted_class"] in [0, 1]
    assert 0 <= protocol_result["risk_probability"] <= 1

    assert protocol_result["protocol_count"] == len(
        protocol_result["suggested_protocols"]
    )

    # Display the final structured response
    print("\nMULTI-AGENT INTEGRATION TEST")
    print(json.dumps(protocol_result, indent=4))

    print("\nIntegration test completed successfully.")


if __name__ == "__main__":
    main()
