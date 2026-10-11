
"""
Project: Multi-Agent Cardiac Risk Prediction System
Module: Protocol Analysis Agent
Description: Evaluates simulated protocols using ML predictions.
"""


class ProtocolAgent:

    def __init__(self):

        # Define the simulated protocol knowledge base
        self.protocols = [
            {
                "id": "P01",
                "feature": "average_spo2",
                "operator": "<",
                "threshold": 88,
                "action": "Initiate immediate respiratory support"
            },
            {
                "id": "P02",
                "feature": "system_load",
                "operator": ">",
                "threshold": 85,
                "action": "Open temporary care units"
            },
            {
                "id": "P03",
                "feature": "average_hr",
                "operator": ">",
                "threshold": 120,
                "action": "Initiate intensive monitoring"
            },
            {
                "id": "P04",
                "feature": "events_24h",
                "operator": ">",
                "threshold": 5,
                "action": "Activate rapid response team"
            },
            {
                "id": "P05",
                "feature": "resource_availability",
                "operator": "<",
                "threshold": 30,
                "action": "Redistribute medical supplies"
            },
            {
                "id": "P06",
                "feature": "waiting_time",
                "operator": ">",
                "threshold": 90,
                "action": "Prioritize emergency triage"
            },
            {
                "id": "P07",
                "feature": "temperature",
                "operator": ">",
                "threshold": 38,
                "action": "Initiate simulated infection protocol"
            },
            {
                "id": "P08",
                "feature": "event_density",
                "operator": ">",
                "threshold": 6,
                "action": "Activate contingency plan"
            },
            {
                "id": "P09",
                "feature": "systolic_bp",
                "operator": "<",
                "threshold": 90,
                "action": "Initiate simulated shock protocol"
            }
        ]

    def check_condition(self, value, operator, threshold):

        # Evaluate a protocol condition
        if operator == "<":
            return value < threshold

        if operator == ">":
            return value > threshold

        raise ValueError(f"Unsupported operator: {operator}")

    def analyze(self, ml_result):

        # Validate the prediction result
        if not isinstance(ml_result, dict):
            raise TypeError("ML result must be a dictionary.")

        required_fields = [
            "predicted_class",
            "risk_probability",
            "risk_level",
            "input_data"
        ]

        for field in required_fields:
            if field not in ml_result:
                raise ValueError(f"Missing ML result field: {field}")

        case_data = ml_result["input_data"]
        applied_protocols = []

        # Evaluate each protocol against the input data
        for protocol in self.protocols:

            feature = protocol["feature"]

            if feature not in case_data:
                raise ValueError(f"Missing input feature: {feature}")

            value = case_data[feature]

            condition_met = self.check_condition(
                value,
                protocol["operator"],
                protocol["threshold"]
            )

            if condition_met:
                applied_protocols.append({
                    "id": protocol["id"],
                    "feature": feature,
                    "observed_value": value,
                    "condition": (
                        f"{feature} "
                        f"{protocol['operator']} "
                        f"{protocol['threshold']}"
                    ),
                    "action": protocol["action"]
                })

        # Apply the simulated systemic risk protocol
        if ml_result["risk_level"] == "HIGH":
            applied_protocols.append({
                "id": "P10",
                "feature": "risk_level",
                "observed_value": "HIGH",
                "condition": "risk_level == HIGH",
                "action": "Activate simulated emergency state"
            })

        # Build the structured agent response
        result = {
            "agent": "ProtocolAgent",
            "predicted_class": ml_result["predicted_class"],
            "risk_probability": ml_result["risk_probability"],
            "risk_level": ml_result["risk_level"],
            "protocol_count": len(applied_protocols),
            "suggested_protocols": applied_protocols
        }

        return result
