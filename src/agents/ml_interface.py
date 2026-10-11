
"""
Project: Multi-Agent Cardiac Risk Prediction System
Module: Machine Learning Integration Interface
Description: Provides a reusable interface for external agents.
"""

from agent_ml import MLPredictionAgent


_ml_agent = None


def get_ml_agent():
    """Initialize and reuse the trained prediction agent."""

    global _ml_agent

    if _ml_agent is None:
        _ml_agent = MLPredictionAgent()

    return _ml_agent


def predict_risk(case_data: dict) -> dict:
    """
    Predict a synthetic risk peak using the trained model.

    Args:
        case_data: Dictionary containing the ten input features.

    Returns:
        Dictionary containing the predicted class,
        estimated probability, risk level and input data.
    """

    if not isinstance(case_data, dict):
        raise TypeError("Case data must be a dictionary.")

    agent = get_ml_agent()

    return agent.predict(case_data)
