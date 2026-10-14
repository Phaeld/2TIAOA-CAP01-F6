
"""
CardioIA - Risk Analysis Agent
Uses the trained ML model to calculate the risk prediction.
"""

from agents import Agent, RunContextWrapper, function_tool

from agent_ml import MLPredictionAgent
from pipeline_context import PipelineContext


# Load the existing trained model
_ml_agent = MLPredictionAgent()


@function_tool
def analyze_risk(
    ctx: RunContextWrapper[PipelineContext],
) -> dict:
    """Run the trained model using the case stored in the shared context."""

    context = ctx.context

    if not context.case_data:
        raise ValueError("No case data was provided.")

    # The prediction comes from the actual trained ML model
    context.ml_result = _ml_agent.predict(context.case_data)

    return context.ml_result


risk_agent = Agent[PipelineContext](
    name="Risk Analysis Agent",
    instructions=(
        "You are the CardioIA risk analysis specialist. "
        "Use the analyze_risk tool to obtain the prediction from the "
        "trained machine learning model. Report the predicted class, "
        "risk probability and risk level exactly as returned by the tool. "
        "Never invent or modify numerical results."
    ),
    tools=[analyze_risk],
)
