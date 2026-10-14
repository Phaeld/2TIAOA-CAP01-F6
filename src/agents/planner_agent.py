
from agents import Agent, RunContextWrapper, function_tool
from pipeline_context import PipelineContext
from response_schema import FinalRecommendation


@function_tool
def get_pipeline_results(ctx: RunContextWrapper[PipelineContext]) -> dict:
    """Retrieve the actual prediction and protocol analysis results."""
    context = ctx.context

    if not context.ml_result:
        raise ValueError("The risk prediction is missing.")

    if not context.protocol_result:
        raise ValueError("The protocol analysis is missing.")

    return {
        "prediction": context.ml_result,
        "protocol_analysis": context.protocol_result,
    }


planner_agent = Agent[PipelineContext](
    name="Response Planning Agent",
    instructions=(
        "You are the response planning specialist for CardioIA, an academic "
        "prototype that analyzes synthetic operational cardiac-risk data. "
        "First, call get_pipeline_results to retrieve the real model prediction "
        "and the protocol engine results. "
        "Build the final structured recommendation using the provided output schema. "
        "Copy predicted_class, risk_probability and risk_level from the prediction "
        "without changing their values. "
        "Copy the protocol identifiers and actions from protocol_analysis. "
        "Do not invent probabilities, classifications, protocols or actions. "
        "Write a concise summary based only on the retrieved results. "
        "This is a simulation, not a clinical diagnosis or medical advice."
    ),
    tools=[get_pipeline_results],
    output_type=FinalRecommendation,
)
