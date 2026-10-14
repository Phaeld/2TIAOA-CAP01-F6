
"""
CardioIA - Protocol Specialist Agent
Applies the existing simulated protocol rules to the ML result.
"""

from agents import Agent, RunContextWrapper, function_tool

from pipeline_context import PipelineContext
from protocol_agent import ProtocolAgent


_protocol_agent = ProtocolAgent()


@function_tool
def analyze_protocols(
    ctx: RunContextWrapper[PipelineContext],
) -> dict:
    """Apply the simulated protocols to the shared ML prediction."""

    context = ctx.context

    if not context.ml_result:
        raise ValueError(
            "Risk prediction is missing. Run the risk agent first."
        )

    context.protocol_result = _protocol_agent.analyze(
        context.ml_result
    )

    return context.protocol_result


protocol_specialist = Agent[PipelineContext](
    name="Protocol Specialist Agent",
    instructions=(
        "You are the CardioIA simulated protocol specialist. "
        "Use analyze_protocols to retrieve the protocols selected by "
        "the existing rule-based engine. Report their identifiers and "
        "actions faithfully. Do not invent additional protocols."
    ),
    tools=[analyze_protocols],
)
