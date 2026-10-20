
from pydantic import BaseModel, Field


class ProtocolSuggestion(BaseModel):
    protocol_id: str = Field(description="Identifier of the simulated protocol")
    action: str = Field(description="Suggested simulated operational action")


class FinalRecommendation(BaseModel):
    predicted_class: int = Field(
        description="Predicted class returned by the ML model"
    )
    risk_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability between 0 and 1"
    )
    risk_level: str = Field(
        description="Risk classification returned by the ML model"
    )
    suggested_protocols: list[ProtocolSuggestion] = Field(
        description="Simulated protocols selected by the rule-based engine"
    )
    summary: str = Field(
        description="Short summary of the synthetic operational risk analysis"
    )
    disclaimer: str = Field(
        default=(
            "Prototipo academico com dados sinteticos. "
            "Esta saida nao e diagnostico clinico nem orientacao medica."
        ),
        description="Limitations of the prototype"
    )
