
"""
Shared context for the CardioIA multi-agent pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """Stores the case and results shared between agents."""

    case_data: dict[str, Any] = field(default_factory=dict)
    ml_result: dict[str, Any] = field(default_factory=dict)
    protocol_result: dict[str, Any] = field(default_factory=dict)
