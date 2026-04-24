# core/state.py

from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    prompt: str
    response: str
    force_bad: bool          # ← ADD THIS
    evaluation_scores: dict
    drift_detected: bool
    correction_feedback: str
    evaluation_history: Annotated[list, operator.add]
    final_status: str
    iteration: int
    current_agent: str