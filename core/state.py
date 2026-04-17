from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    # The prompt sent to the LLM being evaluated
    prompt: str

    # The LLM's response to evaluate
    response: str

    # Scores from evaluation agent (factuality, safety, quality)
    evaluation_scores: dict

    # Whether drift was detected
    drift_detected: bool

    # Feedback generated for fine-tuning
    correction_feedback: str

    # History of evaluation rounds
    evaluation_history: Annotated[list, operator.add]

    # Final status report
    final_status: str

    # How many correction loops have run
    iteration: int