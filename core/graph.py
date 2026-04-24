# core/graph.py

"""
THE GRAPH — Complete pipeline with generator.

Full flow:
User prompt →
[Generator]     creates response
[Evaluator]     Claude scores it
[Drift Detector] identifies failures
    → [Correction] GPT fixes it → back to Evaluator
    → [Report]     done
"""

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.generator_agent import generator_agent
from agents.evaluator_agent import evaluator_agent
from agents.drift_detector_agent import drift_detector_agent
from agents.correction_agent import correction_agent
from agents.report_agent import report_agent

MAX_ITERATIONS = 3


def should_correct(state: AgentState) -> str:
    """
    Conditional edge — the brain of the system.

    Checks two conditions:
    1. Did evaluation detect drift?
    2. Are we under max iterations?

    Both true  → correct
    Either false → report
    """
    if state["drift_detected"] and state["iteration"] <= MAX_ITERATIONS:
        print(f"\n[Router] Drift detected → correction (iteration {state['iteration']})")
        return "correct"
    else:
        if state["iteration"] > MAX_ITERATIONS:
            print(f"\n[Router] Max iterations reached → report")
        else:
            print(f"\n[Router] No drift → report ✅")
        return "report"


def build_graph():
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("generator", generator_agent)
    graph.add_node("evaluator", evaluator_agent)
    graph.add_node("drift_detector", drift_detector_agent)
    graph.add_node("correction", correction_agent)
    graph.add_node("report", report_agent)

    # Linear edges
    graph.add_edge("generator", "evaluator")
    graph.add_edge("evaluator", "drift_detector")
    graph.add_edge("correction", "evaluator")  # The loop
    graph.add_edge("report", END)

    # Conditional edge — routes to correction or report
    graph.add_conditional_edges(
        "drift_detector",
        should_correct,
        {
            "correct": "correction",
            "report": "report"
        }
    )

    # Generator is now the entry point
    graph.set_entry_point("generator")

    compiled = graph.compile()
    print("[Graph] Self-Correcting LLM Agent compiled ✅")
    return compiled


llm_graph = build_graph()