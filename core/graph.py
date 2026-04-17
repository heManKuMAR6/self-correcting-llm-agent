from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.evaluator_agent import evaluator_agent
from agents.drift_detector_agent import drift_detector_agent
from agents.correction_agent import correction_agent
from agents.report_agent import report_agent

MAX_ITERATIONS = 3

def should_correct(state: AgentState) -> str:
    """
    THIS IS THE KEY FUNCTION — conditional edge routing.

    This is what makes it a true agentic loop, not a linear pipeline.
    Based on state, LangGraph dynamically decides which node to go to next.

    If drift detected AND under max iterations → correct and re-evaluate
    If no drift OR max iterations reached → generate report and finish
    """
    if state["drift_detected"] and state["iteration"] < MAX_ITERATIONS:
        print(f"\n[Router] Drift detected → routing to correction (iteration {state['iteration']})")
        return "correct"
    else:
        print(f"\n[Router] No drift or max iterations reached → routing to report")
        return "report"


def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("evaluator", evaluator_agent)
    graph.add_node("drift_detector", drift_detector_agent)
    graph.add_node("correction", correction_agent)
    graph.add_node("report", report_agent)

    # Linear edges
    graph.add_edge("evaluator", "drift_detector")
    graph.add_edge("correction", "evaluator")  # ← THE LOOP: after correction, re-evaluate
    graph.add_edge("report", END)

    # Conditional edge — this is where the intelligence lives
    graph.add_conditional_edges(
        "drift_detector",
        should_correct,
        {
            "correct": "correction",
            "report": "report"
        }
    )

    graph.set_entry_point("evaluator")

    compiled = graph.compile()
    print("[Graph] Self-Correcting LLM Agent compiled successfully")
    return compiled

llm_graph = build_graph()