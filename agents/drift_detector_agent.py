from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def drift_detector_agent(state: AgentState) -> AgentState:
    print(f"\n[Drift Detector] Analyzing drift pattern...")

    scores = state["evaluation_scores"]

    # Find which dimensions failed
    failed = [
        f"{metric} (score: {scores.get(metric, 0):.2f})"
        for metric in ["factuality", "safety", "quality"]
        if scores.get(metric, 0) < 0.7
    ]

    if not failed:
        print("[Drift Detector] No drift — all scores within threshold")
        return {
            "correction_feedback": "No correction needed.",
            "current_agent": "drift_detector"
        }

    messages = [
        SystemMessage(content="You are an LLM alignment expert. Generate specific correction feedback."),
        HumanMessage(content=f"""The following quality dimensions failed evaluation:
{chr(10).join(failed)}

Original Prompt: {state['prompt']}
Original Response: {state['response']}
Evaluator Reasoning: {scores.get('reasoning', 'N/A')}

Generate specific, actionable correction feedback that would improve the response.
Focus on what went wrong and exactly how to fix it.""")
    ]

    result = llm.invoke(messages)

    print(f"[Drift Detector] Correction feedback generated")

    return {
        "correction_feedback": result.content,
        "current_agent": "drift_detector"
    }