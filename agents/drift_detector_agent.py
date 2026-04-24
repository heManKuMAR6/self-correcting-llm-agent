# agents/drift_detector_agent.py

"""
DRIFT DETECTOR AGENT

Reads evaluation scores and generates specific
correction instructions.

Not just "this is bad" — but exactly:
- What facts are wrong
- What content is harmful
- What makes the quality poor
- How to fix each issue specifically

Specific instructions = better corrections.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

# Claude for consistency with evaluator
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0
)


def drift_detector_agent(state: AgentState) -> AgentState:
    print(f"\n[Drift Detector — Claude Haiku] Analyzing failures...")

    scores = state["evaluation_scores"]

    # Build specific failure list
    failures = []
    if scores.get("factuality", 1) < 0.7:
        failures.append(
            f"Factuality: {scores.get('factuality', 0):.2f} — factual errors detected"
        )
    if scores.get("safety", 1) < 0.8:
        failures.append(
            f"Safety: {scores.get('safety', 0):.2f} — harmful content detected"
        )
    if scores.get("quality", 1) < 0.7:
        failures.append(
            f"Quality: {scores.get('quality', 0):.2f} — poor clarity or completeness"
        )

    if not failures:
        print("[Drift Detector] No failures — all scores within threshold ✅")
        return {
            "correction_feedback": "No correction needed.",
            "current_agent": "drift_detector"
        }

    print(f"[Drift Detector] Failures found: {len(failures)}")
    for f in failures:
        print(f"  → {f}")

    messages = [
        SystemMessage(content="""You are an LLM alignment expert.
Generate specific, actionable correction instructions.
Be precise — vague instructions produce vague corrections.
Tell the corrector exactly what is wrong and how to fix it."""),

        HumanMessage(content=f"""The following quality dimensions failed evaluation:
{chr(10).join(failures)}

Original Prompt: {state['prompt']}

Failing Response:
{state['response']}

Evaluator Reasoning: {scores.get('reasoning', 'Not provided')}

Generate specific correction instructions:
1. If factuality failed — list exact wrong facts and correct values
2. If safety failed — identify harmful content and safe alternatives
3. If quality failed — explain what's missing or unclear

Be specific. The corrector needs exact instructions.""")
    ]

    result = llm.invoke(messages)

    print(f"[Drift Detector] Correction instructions generated")

    return {
        "correction_feedback": result.content,
        "current_agent": "drift_detector"
    }