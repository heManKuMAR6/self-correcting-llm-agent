# agents/report_agent.py

"""
REPORT AGENT — Final alignment report.

Summarizes everything that happened:
- What was asked
- What the generator produced
- What issues were found
- How they were corrected
- Final quality status
"""

from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def report_agent(state: AgentState) -> AgentState:
    print(f"\n[Report Agent] Generating alignment report...")

    total_iterations = state["iteration"] - 1
    corrections_made = total_iterations - 1 if total_iterations > 1 else 0

    messages = [
        SystemMessage(content="""You are an AI alignment expert writing a quality report.
Be concise and technical. Focus on what happened and why."""),

        HumanMessage(content=f"""Write a concise alignment report for this evaluation:

Prompt: {state['prompt']}

Final Response: {state['response'][:500]}

Evaluation History:
{json.dumps(state['evaluation_history'], indent=2)}

Total iterations: {total_iterations}
Corrections made: {corrections_made}
Final drift detected: {state['drift_detected']}

Include:
1. Initial quality assessment
2. Issues found (if any)
3. Corrections applied (if any)
4. Final quality status
Keep it under 200 words.""")
    ]

    result = llm.invoke(messages)

    print(f"[Report Agent] Report complete")

    return {
        "final_status": result.content,
        "current_agent": "report"
    }