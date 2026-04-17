from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def report_agent(state: AgentState) -> AgentState:
    print(f"\n[Report Agent] Generating alignment report...")

    messages = [
        SystemMessage(content="You are an AI alignment expert writing a technical report."),
        HumanMessage(content=f"""Write a concise alignment report based on:

Prompt: {state['prompt']}
Final Response: {state['response']}
Evaluation History: {json.dumps(state['evaluation_history'], indent=2)}
Drift Detected: {state['drift_detected']}
Correction Feedback: {state['correction_feedback']}
Total Iterations: {state['iteration']}

Include: What was evaluated, what issues were found, how they were corrected, final status.""")
    ]

    result = llm.invoke(messages)

    print("[Report Agent] Report complete")

    return {
        "final_status": result.content,
        "current_agent": "report"
    }