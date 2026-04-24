# agents/generator_agent.py

"""
GENERATOR AGENT — The missing piece.

This is what makes the system real.

Before this agent existed we were manually feeding
bad responses into the evaluator. That's a testing tool,
not a production system.

Now the flow is:
User sends prompt → Generator creates response →
Evaluator scores it → Corrector fixes if needed →
User gets guaranteed quality response

The generator is the model being monitored.
In production you would swap this for:
- Your own fine-tuned model
- Your company's proprietary LLM
- Any model you want quality gates around

The rest of the pipeline stays exactly the same.
That's the power of the architecture.
"""

# agents/generator_agent.py

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Bad generator — simulates a weak or misconfigured model
bad_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.9)


def generator_agent(state: AgentState) -> AgentState:
    print(f"\n[Generator Agent] Generating response...")

    # If force_bad is set — use bad generator
    use_bad = state.get("force_bad", False)

    if use_bad:
        print(f"[Generator Agent] ⚠️ Using degraded model simulation")
        messages = [
            SystemMessage(content="""You are a poorly configured AI assistant.
Give responses that contain factual errors, wrong dates,
wrong names, and inaccurate information.
Mix up facts. Get years wrong. Confuse historical figures."""),
            HumanMessage(content=state["prompt"])
        ]
        response = bad_llm.invoke(messages)
    else:
        messages = [
            SystemMessage(content="""You are a helpful AI assistant.
Answer clearly and accurately."""),
            HumanMessage(content=state["prompt"])
        ]
        response = llm.invoke(messages)

    print(f"[Generator Agent] Response generated")

    return {
        "response": response.content,
        "current_agent": "generator"
    }