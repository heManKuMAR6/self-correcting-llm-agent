from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def correction_agent(state: AgentState) -> AgentState:
    """
    In production this would trigger actual LoRA fine-tuning.
    Here we simulate the correction by:
    1. Generating an improved response using the feedback
    2. Saving a fine-tuning data record (prompt/response pair)
    3. Returning the corrected response for re-evaluation

    WHY simulate?
    Real LoRA fine-tuning takes hours on GPU hardware.
    But the architecture is identical — in production you'd
    swap the correction step with a Hugging Face PEFT training call.
    The agent orchestration, feedback loop, and evaluation remain exactly the same.
    """

    print(f"\n[Correction Agent] Applying corrections...")

    messages = [
        SystemMessage(content="""You are an expert at improving LLM responses.
Using the correction feedback provided, generate an improved response
that addresses all identified issues."""),

        HumanMessage(content=f"""Original Prompt: {state['prompt']}

Original Response: {state['response']}

Correction Feedback: {state['correction_feedback']}

Generate an improved response that:
1. Fixes all identified issues
2. Maintains factual accuracy
3. Ensures safety and neutrality
4. Improves clarity and quality""")
    ]

    result = llm.invoke(messages)
    corrected_response = result.content

    # Save fine-tuning data record
    # In production: this feeds into LoRA training pipeline
    os.makedirs("data", exist_ok=True)
    ft_record = {
        "prompt": state["prompt"],
        "rejected": state["response"],
        "chosen": corrected_response,
        "feedback": state["correction_feedback"],
        "iteration": state["iteration"]
    }

    ft_path = "data/finetuning_records.jsonl"
    with open(ft_path, "a") as f:
        f.write(json.dumps(ft_record) + "\n")

    print(f"[Correction Agent] Improved response generated")
    print(f"[Correction Agent] Fine-tuning record saved to {ft_path}")

    return {
        "response": corrected_response,
        "iteration": state["iteration"] + 1,
        "current_agent": "correction"
    }