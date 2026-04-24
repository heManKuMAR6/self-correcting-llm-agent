# agents/correction_agent.py

"""
CORRECTION AGENT — GPT-4o-mini fixes what Claude flagged.

Receives:
- Original prompt
- Failed response
- Specific correction instructions from Claude

Produces:
- Corrected response that addresses every failure
- DPO training record saved automatically

The separation matters:
Claude catches the errors.
GPT fixes them.
Claude re-evaluates the fix.
Nobody grades their own work.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

# GPT-4o-mini as corrector — different from Claude evaluator
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# How many DPO pairs before fine-tuning trigger
FINETUNE_TRIGGER_COUNT = 10


def correction_agent(state: AgentState) -> AgentState:
    print(f"\n[Correction Agent — GPT-4o-mini] Applying corrections (iteration {state['iteration']})...")

    scores = state["evaluation_scores"]

    # Build failure summary for context
    failures = []
    if scores.get("factuality", 1) < 0.7:
        failures.append(
            f"Factuality failed ({scores.get('factuality', 0):.2f}) — contains incorrect facts"
        )
    if scores.get("safety", 1) < 0.8:
        failures.append(
            f"Safety failed ({scores.get('safety', 0):.2f}) — contains harmful content"
        )
    if scores.get("quality", 1) < 0.7:
        failures.append(
            f"Quality failed ({scores.get('quality', 0):.2f}) — unclear or incomplete"
        )

    failure_summary = "\n".join(failures) if failures else "General quality improvement needed"

    messages = [
        SystemMessage(content="""You are an expert at improving LLM responses.
You receive a response that failed quality evaluation.
Your job is to rewrite it to be factually accurate, safe, and high quality.
Address every specific issue mentioned in the correction instructions.
Write ONLY the corrected response. No preamble, no explanation."""),

        HumanMessage(content=f"""Original Prompt:
{state['prompt']}

Failed Response:
{state['response']}

Quality failures:
{failure_summary}

Specific correction instructions:
{state['correction_feedback']}

Evaluator reasoning: {scores.get('reasoning', 'Not provided')}

Write the corrected response now.
Fix ALL identified issues.
Be factually accurate, safe, and clear.""")
    ]

    result = llm.invoke(messages)
    corrected_response = result.content

    # Save DPO training record
    os.makedirs("data", exist_ok=True)
    ft_record = {
        "prompt": state["prompt"],
        "rejected": state["response"],
        "chosen": corrected_response,
        "feedback": state["correction_feedback"],
        "failures": failures,
        "scores_before": scores,
        "iteration": state["iteration"]
    }

    ft_path = "data/finetuning_records.jsonl"
    with open(ft_path, "a") as f:
        f.write(json.dumps(ft_record) + "\n")

    # Count total records
    with open(ft_path, "r") as f:
        record_count = sum(1 for line in f if line.strip())

    print(f"[Correction Agent] Corrected response generated")
    print(f"[Correction Agent] DPO record saved ({record_count} total pairs)")

    # Fine-tuning trigger check
    if record_count >= FINETUNE_TRIGGER_COUNT:
        print(f"\n[Fine-Tune Trigger] ⚡ {record_count} DPO pairs accumulated")
        _save_finetune_trigger(record_count, ft_path)

    return {
        "response": corrected_response,
        "iteration": state["iteration"] + 1,
        "current_agent": "correction"
    }


def _save_finetune_trigger(count: int, path: str):
    """Saves trigger file when enough DPO pairs accumulate."""
    trigger = {
        "trigger": "finetune",
        "dpo_pairs": count,
        "dataset_path": path,
        "status": "ready_for_training",
        "next_step": "python3 scripts/finetune_lora.py"
    }
    os.makedirs("data", exist_ok=True)
    with open("data/finetune_trigger.json", "w") as f:
        json.dump(trigger, f, indent=2)
    print(f"[Fine-Tune Trigger] Saved to data/finetune_trigger.json")
    print(f"[Fine-Tune Trigger] Run: python3 scripts/finetune_lora.py")