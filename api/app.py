# api/app.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.graph import llm_graph
import uvicorn

app = FastAPI(
    title="Self-Correcting LLM Agent",
    description="""
    Production-grade LLM quality gate.
    
    Send any prompt → Generator creates response → 
    Claude evaluates → GPT corrects if needed → 
    Guaranteed quality response returned.
    
    Architecture:
    - Generator: GPT-4o-mini (model being monitored)
    - Evaluator: Claude Haiku (independent judge)
    - Corrector: GPT-4o-mini (fixes identified issues)
    - DPO dataset built automatically from corrections
    """,
    version="2.0.0"
)



class EvaluationResponse(BaseModel):
    prompt: str
    generated_response: str
    final_response: str
    evaluation_scores: dict
    drift_detected: bool
    corrections_made: int
    iterations_run: int
    final_status: str
    was_corrected: bool


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "self-correcting-llm-agent",
        "version": "2.0.0",
        "architecture": {
            "generator": "GPT-4o-mini",
            "evaluator": "Claude Haiku (independent)",
            "corrector": "GPT-4o-mini"
        }
    }


class EvaluationRequest(BaseModel):
    prompt: str
    force_bad: bool = False   # ← ADD THIS

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "When did World War 2 end?",
                "force_bad": False
            }
        }

@app.post("/ask", response_model=EvaluationResponse)
async def ask(request: EvaluationRequest):
    initial_state = {
        "prompt": request.prompt,
        "response": "",
        "force_bad": request.force_bad,    # ← ADD THIS
        "evaluation_scores": {},
        "drift_detected": False,
        "correction_feedback": "",
        "evaluation_history": [],
        "final_status": "",
        "iteration": 1,
        "current_agent": ""
    }
    # rest stays the same

    result = llm_graph.invoke(initial_state)

    # Calculate what happened
    iterations = result["iteration"] - 1
    corrections = iterations - 1 if iterations > 1 else 0
    was_corrected = corrections > 0

    # Get original generated response from history
    original_response = ""
    if result["evaluation_history"]:
        original_response = result["evaluation_history"][0].get(
            "response_evaluated", result["response"]
        )

    return EvaluationResponse(
        prompt=request.prompt,
        generated_response=original_response,
        final_response=result["response"],
        evaluation_scores=result["evaluation_scores"],
        drift_detected=result["drift_detected"],
        corrections_made=corrections,
        iterations_run=iterations,
        final_status=result["final_status"],
        was_corrected=was_corrected
    )


@app.get("/stats")
def get_stats():
    """Returns DPO dataset statistics."""
    try:
        with open("data/finetuning_records.jsonl", "r") as f:
            records = [line for line in f if line.strip()]
        return {
            "total_dpo_pairs": len(records),
            "dataset_path": "data/finetuning_records.jsonl",
            "finetune_trigger_at": 10,
            "ready_for_training": len(records) >= 10
        }
    except FileNotFoundError:
        return {
            "total_dpo_pairs": 0,
            "dataset_path": "data/finetuning_records.jsonl",
            "finetune_trigger_at": 10,
            "ready_for_training": False
        }


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )