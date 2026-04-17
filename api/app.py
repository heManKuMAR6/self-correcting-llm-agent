from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from core.graph import llm_graph
import uvicorn

app = FastAPI(
    title="Self-Correcting LLM Evaluation Agent",
    description="Agentic system that evaluates LLM outputs and autonomously triggers corrections when drift is detected",
    version="1.0.0"
)

class EvaluationRequest(BaseModel):
    prompt: str
    response: str

class EvaluationResponse(BaseModel):
    final_response: str
    evaluation_scores: dict
    drift_detected: bool
    iterations_run: int
    final_status: str

@app.get("/health")
def health():
    return {"status": "healthy", "service": "self-correcting-llm-agent"}

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    print(f"\n[API] Evaluating response for prompt: {request.prompt[:50]}...")

    initial_state = {
        "prompt": request.prompt,
        "response": request.response,
        "evaluation_scores": {},
        "drift_detected": False,
        "correction_feedback": "",
        "evaluation_history": [],
        "final_status": "",
        "iteration": 1
    }

    result = llm_graph.invoke(initial_state)

    return EvaluationResponse(
        final_response=result["response"],
        evaluation_scores=result["evaluation_scores"],
        drift_detected=result["drift_detected"],
        iterations_run=result["iteration"],
        final_status=result["final_status"]
    )

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8002, reload=True)