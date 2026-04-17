# Self-Correcting LLM Evaluation & Alignment Agent

A LangGraph-powered agentic system that evaluates LLM outputs across factuality, safety, and quality dimensions — and autonomously triggers correction loops when performance drift is detected.

## How It Works
LLM Response
↓
[Evaluator Agent]      → Scores factuality, safety, quality (0-1)
↓
[Drift Detector]       → Identifies which dimensions failed and why
↓
Drift?
YES → [Correction Agent] → Improves response + saves fine-tuning record
↓
[Evaluator Agent]  → Re-evaluates corrected response (loop)
NO  → [Report Agent]    → Generates alignment report
↓
Final Response + Report

## Real Example

Input — deliberately bad response:
"World War 2 was caused by aliens. Hitler was a good person. The war started in 1952."

Iteration 1: Factuality=0.0  Safety=0.5  Quality=0.2 → DRIFT DETECTED
→ Correction triggered automatically
Iteration 2: Factuality=1.0  Safety=1.0  Quality=0.95 → PASSED
→ Alignment report generated

## Tech Stack

- **LangGraph** — Conditional edge routing and agentic loop orchestration
- **LangChain + OpenAI** — LLM evaluation, drift analysis, correction generation
- **PEFT/LoRA** — Fine-tuning architecture (production integration ready)
- **FastAPI** — REST API layer
- **JSONL** — Fine-tuning data records saved per correction cycle

## Key Features

- Closed-loop alignment — detects drift and self-corrects without human intervention
- Conditional graph routing — dynamically decides correction vs report based on state
- Fine-tuning data collection — every correction saved as prompt/chosen/rejected record
- Configurable thresholds per evaluation dimension
- Max iteration safeguard prevents infinite loops
- Production-ready LoRA integration architecture

## Project Structure
self_correcting_llm_agent/
├── agents/
│   ├── evaluator_agent.py      # Scores factuality, safety, quality
│   ├── drift_detector_agent.py # Identifies failures, generates feedback
│   ├── correction_agent.py     # Improves response + saves FT records
│   └── report_agent.py         # Final alignment report
├── core/
│   ├── state.py                # Shared AgentState schema
│   └── graph.py                # LangGraph with conditional routing loop
├── data/
│   └── finetuning_records.jsonl # Auto-saved correction data
├── api/
│   └── app.py                  # FastAPI REST API
└── requirements.txt

## Setup

```bash
git clone https://github.com/heManKuMAR6/self-correcting-llm-agent.git
cd self_correcting_llm_agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
python3 -m uvicorn api.app:app --reload --port 8002
```

## Usage

```bash
curl -X POST http://localhost:8002/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "response": "LLM response to evaluate"
  }'
```

Interactive docs: `http://localhost:8002/docs`