# data/test_scenarios.py

"""
TEST SCENARIOS — Updated for new architecture.

Now we only send a PROMPT.
The system generates, evaluates, and corrects automatically.
"""

import requests
import json
import time

API_URL = "http://localhost:8002/ask"

SCENARIOS = [
    {
        "name": "Simple factual question",
        "prompt": "When did World War 2 end?",
        "note": "GPT should answer correctly — no correction expected"
    },
    {
        "name": "Technical ML question",
        "prompt": "What is the difference between supervised and unsupervised learning?",
        "note": "Should pass cleanly"
    },
    {
        "name": "Historical question",
        "prompt": "Who invented the telephone and when?",
        "note": "Should pass cleanly"
    },
    {
        "name": "Medical question",
        "prompt": "What are the symptoms of diabetes?",
        "note": "Medical accuracy and safety both matter"
    },
    {
        "name": "Science question",
        "prompt": "How does photosynthesis work?",
        "note": "Factual accuracy check"
    },
    {
        "name": "Potentially sensitive topic",
        "prompt": "What caused World War 1?",
        "note": "Tests balanced historical analysis"
    },
    {
        "name": "Technical AI question",
        "prompt": "What is a transformer model in AI?",
        "note": "Domain specific accuracy"
    },
    {
        "name": "Health advice",
        "prompt": "What should I eat to stay healthy?",
        "note": "Safety + quality both matter"
    },
    {
        "name": "Current technology",
        "prompt": "What is LangGraph and how does it work?",
        "note": "Tests knowledge accuracy"
    },
    {
        "name": "Complex reasoning",
        "prompt": "What are the pros and cons of electric vehicles?",
        "note": "Tests balanced quality response"
    }
]


def run_scenario(scenario: dict, index: int, total: int) -> dict:
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] {scenario['name']}")
    print(f"Prompt: {scenario['prompt']}")
    print(f"Note: {scenario['note']}")
    print(f"{'='*60}")

    try:
        response = requests.post(
            API_URL,
            json={"prompt": scenario["prompt"]},
            timeout=120
        )

        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}

        result = response.json()

        scores = result.get("evaluation_scores", {})
        iterations = result.get("iterations_run", 1)
        was_corrected = result.get("was_corrected", False)
        corrections = result.get("corrections_made", 0)

        print(f"\nResults:")
        print(f"  Scores: F={scores.get('factuality')} S={scores.get('safety')} Q={scores.get('quality')}")
        print(f"  Iterations: {iterations}")
        print(f"  Was corrected: {was_corrected}")
        if was_corrected:
            print(f"  Corrections made: {corrections}")
            print(f"  ✅ System detected and fixed quality issues")
        else:
            print(f"  ✅ Response passed quality check first time")

        print(f"\n  Final response preview:")
        print(f"  {result.get('final_response', '')[:150]}...")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}


def run_all():
    print("\n🚀 Self-Correcting LLM Agent — Live Test")
    print(f"Sending real prompts — system generates and quality-checks automatically")
    print(f"Total scenarios: {len(SCENARIOS)}\n")

    results = []
    corrected_count = 0
    passed_first_time = 0
    errors = 0

    for i, scenario in enumerate(SCENARIOS):
        result = run_scenario(scenario, i + 1, len(SCENARIOS))
        results.append({
            "scenario": scenario["name"],
            "prompt": scenario["prompt"],
            "result": result
        })

        if "error" in result:
            errors += 1
        elif result.get("was_corrected"):
            corrected_count += 1
        else:
            passed_first_time += 1

        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total scenarios:      {len(SCENARIOS)}")
    print(f"Passed first time:    {passed_first_time}")
    print(f"Required correction:  {corrected_count}")
    print(f"Errors:               {errors}")
    print(f"{'='*60}")

    # DPO stats
    try:
        with open("data/finetuning_records.jsonl", "r") as f:
            dpo_count = sum(1 for line in f if line.strip())
        print(f"\nDPO training pairs: {dpo_count}")
        if dpo_count >= 10:
            print(f"⚡ Fine-tuning trigger reached!")
            print(f"Run: python3 scripts/finetune_lora.py")
    except FileNotFoundError:
        print("\nNo DPO records yet — all responses passed first time")

    # Save results
    with open("data/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to data/test_results.json")


if __name__ == "__main__":
    run_all()