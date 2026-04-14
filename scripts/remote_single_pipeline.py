"""Run one full pipeline on the champion model, report per-stage latency."""
import json, time, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.benchmark.prompts import (
    STAGE1_SYSTEM, STAGE1_USER, STAGE2_SYSTEM, STAGE2_USER,
    STAGE3_SYSTEM, STAGE3_USER, STAGE4_SYSTEM, STAGE4_USER,
)
from src.benchmark.metrics import (
    score_stage1_workflow, score_stage2_problems,
    score_stage3_solutions, score_stage4_evaluation, score_usefulness,
)

JOB = "Accountant"
PATTERNS = ["data entry", "verification", "reporting", "reconciliation", "audit"]
MODEL = "models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"

def gen(model, system, user):
    start = time.perf_counter()
    r = model.create_chat_completion(
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=250, temperature=0.0)
    ms = (time.perf_counter() - start) * 1000
    out = r["choices"][0]["message"]["content"]
    tok = r["usage"]["completion_tokens"]
    return out, ms, tok

def main():
    threads = os.cpu_count() or 8
    print(f"Loading champion model ({threads} threads)...")
    model = Llama(model_path=MODEL, n_ctx=2048, n_threads=threads, verbose=False)

    # Warmup
    model.create_chat_completion(
        messages=[{"role":"user","content":"Hi"}], max_tokens=5, temperature=0.0)

    stages = []

    # Stage 1
    out1, ms1, tok1 = gen(model, STAGE1_SYSTEM, STAGE1_USER.format(job=JOB))
    s1 = score_stage1_workflow(out1, PATTERNS)
    stages.append({"name":"workflow_generator","latency_ms":round(ms1,1),"tokens":tok1,"scores":s1})
    print(f"Stage 1: {ms1:.0f}ms  score={s1['stage_score']:.3f}")

    # Stage 2
    out2, ms2, tok2 = gen(model, STAGE2_SYSTEM, STAGE2_USER.format(workflow=out1))
    s2 = score_stage2_problems(out2, out1.split("\n"))
    stages.append({"name":"problem_extractor","latency_ms":round(ms2,1),"tokens":tok2,"scores":s2})
    print(f"Stage 2: {ms2:.0f}ms  score={s2['stage_score']:.3f}")

    # Stage 3
    out3, ms3, tok3 = gen(model, STAGE3_SYSTEM, STAGE3_USER.format(problems=out2))
    s3 = score_stage3_solutions(out3, s2.get("problem_count",5))
    stages.append({"name":"solution_mapper","latency_ms":round(ms3,1),"tokens":tok3,"scores":s3})
    print(f"Stage 3: {ms3:.0f}ms  score={s3['stage_score']:.3f}")

    # Stage 4
    out4, ms4, tok4 = gen(model, STAGE4_SYSTEM, STAGE4_USER.format(solutions=out3))
    s4 = score_stage4_evaluation(out4)
    e2e = score_usefulness(out4)
    stages.append({"name":"evaluator","latency_ms":round(ms4,1),"tokens":tok4,"scores":s4,"usefulness":e2e})
    print(f"Stage 4: {ms4:.0f}ms  score={s4['stage_score']:.3f}")

    total_ms = sum(s["latency_ms"] for s in stages)
    final = 0.25*s1["stage_score"]+0.20*s2["stage_score"]+0.20*s3["stage_score"]+0.20*s4["stage_score"]+0.15*e2e["usefulness_score"]

    result = {
        "job": JOB, "model": "Qwen2.5-1.5B-Instruct",
        "hardware": {"cpu_threads": threads, "machine": "windows_remote"},
        "stages": stages,
        "total_latency_ms": round(total_ms, 1),
        "final_score": round(final, 3),
    }
    print(f"\nTotal: {total_ms:.0f}ms  Final score: {final:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/single_pipeline_result.json","w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/single_pipeline_result.json")

if __name__ == "__main__":
    main()
