"""Run the full pipeline 10 times sequentially, record per-stage latency each run."""
import json, time, os, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.benchmark.prompts import (
    STAGE1_SYSTEM, STAGE1_USER, STAGE2_SYSTEM, STAGE2_USER,
    STAGE3_SYSTEM, STAGE3_USER, STAGE4_SYSTEM, STAGE4_USER,
)

JOB = "Accountant"
MODEL = "models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"
RUNS = 10

def gen(model, system, user):
    start = time.perf_counter()
    r = model.create_chat_completion(
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=250, temperature=0.0)
    ms = (time.perf_counter() - start) * 1000
    return r["choices"][0]["message"]["content"], ms

def run_pipeline(model):
    out1, ms1 = gen(model, STAGE1_SYSTEM, STAGE1_USER.format(job=JOB))
    out2, ms2 = gen(model, STAGE2_SYSTEM, STAGE2_USER.format(workflow=out1))
    out3, ms3 = gen(model, STAGE3_SYSTEM, STAGE3_USER.format(problems=out2))
    out4, ms4 = gen(model, STAGE4_SYSTEM, STAGE4_USER.format(solutions=out3))
    return {"s1_ms":round(ms1,1),"s2_ms":round(ms2,1),"s3_ms":round(ms3,1),"s4_ms":round(ms4,1),
            "total_ms":round(ms1+ms2+ms3+ms4,1)}

def main():
    threads = os.cpu_count() or 8
    print(f"Loading model ({threads} threads)...")
    model = Llama(model_path=MODEL, n_ctx=2048, n_threads=threads, verbose=False)
    # Warmup
    model.create_chat_completion(messages=[{"role":"user","content":"Hi"}],max_tokens=5,temperature=0.0)

    all_runs = []
    for i in range(RUNS):
        r = run_pipeline(model)
        all_runs.append(r)
        print(f"Run {i+1}/{RUNS}: s1={r['s1_ms']:.0f} s2={r['s2_ms']:.0f} s3={r['s3_ms']:.0f} s4={r['s4_ms']:.0f} total={r['total_ms']:.0f}ms")

    avg = {k: round(statistics.mean(r[k] for r in all_runs),1) for k in all_runs[0]}
    std = {k: round(statistics.stdev(r[k] for r in all_runs),1) for k in all_runs[0]}
    mn = {k: round(min(r[k] for r in all_runs),1) for k in all_runs[0]}
    mx = {k: round(max(r[k] for r in all_runs),1) for k in all_runs[0]}

    print(f"\n{'='*60}")
    print(f"  AVERAGE over {RUNS} runs:")
    for k in ["s1_ms","s2_ms","s3_ms","s4_ms","total_ms"]:
        print(f"  {k}: {avg[k]:.0f} +/- {std[k]:.0f}ms (min={mn[k]:.0f}, max={mx[k]:.0f})")

    result = {"job":JOB,"model":"Qwen2.5-1.5B-Instruct","runs":RUNS,
              "hardware":{"cpu_threads":threads},"all_runs":all_runs,
              "avg":avg,"std":std,"min":mn,"max":mx}
    os.makedirs("results",exist_ok=True)
    with open("results/latency_10runs.json","w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/latency_10runs.json")

if __name__=="__main__":
    main()
