"""Run pipeline 10 times with CPU limited via CPU_PERCENT from .env (default 50)."""
import json, time, os, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.pipeline.cpu_config import get_thread_count, get_cpu_info
from src.benchmark.prompts import (
    STAGE1_SYSTEM, STAGE1_USER, STAGE2_SYSTEM, STAGE2_USER,
    STAGE3_SYSTEM, STAGE3_USER, STAGE4_SYSTEM, STAGE4_USER,
)

JOB = "Accountant"
MODEL = "models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"
RUNS = int(os.environ.get("BENCH_RUNS", "5"))

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
    cpu = get_cpu_info()
    threads = cpu["used_threads"]
    print(f"CPU config: {threads}/{cpu['total_threads']} threads ({cpu['target_percent']}% target, {cpu['actual_percent']}% actual)")

    model = Llama(model_path=MODEL, n_ctx=2048, n_threads=threads, verbose=False)
    model.create_chat_completion(messages=[{"role":"user","content":"Hi"}],max_tokens=5,temperature=0.0)

    all_runs = []
    for i in range(RUNS):
        r = run_pipeline(model)
        all_runs.append(r)
        print(f"Run {i+1}/{RUNS}: total={r['total_ms']:.0f}ms")

    avg = {k: round(statistics.mean(r[k] for r in all_runs),1) for k in all_runs[0]}
    std = {k: round(statistics.stdev(r[k] for r in all_runs),1) for k in all_runs[0]}

    print(f"\nAVERAGE ({threads} threads, ~{cpu['actual_percent']}% CPU):")
    for k in ["s1_ms","s2_ms","s3_ms","s4_ms","total_ms"]:
        print(f"  {k}: {avg[k]:.0f} +/- {std[k]:.0f}ms")

    result = {"job":JOB,"model":"Qwen2.5-1.5B-Instruct","runs":RUNS,
              "cpu":cpu,"all_runs":all_runs,"avg":avg,"std":std}
    os.makedirs("results",exist_ok=True)
    with open("results/latency_10runs_cpu_limited.json","w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/latency_10runs_cpu_limited.json")

if __name__=="__main__":
    main()
