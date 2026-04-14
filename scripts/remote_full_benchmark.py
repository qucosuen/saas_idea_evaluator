"""
Full pipeline benchmark on all 30 jobs. Saves detailed per-stage metrics.
Run: python scripts/remote_full_benchmark.py
"""
import json, os, sys, time, re, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---- Inline metrics (avoid import issues on remote) ----

ACTION_VERBS = {"collect","gather","review","verify","enter","process","submit","check","validate","update","create","generate","send","prepare","analyze","compile","organize","schedule","assign","approve","monitor","track","report","file","archive","distribute","sort","inspect","calculate","reconcile","coordinate","manage","evaluate","document","record","input","extract","transfer","notify","escalate"}
VAGUE_TERMS = {"do stuff","handle things","work on it","take care of","deal with","do the needful","various tasks","miscellaneous","etc","and so on"}
PROBLEM_KW = {"time-consuming","repetitive","error-prone","manual","tedious","slow","inefficient"}
VALID_SOLUTIONS = {"automation script","ocr extraction","api integration","dashboard","notification system"}

def score_s1(output, patterns):
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    step_pat = re.compile(r"^Step\s+\d+\s*:", re.IGNORECASE)
    steps = [l for l in lines if step_pat.match(l)]
    fmt = 1.0 if len(steps)==5 else (len(steps)/5.0)*0.7
    tl = output.lower()
    vhits = sum(1 for v in ACTION_VERBS if v in tl)
    vague = sum(1 for v in VAGUE_TERMS if v in tl)
    conc = min(1.0, vhits/5.0) * (1.0 - min(1.0, vague/2.0))
    phits = sum(1 for p in patterns if p.lower() in tl)
    rel = phits / max(len(patterns),1)
    ss = 0.40*fmt + 0.30*conc + 0.30*rel
    return {"format":round(fmt,3),"concreteness":round(conc,3),"relevance":round(rel,3),"stage_score":round(ss,3),"step_count":len(steps)}

def score_s2(output, wf_steps):
    tl = output.lower()
    kh = sum(1 for k in PROBLEM_KW if k in tl)
    constraint = min(1.0, kh/2.0)
    probs = re.findall(r"^(\d+[\.\):]|\-|\*)\s+", output, re.MULTILINE)
    pc = len(probs) if probs else len([l for l in output.strip().split("\n") if l.strip()])
    coverage = 1.0 if pc==5 else max(0, 1.0-abs(pc-5)/5.0)
    ah = 0
    for s in wf_steps:
        if any(w in tl for w in s.lower().split() if len(w)>4): ah+=1
    alignment = ah/max(len(wf_steps),1)
    ss = 0.30*constraint + 0.35*coverage + 0.35*alignment
    return {"constraint":round(constraint,3),"coverage":round(coverage,3),"alignment":round(alignment,3),"stage_score":round(ss,3),"problem_count":pc}

def score_s3(output, pc):
    tl = output.lower()
    sh = sum(1 for s in VALID_SOLUTIONS if s in tl)
    valid = min(1.0, sh/max(pc,1))
    nums = re.findall(r"^(\d+[\.\):]|\-|\*)\s+", output, re.MULTILINE)
    sc = len(nums) if nums else len([l for l in output.strip().split("\n") if l.strip()])
    cov = 1.0 if sc>=pc else sc/max(pc,1)
    fp = {"repetitive":"automation script","manual":"automation script","data entry":"ocr extraction","reporting":"dashboard","notification":"notification system","integration":"api integration"}
    fh = sum(1 for k,v in fp.items() if k in tl and v in tl)
    fit = min(1.0, fh/3.0)
    ss = 0.35*valid + 0.30*cov + 0.35*fit
    return {"valid_choices":round(valid,3),"coverage":round(cov,3),"fit":round(fit,3),"stage_score":round(ss,3)}

def score_s4(output):
    tl = output.lower()
    has_scores = bool(re.search(r"\d\s*/\s*5|\bscore\b.*\d|:\s*[1-5]\b", tl))
    fmt = 1.0 if has_scores else 0.3
    sf = [int(x) for x in re.findall(r"\b([1-5])\b(?:\s*/\s*5)?", output)]
    rng = 1.0 if (sf and all(1<=s<=5 for s in sf) and len(sf)>=3) else len(sf)/10.0
    cons = sum(["feasibility" in tl or "feasible" in tl, "impact" in tl, "complexity" in tl or "complex" in tl])/3.0
    ss = 0.35*fmt + 0.35*rng + 0.30*cons
    return {"format":round(fmt,3),"range":round(rng,3),"consistency":round(cons,3),"stage_score":round(ss,3)}

def score_useful(output):
    feas = [int(f) for f in re.findall(r"feasibility[:\s]*([1-5])", output.lower())]
    imp = [int(i) for i in re.findall(r"impact[:\s]*([1-5])", output.lower())]
    hf = sum(1 for f in feas if f>=4)
    hi = sum(1 for i in imp if i>=4)
    return 1.0 if (hf>=2 and hi>=2) else 0.0


# ---- Prompts ----
S1SYS = "You are a workflow analyst. Given a job title, generate exactly 5 workflow steps. Each step must start with 'Step X:' where X is the step number. Be specific and use action verbs."
S2SYS = "You are a process analyst. Given a workflow, identify exactly 5 problems. Each problem must be categorized as: time-consuming, repetitive, error-prone, manual, or tedious. Number each problem."
S3SYS = "You are a solutions architect. Given problems, propose one solution per problem from: Automation Script, OCR Extraction, API Integration, Dashboard, Notification System. Number each."
S4SYS = "You are a project evaluator. For each solution, score Feasibility (1-5), Impact (1-5), Complexity (1-5). Format: solution name then three scores."

def run_stage(model, sys_prompt, user_prompt, max_tok=250):
    start = time.time()
    r = model.create_chat_completion(
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
        max_tokens=max_tok, temperature=0.0)
    elapsed = time.time() - start
    return r["choices"][0]["message"]["content"], elapsed, r["usage"]["completion_tokens"]

def run_pipeline(model, job, patterns):
    out1, t1, tok1 = run_stage(model, S1SYS, f"Generate a 5-step daily workflow for: {job}")
    s1 = score_s1(out1, patterns)
    wf_lines = [l.strip() for l in out1.split("\n") if l.strip()]

    out2, t2, tok2 = run_stage(model, S2SYS, f"Identify problems in this workflow:\n\n{out1}")
    s2 = score_s2(out2, wf_lines)

    out3, t3, tok3 = run_stage(model, S3SYS, f"Propose solutions for these problems:\n\n{out2}")
    s3 = score_s3(out3, s2["problem_count"])

    out4, t4, tok4 = run_stage(model, S4SYS, f"Evaluate these solutions:\n\n{out3}")
    s4 = score_s4(out4)
    useful = score_useful(out4)

    final = 0.25*s1["stage_score"] + 0.20*s2["stage_score"] + 0.20*s3["stage_score"] + 0.20*s4["stage_score"] + 0.15*useful
    total_ms = (t1+t2+t3+t4)*1000
    return {
        "job": job, "final_score": round(final,3),
        "stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4,
        "usefulness": useful,
        "latency": {"s1_ms":round(t1*1000),"s2_ms":round(t2*1000),"s3_ms":round(t3*1000),"s4_ms":round(t4*1000),"total_ms":round(total_ms)},
        "tokens": {"s1":tok1,"s2":tok2,"s3":tok3,"s4":tok4,"total":tok1+tok2+tok3+tok4},
    }

def main():
    from llama_cpp import Llama
    ds_path = Path(__file__).parent.parent / "src" / "benchmark" / "dataset.json"
    with open(ds_path) as f:
        dataset = json.load(f)

    model_path = Path(__file__).parent.parent / "models" / "gguf" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    print(f"Model: {model_path.name}")
    print(f"Jobs: {len(dataset)}")
    model = Llama(model_path=str(model_path), n_ctx=2048, n_threads=os.cpu_count() or 8, verbose=False)

    results = []
    for i, jd in enumerate(dataset):
        job, diff, pats = jd["job"], jd["difficulty"], jd["expected_patterns"]
        print(f"[{i+1}/{len(dataset)}] {job} ({diff})", end=" ... ", flush=True)
        r = run_pipeline(model, job, pats)
        r["difficulty"] = diff
        results.append(r)
        print(f"score={r['final_score']:.3f} latency={r['latency']['total_ms']}ms")

    # Aggregate
    finals = [r["final_score"] for r in results]
    by_diff = {}
    for diff in ["easy","medium","hard"]:
        ds = [r["final_score"] for r in results if r["difficulty"]==diff]
        if ds: by_diff[diff] = {"mean":round(statistics.mean(ds),3),"count":len(ds)}

    stage_avgs = {}
    for sn in ["stage1","stage2","stage3","stage4"]:
        vals = [r[sn]["stage_score"] for r in results]
        stage_avgs[sn] = {"mean":round(statistics.mean(vals),3),"std":round(statistics.stdev(vals),3) if len(vals)>1 else 0}

    # Stage 1 sub-metric averages
    s1_fmt = [r["stage1"]["format"] for r in results]
    s1_conc = [r["stage1"]["concreteness"] for r in results]
    s1_rel = [r["stage1"]["relevance"] for r in results]

    report = {
        "model": "qwen2.5-1.5b-instruct-q4_k_m",
        "jobs": len(dataset),
        "final_score": {"mean":round(statistics.mean(finals),3),"std":round(statistics.stdev(finals),3),"min":round(min(finals),3),"max":round(max(finals),3)},
        "stage_scores": stage_avgs,
        "stage1_detail": {"format_mean":round(statistics.mean(s1_fmt),3),"concreteness_mean":round(statistics.mean(s1_conc),3),"relevance_mean":round(statistics.mean(s1_rel),3)},
        "by_difficulty": by_diff,
        "usefulness_rate": round(sum(1 for r in results if r["usefulness"]>0)/len(results),3),
        "per_job": results,
    }

    out = Path(__file__).parent.parent / "results" / "full_pipeline_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
