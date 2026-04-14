"""Test if GPU inference works with llama-cpp-python."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama_cpp import Llama

MODEL = "models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("Loading model with n_gpu_layers=-1 (all layers on GPU)...")
try:
    m = Llama(model_path=MODEL, n_ctx=512, n_gpu_layers=-1, verbose=True)
    print("\nModel loaded. Running inference...")
    start = time.perf_counter()
    r = m.create_chat_completion(
        messages=[{"role":"user","content":"Say hello in one word."}],
        max_tokens=10, temperature=0.0)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\nOutput: {r['choices'][0]['message']['content']}")
    print(f"Latency: {elapsed:.0f}ms")
    print("GPU_TEST_PASSED")
except Exception as e:
    print(f"GPU_TEST_FAILED: {e}")
