#!/bin/bash
# =============================================================================
# Self-healing deploy script — tries multiple approaches, auto-fixes errors
# Usage: ./scripts/deploy_remote.sh user@host
# =============================================================================
set -o pipefail

REMOTE="${1:?Usage: $0 user@host}"
USER="${REMOTE%%@*}"
RDIR="C:/Users/${USER}/slm_evaluator"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$LOCAL/deploy.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run_remote() { ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" "$@" 2>&1; }

log "============================================"
log "  Deploy to $REMOTE"
log "  Log: $LOG"
log "============================================"

# --- Test SSH connection ---
log "[0] Testing SSH connection..."
if ! run_remote "echo SSH_OK" | grep -q "SSH_OK"; then
    log "FATAL: Cannot SSH to $REMOTE. Check key auth: ssh-copy-id $REMOTE"
    exit 1
fi
log "  SSH connection OK"

# --- Detect remote OS and tools ---
log "[1] Detecting remote environment..."
REMOTE_OS=$(run_remote "echo %OS%" | head -1)
HAS_TAR=$(run_remote "where tar" 2>/dev/null | grep -c "tar" || echo 0)
HAS_PYTHON=$(run_remote "where python" 2>/dev/null | grep -c "python" || echo 0)
log "  OS: $REMOTE_OS, tar: $HAS_TAR, python: $HAS_PYTHON"

if [ "$HAS_PYTHON" = "0" ]; then
    log "FATAL: Python not found on remote. Install Python 3.10+ first."
    exit 1
fi

# --- Create directories ---
log "[2] Creating remote directories..."
# Try PowerShell mkdir (works on all Windows with OpenSSH)
run_remote "powershell -Command \"New-Item -ItemType Directory -Force -Path '$RDIR/src/benchmark','$RDIR/src/pipeline','$RDIR/src/evaluator','$RDIR/src/data','$RDIR/src/training','$RDIR/src/tracking','$RDIR/scripts','$RDIR/data','$RDIR/results/benchmark','$RDIR/models/gguf','$RDIR/ui' | Out-Null\"" \
    && log "  Directories created" \
    || { log "  PowerShell mkdir failed, trying cmd..."; \
         run_remote "mkdir \"${RDIR//\//\\}\" 2>nul & mkdir \"${RDIR//\//\\}\\src\\benchmark\" 2>nul & mkdir \"${RDIR//\//\\}\\scripts\" 2>nul & mkdir \"${RDIR//\//\\}\\data\" 2>nul & mkdir \"${RDIR//\//\\}\\results\" 2>nul & mkdir \"${RDIR//\//\\}\\models\\gguf\" 2>nul"; }

# --- Copy files ---
log "[3] Copying project files..."

# Method 1: tar + scp (fastest if tar available on both sides)
if [ "$HAS_TAR" != "0" ]; then
    log "  Using tar+scp method..."
    cd "$LOCAL"
    tar czf /tmp/slm_deploy.tar.gz \
        --exclude='.venv' --exclude='models/gguf/*.gguf' \
        --exclude='__pycache__' --exclude='.cache' --exclude='*.pyc' \
        --exclude='.git' --exclude='feast/feature_repo/data' \
        --exclude='node_modules' --exclude='.env' . 2>/dev/null

    scp -q /tmp/slm_deploy.tar.gz "$REMOTE:$RDIR/" \
        && run_remote "cd $RDIR && tar xzf slm_deploy.tar.gz && del slm_deploy.tar.gz" \
        && log "  Files copied via tar+scp" \
        || { log "  tar extract failed on remote, falling back to scp -r..."; HAS_TAR=0; }
    rm -f /tmp/slm_deploy.tar.gz
fi

# Method 2: scp -r (slower but always works)
if [ "$HAS_TAR" = "0" ]; then
    log "  Using scp -r method (slower)..."
    cd "$LOCAL"
    # Copy key directories individually
    for dir in src scripts data results ui; do
        log "    Copying $dir/..."
        scp -rq "$dir" "$REMOTE:$RDIR/" 2>&1 | tail -1 || true
    done
    # Copy root files
    for f in evaluate.py evaluate_fast.py README.md requirements.txt; do
        [ -f "$f" ] && scp -q "$f" "$REMOTE:$RDIR/" 2>/dev/null || true
    done
    log "  Files copied via scp -r"
fi

# --- Setup Python venv ---
log "[4] Setting up Python environment..."
VENV_EXISTS=$(run_remote "if exist \"$RDIR\\.venv\\Scripts\\python.exe\" (echo YES) else (echo NO)" | tr -d '\r')
if [ "$VENV_EXISTS" = "YES" ]; then
    log "  Venv already exists, skipping creation"
else
    run_remote "cd $RDIR && python -m venv .venv" \
        && log "  Venv created" \
        || { log "ERROR: venv creation failed"; exit 1; }
fi
run_remote "cd $RDIR && .venv\\Scripts\\python -m pip install --upgrade pip -q" \
    && log "  pip upgraded" || log "  pip upgrade failed (non-fatal)"

# --- Install dependencies ---
log "[5] Installing dependencies..."

# Check if torch already installed
TORCH_OK=$(run_remote "cd $RDIR && .venv\\Scripts\\python -c \"import torch; print('OK')\"" 2>/dev/null | tr -d '\r')
if [ "$TORCH_OK" = "OK" ]; then
    log "  PyTorch already installed, skipping"
else
    log "  Installing PyTorch with CUDA (this takes a few minutes)..."
    run_remote "cd $RDIR && .venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu121 -q" \
        && log "  PyTorch installed" \
        || { log "  CUDA torch failed, trying CPU torch..."; \
             run_remote "cd $RDIR && .venv\\Scripts\\pip install torch -q"; }
fi

# Install other packages
log "  Installing ML packages..."
run_remote "cd $RDIR && .venv\\Scripts\\pip install transformers accelerate huggingface-hub -q" \
    && log "  transformers+accelerate installed" || log "  WARNING: some packages failed"

# Try llama-cpp-python (may need cmake on Windows)
LLAMA_OK=$(run_remote "cd $RDIR && .venv\\Scripts\\python -c \"from llama_cpp import Llama; print('OK')\"" 2>/dev/null | tr -d '\r')
if [ "$LLAMA_OK" = "OK" ]; then
    log "  llama-cpp-python already installed"
else
    log "  Installing llama-cpp-python..."
    # Try prebuilt wheel first, then source
    run_remote "cd $RDIR && .venv\\Scripts\\pip install llama-cpp-python -q" \
        && log "  llama-cpp-python installed" \
        || { log "  Prebuilt failed, trying with CUDA cmake..."; \
             run_remote "cd $RDIR && set CMAKE_ARGS=-DGGML_CUDA=on && .venv\\Scripts\\pip install llama-cpp-python --no-cache-dir -q" \
             && log "  llama-cpp-python installed with CUDA" \
             || log "  WARNING: llama-cpp-python install failed. Will use transformers backend."; }
fi

# --- Download model ---
log "[6] Downloading GGUF model..."
MODEL_EXISTS=$(run_remote "if exist \"$RDIR\\models\\gguf\\qwen2.5-1.5b-instruct-q4_k_m.gguf\" (echo YES) else (echo NO)" | tr -d '\r')
if [ "$MODEL_EXISTS" = "YES" ]; then
    log "  Model already cached"
else
    log "  Downloading Qwen2.5-1.5B Q4_K_M (~1.1GB)..."
    run_remote "cd $RDIR && .venv\\Scripts\\python -c \"from huggingface_hub import hf_hub_download; import os; os.makedirs('models/gguf',exist_ok=True); hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf',local_dir='models/gguf'); print('DONE')\"" \
        && log "  Model downloaded" \
        || log "  WARNING: Model download failed. Run manually on remote."
fi

# --- Run evaluation ---
log "[7] Running GPU evaluation..."
run_remote "cd $RDIR && .venv\\Scripts\\python scripts/run_gpu_eval.py" 2>&1 | tee -a "$LOG" \
    || log "  WARNING: Evaluation had errors (check log)"

# --- Copy results back ---
log "[8] Copying results back..."
mkdir -p "$LOCAL/results"
scp -q "$REMOTE:$RDIR/results/gpu_benchmark/gpu_eval_results.json" "$LOCAL/results/gpu_eval_results.json" 2>/dev/null \
    && log "  Results copied to results/gpu_eval_results.json" \
    || log "  No results file to copy (eval may not have completed)"

log ""
log "============================================"
log "  Deploy finished! Check $LOG for details."
log "============================================"
