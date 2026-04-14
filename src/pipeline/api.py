"""
REST API for the local-first LLM pipeline.

Usage:
  # Start the server
  python -m src.pipeline.api

  # Query it
  curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"job": "Accountant"}'
  curl http://localhost:8000/health
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

from src.pipeline.model import find_model, load_model
from src.pipeline.runner import PipelineRunner

# Global pipeline runner (initialized on startup)
_runner: PipelineRunner | None = None


def get_runner() -> PipelineRunner:
    global _runner
    if _runner is None:
        model_path = find_model()
        print(f"Loading model: {model_path.name}", file=sys.stderr)
        model = load_model(model_path)
        _runner = PipelineRunner(model, max_tokens=350, max_retries=2, cache_dir=".cache/pipeline")
    return _runner


class PipelineHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": "loaded" if _runner else "not loaded"})
        else:
            self._send_json(404, {"error": "Not found. Use POST /analyze or GET /health"})

    def do_POST(self):
        if self.path != "/analyze":
            self._send_json(404, {"error": "Not found. Use POST /analyze"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        job = data.get("job", "").strip()
        if not job:
            self._send_json(400, {"error": "Missing 'job' field"})
            return

        runner = get_runner()
        result = runner.run(job)
        self._send_json(200, result.to_dict())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    port = int(os.environ.get("PORT", 8000))
    # Pre-load model
    get_runner()
    server = HTTPServer(("0.0.0.0", port), PipelineHandler)
    print(f"Pipeline API running on http://localhost:{port}", file=sys.stderr)
    print(f"  POST /analyze  {{\"job\": \"Accountant\"}}", file=sys.stderr)
    print(f"  GET  /health", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
