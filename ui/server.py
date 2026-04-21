"""Dashboard server. Run: python ui/server.py → http://localhost:3000"""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent.parent
PORT = int(os.environ.get("DASHBOARD_PORT", 3000))


def load_json(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None


def get_data():
    d = {}

    # Pipeline runs
    runs = []
    results_dir = ROOT / "results"
    if results_dir.exists():
        for f in sorted(results_dir.glob("pipeline_*.json")):
            try:
                r = json.load(open(f))
                if r and r.get("stages"):
                    runs.append(r)
            except (json.JSONDecodeError, KeyError):
                continue
    d["pipeline_runs"] = runs

    d["champion"] = load_json("results/champion_selection.json") or {}

    return d


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            body = json.dumps(get_data()).encode()
            ct = "application/json"
            code = 200
        elif self.path in ("/", "/index.html"):
            body = (ROOT / "ui" / "app.html").read_bytes()
            ct = "text/html"
            code = 200
        else:
            body = b"Not found"
            ct = "text/plain"
            code = 404
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
