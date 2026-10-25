"""Read-only human monitor. Never dispatch robot tools or evaluator actions."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tools.util.files import load_json
from tools.util.protocol import request

PAGE = '''<!doctype html><meta charset="utf-8"><title>R1 Pro workspace</title>
<style>body{font:16px system-ui;margin:30px;background:#f7f8fa;color:#172536}img{max-width:100%}
section{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap}a{color:#14619c}</style>
<h1>R1 Pro · current perception</h1><p id="status">Waiting for a session.</p>
<section><div><h2>Head RGB</h2><img id="head"></div><div><h2>Measured robot posture</h2><img id="body"></div></section>
<h2>Recent tool calls</h2><div id="calls"></div>
<script>async function refresh(){let s=await(await fetch('/state')).json();
document.querySelector('#status').textContent=JSON.stringify(s.driver);
if(s.latest){let t=s.latest.sequence;document.querySelector('#head').src='/file/perception/exteroception/head_rgb.png?v='+t;
document.querySelector('#body').src='/file/perception/proprioception/robot_three_views.png?v='+t}
let box=document.querySelector('#calls');box.replaceChildren();for(let c of s.calls||[]){let p=document.createElement('p');
let a=document.createElement('a');a.href='/file/tool_call_history/'+c+'/result.json';a.textContent=c;p.append(a);box.append(p)}}
refresh();setInterval(refresh,5000)</script>'''


def handler(run_dir, socket):
    class Monitor(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            route = urlsplit(self.path).path
            binding = load_json(run_dir / "active_session.json", {})
            workspace = Path(binding["workspace"]) if binding.get("workspace") else None
            if route == "/":
                body, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif route.startswith("/file/") and workspace:
                path = (workspace / unquote(route[6:])).resolve()
                if not path.is_relative_to(workspace.resolve()) or not path.is_file() or any(p.startswith(".") for p in path.relative_to(workspace).parts):
                    self.send_error(404); return
                body, mime = path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            elif route in ("/state", "/api/agent_monitor", "/__official__/idle_probe", "/__official__/healthz"):
                try:
                    status = request(socket, {"operation": "status"}, timeout=2)
                except (OSError, ValueError):
                    status = {}
                active = bool(status.get("pending_action") or status.get("inflight_action"))
                value = {"ok": True, "driver": status, "session_ticks": status.get("ticks"),
                    "episode_initialization": status.get("initialization", {}), "idle": not active,
                    "action_source": "downstream-model" if active else "idle",
                    "latest": load_json(workspace / "perception/latest.json", {}) if workspace else {},
                    "calls": sorted((p.name for p in (workspace / "tool_call_history").glob("[0-9]*") if p.is_dir()), reverse=True)[:50] if workspace else []}
                body, mime = json.dumps(value).encode(), "application/json"
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
    return Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    options = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", options.port), handler(options.run_dir.resolve(), options.socket)).serve_forever()


if __name__ == "__main__":
    main()
