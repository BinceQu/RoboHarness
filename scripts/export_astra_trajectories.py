#!/usr/bin/env python3
"""Export the counted Astra attempts as portable, credential-free trajectory bundles.

Keeps agent-visible conversation, tool calls/results, returned images and executed
joint commands. Simulator RGB-D caches, provider wire logs and runtime credentials
are deliberately outside the export schema. Outputs belong outside the git tree.
"""

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._~-]{15,})")
PRIVATE_URL = re.compile(r"https?://(?:localhost|127\.0\.0\.1|10(?:\.\d+){3}|192\.168(?:\.\d+){2}|100(?:\.\d+){3})(?::\d+)?")
SECRET_KEYS = {"api_key", "apikey", "access_token", "refresh_token", "authorization", "cookie", "password", "secret", "client_secret"}
DROP_KEYS = {"encrypted_content", "internal_chat_message_metadata_passthrough"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Bundle:
    def __init__(self, archive, case):
        self.archive, self.case = archive, case
        self.files, self.redactions = {}, 0

    def add(self, name, data):
        if name in self.files:
            return
        if isinstance(data, str):
            data = data.encode()
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        self.archive.addfile(info, io.BytesIO(data))
        self.files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def clean(self, value):
        if isinstance(value, dict):
            return {k: ("[REDACTED]" if k.lower() in SECRET_KEYS else self.clean(v))
                    for k, v in value.items() if k not in DROP_KEYS}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if not isinstance(value, str):
            return value
        if value.startswith("data:image/") and ";base64," in value:
            head, encoded = value.split(";base64,", 1)
            data = base64.b64decode(encoded)
            suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(head[5:], "bin")
            name = f"images/{hashlib.sha256(data).hexdigest()}.{suffix}"
            self.add(name, data)
            return name
        value = value.replace(str(self.case / "workspace"), "workspace").replace(str(self.case), "episode")
        value = re.sub(r"/(?:mnt/nas_nfs/home|home)/bince(?:/[^\s\"'<>]*)?", "[LOCAL_PATH]", value)
        value, count = SECRET.subn("[REDACTED_CREDENTIAL]", value)
        self.redactions += count
        return PRIVATE_URL.sub("http://robot-service", value)

    def json(self, name, data):
        self.add(name, json.dumps(self.clean(data), ensure_ascii=False, separators=(",", ":")) + "\n")

    def jsonl(self, name, source, session=False):
        rows = []
        with source.open(errors="replace") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # A killed writer may leave one incomplete final record.
                if session:
                    kind = event.get("type")
                    if kind not in {"response_item", "session_meta", "turn_context", "compacted"}:
                        continue
                    if kind == "session_meta":
                        event["payload"] = {k: v for k, v in event["payload"].items()
                                            if k in {"session_id", "id", "timestamp", "cli_version", "originator", "source"}}
                    elif kind == "turn_context":
                        event["payload"] = {k: v for k, v in event["payload"].items()
                                            if k in {"model", "effort", "summary", "approval_policy", "sandbox_policy"}}
                rows.append(json.dumps(self.clean(event), ensure_ascii=False, separators=(",", ":")))
        self.add(name, "\n".join(rows) + "\n")
        return len(rows)


def export_episode(run, out, episode, settings):
    key = episode["id"]
    target = out / "trajectories" / f"{key}.tar.gz"
    receipt = out / ".receipts" / f"{key}.json"
    if target.exists() and receipt.exists():
        cached = json.loads(receipt.read_text())
        if cached.get("q") == episode["q"] and cached.get("score_source") == episode["score_source"]:
            return cached
    job = run / "jobs" / f"{key}-i301-r0"
    complete = sorted(p for p in job.glob("attempt*") if (p / "summary.json").is_file()
                      and json.loads((p / "summary.json").read_text()).get("phase") == "complete")
    attempt = complete[-1] if complete else job / "attempt01"
    case = attempt / "instance_301"
    prompt = (case / "prompt.txt").read_text()
    if prompt != episode["prompt"]:
        raise ValueError(f"Prompt mismatch for {key}")
    partial = target.with_suffix(".partial")
    counts = {"tool_calls": 0, "sessions": 0, "conversation_records": 0}
    with tarfile.open(partial, "w:gz", compresslevel=4) as archive:
        bundle = Bundle(archive, case)
        bundle.add("prompt.txt", prompt)
        bundle.json("episode.json", {**episode, "source_attempt": attempt.name, "settings": settings})
        metric = run / "submission" / "metrics" / f"{episode['task']}_301_0.json"
        if metric.exists():
            bundle.json("metrics.json", json.loads(metric.read_text()))
        for index, path in enumerate(sorted(case.glob("native/*/stream.jsonl")), 1):
            bundle.jsonl(f"agent/stream-{index:02d}.jsonl", path)
        sessions = sorted((case / "workspace/.runtime/codex-home/sessions").rglob("*.jsonl"))
        for index, path in enumerate(sessions, 1):
            counts["conversation_records"] += bundle.jsonl(f"agent/conversation-{index:02d}.jsonl", path, session=True)
        counts["sessions"] = len(sessions)
        history = case / "workspace/tool_call_history"
        for call in sorted(history.iterdir()):
            if not call.is_dir() or call.is_symlink():
                continue
            counts["tool_calls"] += 1
            # Top-level images are the rendered observations/clicks returned to the agent.
            paths = list(call.iterdir()) + list((call / "observation").glob("*.json"))
            for path in sorted(paths):
                if not path.is_file() or path.is_symlink():
                    continue
                name = "tool_calls/" + str(path.relative_to(history))
                if path.suffix == ".json":
                    try:
                        bundle.json(name, json.loads(path.read_text()))
                    except ValueError:
                        continue
                elif path.suffix == ".jsonl":
                    bundle.jsonl(name, path)
                elif path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    bundle.add(name, path.read_bytes())
        workspace = case / "workspace"
        sources = [workspace / "AGENTS.md", workspace / "README.md"]
        for section in ("tools", "skills", ".agents/skills"):
            sources.extend((workspace / section).rglob("*"))
        for path in sorted(set(sources)):
            if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".sh"}:
                bundle.add("workspace/" + str(path.relative_to(workspace)), bundle.clean(path.read_text(errors="replace")))
        bundle.json("manifest.json", {"schema": 1, "id": key, "counts": counts,
                                     "credential_redactions": bundle.redactions, "files": dict(bundle.files)})
    partial.replace(target)
    result = {"id": key, "task": episode["task"], "q": episode["q"], "score_source": episode["score_source"],
              "attempt": attempt.name, "file": str(target.relative_to(out)), "bytes": target.stat().st_size,
              "sha256": digest(target), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **counts}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    for name in ("trajectories", ".receipts"):
        (args.out / name).mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "website/astra/episodes.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(export_episode, args.run, args.out, row, data["settings"]) for row in data["episodes"]]
        results = []
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            print(row["id"], row["tool_calls"], "calls", round(row["bytes"] / 1e6, 1), "MB", flush=True)
    results.sort(key=lambda x: x["id"])
    (args.out / "trajectories.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "episodes.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    (args.out / "estimates.json").write_text((ROOT / "website/astra/estimates.json").read_text())
    fields = ["id", "task", "q", "score_source", "attempt", "tool_calls", "sessions", "bytes", "file", "sha256"]
    with (args.out / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print("COMPLETE", len(results), "episodes", sum(r["bytes"] for r in results), "bytes", flush=True)


if __name__ == "__main__":
    main()
