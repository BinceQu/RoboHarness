#!/usr/bin/env python3
"""Read-only workspace trajectory browser. No robot imports or action endpoints."""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
import functools
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import shlex
import threading
from urllib.parse import parse_qs, quote, urlsplit

WEB = Path(__file__).parent / "web"
NAME = re.compile(r"[A-Za-z0-9_.-]+/instance_[0-9]+\Z")
CALL = re.compile(r"[0-9]{6}_[a-z0-9_]+\Z")
DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def blocks(event):
    content = event.get("message", {}).get("content", [])
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


def native_paths(case):
    """Claude Code project transcripts, or pi's per-case session files."""
    return sorted([*(case / "workspace/.runtime/claude-home/projects").glob("*/*.jsonl"),
                   *(case / "pi-sessions").glob("*.jsonl")])


def pi_block(block):
    if not isinstance(block, dict):
        return block
    if block.get("type") == "toolCall":
        return {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": block.get("arguments", {})}
    if block.get("type") == "image" and "data" in block:
        return {"type": "image", "source": {"type": "base64", "media_type": block.get("mimeType"), "data": block["data"]}}
    return block


def pi_event(event):
    """Present a pi session line in the Claude transcript shape read below.

    pi writes one message per line: an assistant reply with all its tool calls,
    then one toolResult per call. Its compaction summary and injected custom
    messages are model-facing context. Session header, model/thinking changes
    and the system prompt pass through unchanged (shown only as raw records).
    """
    base = {"uuid": event.get("id"), "timestamp": event.get("timestamp"), "pi_type": event.get("type")}
    if event.get("type") == "compaction":
        return {**base, "type": "user", "isCompactSummary": True,
                "message": {"content": [{"type": "text", "text": event.get("summary", "")}]}}
    if event.get("type") == "custom_message":
        return {**base, "type": "attachment",
                "attachment": {"type": event.get("customType") or "context", "content": event.get("content")}}
    message = event.get("message")
    if event.get("type") != "message" or not isinstance(message, dict):
        return event
    content = message.get("content")
    content = [pi_block(b) for b in content] if isinstance(content, list) else content
    role = message.get("role")
    if role == "assistant":
        return {**base, "type": "assistant", "message": {
            "id": event.get("id"), "model": message.get("model"), "stop_reason": message.get("stopReason"),
            "usage": message.get("usage"), "content": content}}
    if role == "toolResult":
        return {**base, "type": "user", "message": {"content": [{
            "type": "tool_result", "tool_use_id": message.get("toolCallId"), "content": content,
            "is_error": bool(message.get("isError"))}]}}
    if role == "user":
        return {**base, "type": "user", "message": {"content": content}}
    return event


def texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from texts(item)
    elif isinstance(value, dict):
        if value.get("type") == "text":
            yield value.get("text", "")


def path_label(value, workspace, cwd=None):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(cwd or workspace) / path
    # Do not resolve live 'latest' symlinks: the transcript names what was read.
    # A case's workspace link may point at the private SSD copy the agent used.
    for root in (workspace, real_workspace(workspace)):
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


@functools.lru_cache(maxsize=64)
def real_workspace(workspace):
    return Path(workspace).resolve()


def accesses(event, workspace):
    """Conservative transcript evidence, never infer reading from execution."""
    found = []

    def add(kind, path, evidence):
        label = path_label(path, workspace, event.get("cwd"))
        if label:
            found.append({"kind": kind, "path": label, "evidence": evidence})

    if event.get("type") != "assistant":
        return found
    for block in blocks(event):
        if block.get("type") != "tool_use":
            continue
        # pi names its tools read/bash/grep/find/ls and reads 'path'.
        name, args = block.get("name"), block.get("input", {})
        if name in ("Read", "read"):
            add("read", args.get("file_path") or args.get("path"), json.dumps(args, ensure_ascii=False))
            continue
        if name in ("Grep", "Glob", "grep", "find", "ls"):
            add("search", args.get("path", "."), json.dumps(args, ensure_ascii=False))
            continue
        if name not in ("Bash", "bash"):
            continue
        command = args.get("command", "")
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars="|;&")
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            continue
        python_sources = []
        # Recognize literal Python heredocs without executing or expanding them.
        for match in re.finditer(r"(?:^|\n)[^\n]*\bpython(?:3(?:\.10)?)?\s+-\s+<<[\"']?(\w+)[\"']?[^\n]*\n", command):
            end = re.search(r"^" + re.escape(match[1]) + r"\s*$", command[match.end():], re.MULTILINE)
            if end:
                python_sources.append(command[match.end():match.end() + end.start()])
        for i, token in enumerate(tokens):
            basename = Path(token).name
            if basename in {"cat", "sed", "head", "tail", "rg", "grep"}:
                kind = "search" if basename in {"rg", "grep"} else "read"
                for target in tokens[i + 1:]:
                    if target in {";", "|", "&&", "||"}:
                        break
                    if target.startswith(("tools/", "robot_config/", "skills/", "perception/", "tool_call_history/", "/")) or target in {"README.md", "AGENTS.md", "CLAUDE.md"}:
                        add(kind, target, command)
            if basename not in {"python", "python3", "python3.10"} or i + 1 == len(tokens):
                continue
            target = tokens[i + 1]
            if target.endswith(".py"):
                add("schema" if "--schema" in tokens[i + 2:] else "execute", target, command)
            if target != "-c" or i + 2 >= len(tokens):
                continue
            python_sources.append(tokens[i + 2])
        for code in python_sources:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            constants = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target_node in node.targets:
                        if isinstance(target_node, ast.Name):
                            constants[target_node.id] = node.value.value
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("tools.", "robot_config.")):
                    add("import", node.module.replace(".", "/") + ".py", command)
                elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("util."):
                    target = "tools/" + node.module.replace(".", "/") + ".py"
                    if (workspace / target).is_file():
                        add("import", target, command)
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = node.func
                if not ((isinstance(func, ast.Name) and func.id == "open") or
                        (isinstance(func, ast.Attribute) and func.attr == "open")):
                    continue
                value = node.args[0]
                target = value.value if isinstance(value, ast.Constant) else constants.get(value.id) if isinstance(value, ast.Name) else None
                if not isinstance(target, str):
                    continue
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                if isinstance(mode, str) and any(flag in mode for flag in "wax"):
                    continue
                add("process_image" if isinstance(func, ast.Attribute) else "program_read", target, command)
    return list({(x["kind"], x["path"], x["evidence"]): x for x in found}.values())


class NativeLog:
    def __init__(self, case_id, workspace, paths):
        self.id, self.workspace, self.paths = case_id, workspace, paths
        self.rows, self.positions, self.reads, self.receipts, self.tools = [], {}, [], {}, {}
        self.lock = threading.RLock()

    def refresh(self):
        with self.lock:
            for path in self.paths:
                stat = path.stat()
                previous = self.positions.get(path)
                if previous and (previous[1] != stat.st_ino or previous[0] > stat.st_size):
                    self.rows, self.positions, self.reads, self.receipts, self.tools = [], {}, [], {}, {}
                    return self.refresh()
                offset = previous[0] if previous else 0
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b"\n"):
                            break
                        try:
                            event = json.loads(line)
                        except ValueError:
                            event = {"type": "invalid-json-line", "text": line.decode(errors="replace")}
                        if not isinstance(event, dict):
                            event = {"type": "non-object-json", "value": event}
                        event = pi_event(event)
                        number = len(self.rows) + 1
                        row = {"number": number, "offset": start, "path": path,
                               "event": self.sanitize(event, number)}
                        self.rows.append(row)
                        for access in accesses(event, self.workspace):
                            self.reads.append({**access, "line": number, "timestamp": event.get("timestamp")})
                        for block in blocks(event):
                            if block.get("type") == "tool_use":
                                self.tools[block.get("id")] = number
                            if block.get("type") != "tool_result":
                                continue
                            for text in texts(block.get("content")):
                                try:
                                    receipt = json.loads(text)
                                except (ValueError, TypeError):
                                    continue
                                if not isinstance(receipt, dict) or not isinstance(receipt.get("record"), str):
                                    continue
                                label = path_label(receipt["record"], self.workspace)
                                match = re.fullmatch(r"tool_call_history/([^/]+)/result.json", label or "")
                                if match:
                                    self.receipts[match[1]] = {"result_line": number, "tool_line": self.tools.get(block.get("tool_use_id"))}
                        offset = stream.tell()
                self.positions[path] = (offset, stat.st_ino)
        return self

    def sanitize(self, value, number, pointer=()):
        if isinstance(value, list):
            return [self.sanitize(x, number, (*pointer, i)) for i, x in enumerate(value)]
        if not isinstance(value, dict):
            return value
        if isinstance(value.get("data"), str) and (value.get("media_type") or value.get("mimeType") or "").startswith("image/"):
            raw = base64.b64decode(value["data"], validate=True)
            selector = base64.urlsafe_b64encode(json.dumps([*pointer, "data"]).encode()).decode().rstrip("=")
            return {**{k: v for k, v in value.items() if k != "data"},
                    "image_url": f"/api/native-image?id={quote(self.id)}&line={number}&pointer={selector}",
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return {k: self.sanitize(v, number, (*pointer, k)) for k, v in value.items()}

    def raw(self, number):
        if number < 1 or number > len(self.rows):
            raise ValueError("Unknown message")
        row = self.rows[number - 1]
        with row["path"].open("rb") as stream:
            stream.seek(row["offset"])
            # Image selectors point into the same converted shape as the rows.
            return pi_event(json.loads(stream.readline()))

    def entry(self, number):
        row = self.rows[number - 1]
        return {"number": number, "source": str(row["path"].relative_to(self.workspace.parent)), "event": row["event"]}


def content_blocks(value):
    """Keep message content, without its transport/event envelope."""
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    if isinstance(value, list):
        return [block for item in value for block in content_blocks(item)]
    return [value] if isinstance(value, dict) else []


def attachment_content(attachment):
    """Only saved model-facing content; hook stdout is a duplicate audit log."""
    kind = attachment.get("type", "")
    if kind.startswith("hook_") and kind != "hook_additional_context":
        return []
    if kind == "compact_file_reference":
        return content_blocks(attachment.get("filename"))
    value = attachment.get("content", attachment.get("text"))
    if kind == "file" and isinstance(value, dict) and "file" in value:
        value = value["file"].get("content")
    return content_blocks(value)


def robot_command_names(event, block, workspace):
    """Recognize executed workspace entry points, not mentions or schema reads."""
    tool = block.get("name", "")
    if tool.startswith("mcp__"):
        name = tool.rsplit("__", 1)[-1]
        return [name] if (workspace / "tools" / (name + ".py")).is_file() else []
    if tool not in ("Bash", "bash"):
        return []
    command = block.get("input", {}).get("command", "")
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|;&")
        lexer.whitespace_split = True
        lexer.commenters = ""
        segments, segment = [], []
        for token in lexer:
            if token in {";", "|", "&&", "||", "&"}:
                segments.append(segment)
                segment = []
            else:
                segment.append(token)
        segments.append(segment)
    except ValueError:
        return []
    names = []
    for tokens in segments:
        if any(x in tokens for x in ("--schema", "--help", "-h")):
            continue
        for i, token in enumerate(tokens[:-1]):
            if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(token).name):
                continue
            label = path_label(tokens[i + 1], workspace, event.get("cwd")) or ""
            if re.fullmatch(r"tools/[a-z][a-z0-9_]*\.py", label) and (workspace / label).is_file():
                names.append(Path(label).stem)
            elif label == "robot_config/control.py":
                names.append("joint API")
    return list(dict.fromkeys(names))


def conversation(log):
    """Join native tool_use/result IDs and response IDs, never robot log order.

    Claude streams thinking, text and tool_use as separate JSONL entries. A
    response may issue multiple tools, even with hook/results interleaved. Only
    content already emitted by that same response is its preceding reasoning.
    """
    with log.lock:
        rows = list(log.rows)
    entries, groups, tools, seen = [], {}, {}, set()
    ordinal, fallback_group, last_assistant = 0, None, False

    def add_context(row, label, value, kind="context"):
        content = content_blocks(value)
        if not content:
            return
        chunk = {"line": row["number"], "label": label, "blocks": content}
        # Reminders and post-tool context stay with the return they follow, but
        # remain explicitly distinct from the tool's own output.
        if kind == "context" and entries and entries[-1]["kind"] == "tool" and entries[-1]["results"]:
            entries[-1]["after"].append(chunk)
        elif kind == "context" and entries and entries[-1]["kind"] == "context":
            entries[-1]["chunks"].append(chunk)
        else:
            entries.append({"id": f"context-{row['number']}", "kind": kind,
                            "line": row["number"], "chunks": [chunk]})

    for row in rows:
        event, number, source = row["event"], row["number"], str(row["path"])
        identity = (source, event.get("uuid"))
        if identity[1] and identity in seen:
            continue
        seen.add(identity)
        kind = event.get("type")
        if kind == "assistant":
            message_id = event.get("message", {}).get("id")
            if not message_id:
                if not last_assistant or fallback_group is None:
                    fallback_group = f"unidentified-{number}"
                message_id = fallback_group
            last_assistant = True
            group = groups.setdefault((source, message_id), {"prefix": [], "pending": None, "last_call": None})
            for index, block in enumerate(blocks(event)):
                if block.get("type") != "tool_use":
                    item = {"line": number, "block": block}
                    group["prefix"].append(item)
                    if group["pending"] is None:
                        group["pending"] = {"id": f"reply-{number}-{index}", "kind": "assistant",
                                            "line": number, "before": []}
                        entries.append(group["pending"])
                    group["pending"]["before"].append(item)
                    continue
                ordinal += 1
                entry = group["pending"]
                if entry is None:
                    entry = {}
                    entries.append(entry)
                previous = group["last_call"]
                entry.update({"id": f"tool-{number}-{index}", "kind": "tool", "number": ordinal,
                              "line": number, "before": list(group["prefix"]), "call": block,
                              "shared_before": previous["id"] if previous and previous["before"] == group["prefix"] else None,
                              "results": [], "after": [], "robot_records": [],
                              "robot_tools": robot_command_names(event, block, log.workspace)})
                group["pending"], group["last_call"] = None, entry
                tools[(source, block.get("id"))] = entry
        elif kind == "user":
            other = []
            for index, block in enumerate(blocks(event)):
                if block.get("type") != "tool_result":
                    other.append(block)
                    continue
                last_assistant = False
                entry = tools.get((source, block.get("tool_use_id")))
                if entry is None:
                    entry = {"id": f"orphan-{number}-{index}", "kind": "orphan_result", "line": number, "results": []}
                    entries.append(entry)
                entry["results"].append({"line": number, "block": block})
                if entry["kind"] != "tool":
                    continue
                for value in texts(block.get("content")):
                    try:
                        receipt = json.loads(value)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(receipt, dict):
                        continue
                    label = path_label(receipt.get("record"), log.workspace)
                    match = re.fullmatch(r"tool_call_history/([0-9]{6}_[a-z0-9_]+)/result.json", label or "")
                    if match:
                        entry["robot_records"].append(match[1])
                        name = receipt.get("tool_name") or match[1][7:]
                        if name not in entry["robot_tools"]:
                            entry["robot_tools"].append(name)
            if other:
                last_assistant, fallback_group = False, None
                add_context(row, "压缩摘要" if event.get("isCompactSummary") else "User", other, "summary" if event.get("isCompactSummary") else "user")
        elif kind == "attachment":
            attachment = event.get("attachment", {})
            label = attachment.get("type", "context")
            if label == "file":
                label = attachment.get("displayPath") or attachment.get("filename") or label
            add_context(row, label, attachment_content(attachment))
        elif kind == "system" and event.get("subtype") == "compact_boundary":
            last_assistant, fallback_group = False, None
            entries.append({"id": f"boundary-{number}", "kind": "boundary", "line": number})

    for entry in entries:
        if entry["kind"] == "tool":
            entry["action"] = bool(entry["robot_tools"])
            entry["status"] = "pending" if not entry["results"] else "error" if any(x["block"].get("is_error") for x in entry["results"]) else "returned"
        # Allows the browser to append/update only changed rows on live refresh,
        # preserving text selection, expanded blocks and in-cell scroll positions.
        entry["revision"] = hashlib.sha256(json.dumps(entry, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    return {"entries": entries, "tool_count": ordinal,
            "action_count": sum(bool(e.get("action")) for e in entries),
            "native_total": len(rows)}


def session_identity(case, summary, info):
    """Use this instance's original start, never a later resume or log mtime."""
    task = summary.get("task_id")
    if task is None:
        match = re.search(r"task(\d+)", case.parent.name)
        task = int(match[1]) if match else None
    instance = info.get("instance_id", case.name[9:])
    started = read_json(case / "session-timing.json").get("started_at")
    if started is None:
        history = sorted((case / "workspace/tool_call_history").glob("*/input.json"))
        if history:
            started = read_json(history[0]).get("started_at")
    try:
        if isinstance(started, str):
            stamp = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=DISPLAY_TIMEZONE)
            started = stamp.timestamp()
        started = float(started)
        date = datetime.fromtimestamp(started, DISPLAY_TIMEZONE).strftime("%Y%m%d-%H%M%S")
    except (TypeError, ValueError, OverflowError):
        started, date = None, "unknown-start"
    task_name = f"task{int(task):02d}" if task is not None else "task-unknown"
    return {"task": task, "started_at": started,
            "display_name": f"{task_name}_instance{instance}_{date}"}


class Viewer:
    def __init__(self, runs, run_glob="*"):
        self.runs = runs.resolve()
        self.run_glob = run_glob
        self.logs = OrderedDict()
        self.lock = threading.RLock()

    def case(self, key):
        if not NAME.fullmatch(key) or ".." in key:
            raise ValueError("Invalid session")
        case = (self.runs / key).resolve()
        if not case.is_relative_to(self.runs) or not (case / "workspace").is_dir():
            raise ValueError("Unknown session")
        return case

    def native(self, key):
        case = self.case(key)
        paths = native_paths(case)
        with self.lock:
            log = self.logs.get(key)
            if log is None or log.paths != paths:
                log = self.logs[key] = NativeLog(key, case / "workspace", paths)
            self.logs.move_to_end(key)
            while len(self.logs) > 3:
                self.logs.popitem(last=False)
        return log.refresh()

    def catalog(self):
        rows = []
        for case in self.runs.glob(f"{self.run_glob}/instance_*"):
            if not (case / "workspace/tool_call_history").is_dir():
                continue
            key = str(case.relative_to(self.runs))
            if not NAME.fullmatch(key):
                continue
            summary = read_json(case.parent / "summary.json")
            info = read_json(case / "case.json")
            result = next((x for x in summary.get("cases", []) if x.get("instance_id") == info.get("instance_id")), None)
            active = read_json(case.parent / "active_session.json")
            # Resident tests keep the binding with the evaluator, and use
            # phase/instances rather than the older task01 status/n_expected.
            resident = Path(summary.get("resident", "")).resolve()
            if summary.get("resident") and resident.is_relative_to(self.runs):
                active = read_json(resident / "active_session.json")
            run_status = summary.get("status", summary.get("phase"))
            expected = summary.get("n_expected", len(summary.get("instances", [])) or None)
            state = read_json(case / "workspace/.runtime/state.json")
            running = run_status == "running" and active.get("workspace") == str(case / "workspace")
            native = native_paths(case)
            rows.append({"id": key, "run": case.parent.name, "instance": info.get("instance_id", case.name[9:]),
                         **session_identity(case, summary, info),
                         "session_id": info.get("session_id"), "status": "completed" if result else "running" if running else "incomplete",
                         "q": result.get("q") if result else None, "steps": result.get("steps") if result else None,
                         "calls": state.get("counter", 0), "active_skill": state.get("active_skill"),
                         "mtime": max((p.stat().st_mtime for p in native), default=case.stat().st_mtime),
                         "native_sessions": [p.stem for p in native], "run_status": run_status,
                         "run_finished": summary.get("n_finished", 0), "run_expected": expected,
                         "mean_q": summary.get("mean_q"), "diagnostic": bool("smoke" in case.parent.name or "relocation" in case.parent.name)})
        return sorted(rows, key=lambda row: row["mtime"], reverse=True)

    def file_url(self, key, path):
        return f"/file?id={quote(key)}&path={quote(path)}"

    def file(self, key, value):
        case = self.case(key)
        # Deliberately not a general workspace/environment file server.
        if value in {"prompt.txt", "prompt.source.txt", "case.json", "comparison.json", "agent.stderr.log"}:
            pass
        elif not value.startswith(("workspace/tool_call_history/", "trajectory/calls/")):
            raise ValueError("File is outside published trajectory records")
        parts = Path(value).parts
        if any(p.startswith(".") for p in parts) or ".." in parts:
            raise ValueError("Invalid file")
        path = (case / value).resolve()
        if not path.is_relative_to(case) or not path.is_file():
            raise ValueError("Missing file")
        return path

    def calls(self, key, log):
        case = self.case(key)
        rows = []
        for folder in sorted((case / "workspace/tool_call_history").iterdir()):
            if not CALL.fullmatch(folder.name) or not folder.is_dir():
                continue
            inp = read_json(folder / "input.json")
            result = read_json(folder / "result.json")
            bulk = case / "trajectory/calls" / folder.name
            rows.append({"id": folder.name, "number": int(folder.name[:6]), "tool": inp.get("tool", folder.name[7:]),
                         "input": inp.get("arguments", {}), "timestamp": inp.get("started_at"),
                         "ok": result.get("ok"), "duration": result.get("duration_s"),
                         "commands": result.get("joint_commands_sent"), "skill": result.get("task_skill_state", {}).get("active_skill"),
                         "error": result.get("error") or result.get("response", {}).get("error") or read_json(bulk / "output.json" if (bulk / "output.json").is_file() else folder / "output.json").get("error"),
                         "native": log.receipts.get(folder.name),
                         "image_count": len(result.get("images", [])), "has_click": bool(list(folder.glob("click_*_crosshair.png")))})
        return rows

    def detail(self, key, call):
        if not CALL.fullmatch(call):
            raise ValueError("Invalid call")
        case, log = self.case(key), self.native(key)
        folder = case / "workspace/tool_call_history" / call
        if not folder.is_dir():
            raise ValueError("Missing call")
        bulk = case / "trajectory/calls" / call
        images = []
        # Old layout keeps observation/ inside the workspace record; new layout keeps bulk files in trajectory/.
        for pattern, label in [("click_*_crosshair.png", "选点十字 · 调用前原图"), ("returned_image_*", "实际返给模型的图片"),
                               ("observation/robot_three_views.png", "本体感知 · 测得姿态三视图"),
                               ("observation/head_rgb.png", "调用后头部 RGB 原图"), ("observation/head_depth.png", "头部深度预览")]:
            for path in sorted(folder.glob(pattern)):
                images.append({"label": label, "path": str(path.relative_to(case)), "url": self.file_url(key, str(path.relative_to(case)))})
        for path in sorted(bulk.glob("observation/*.png")):
            images.append({"label": "调用后观测归档 · " + path.stem, "path": str(path.relative_to(case)),
                           "url": self.file_url(key, str(path.relative_to(case)))})
        links = [{"name": str(p.relative_to(root)), "url": self.file_url(key, str(p.relative_to(case)))}
                 for root in (folder, bulk) if root.is_dir()
                 for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".json", ".jsonl", ".npy", ".npz"}]
        def bulk_json(name, default=None):
            return read_json(bulk / name, default) if (bulk / name).is_file() else read_json(folder / name, default)
        match = log.receipts.get(call, {})
        selected = []
        if match.get("tool_line"):
            number = match["tool_line"]
            start = number
            while start > 1 and log.rows[start - 2]["event"].get("type") == "assistant":
                start -= 1
            selected.extend(range(start, number + 1))
        if match.get("result_line"):
            selected.append(match["result_line"])
        return {"id": call, "input": read_json(folder / "input.json"), "result": read_json(folder / "result.json"),
                "output": bulk_json("output.json"), "selected_points": bulk_json("selected_points.json", []),
                "images": images, "files": links, "native": [log.entry(n) for n in selected]}

    def session(self, key):
        log = self.native(key)
        case = self.case(key)
        calls = self.calls(key, log)
        accesses_by_file = {}
        for item in log.reads:
            group = accesses_by_file.setdefault((item["kind"], item["path"]), {"kind": item["kind"], "path": item["path"], "events": []})
            group["events"].append({k: v for k, v in item.items() if k not in {"kind", "path"}})
        skills = []
        for call in calls:
            if (call["tool"] == "activate_skill" and call["ok"] is not False
                    and not call["error"] and call["skill"] == call["input"].get("name")):
                skills.append({"name": call["input"].get("name"), "call": call["id"], "native": call["native"]})
        counts = Counter(item["kind"] for item in log.reads)
        summary = read_json(case.parent / "summary.json")
        return {"id": key, "calls": calls, "accesses": list(accesses_by_file.values()), "access_counts": dict(counts),
                "activated_skills": skills, "native_total": len(log.rows), "workspace": str(case / "workspace"),
                "prompt_url": self.file_url(key, "prompt.txt"), "case_info": read_json(case / "case.json"),
                "run_summary": summary, "delivery_audit": read_json(case / "native-delivery-audit.json"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "limits": "文件访问只从原生工具调用取证；执行 Python、程序打开图片、自动注入 skill 都不等同于模型阅读算法。未记录的文件访问和内置系统提示词无法还原。"}


def handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, mime="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                parsed = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                key = query.get("id", "")
                route = parsed.path
                if route in {"/", "/app.js", "/style.css"}:
                    path = WEB / ("index.html" if route == "/" else route[1:])
                    self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] + "; charset=utf-8")
                elif route == "/api/sessions":
                    self.send(viewer.catalog())
                elif route == "/api/session":
                    self.send(viewer.session(key))
                elif route == "/api/call":
                    self.send(viewer.detail(key, query["call"]))
                elif route == "/api/conversation":
                    self.send(conversation(viewer.native(key)))
                elif route == "/api/native":
                    log = viewer.native(key)
                    page = max(0, int(query.get("page", "0")))
                    search, kind = query.get("q", "").lower(), query.get("kind", "")
                    rows = [r for r in log.rows if (not kind or r["event"].get("type") == kind) and
                            (not search or search in json.dumps(r["event"], ensure_ascii=False).lower())]
                    self.send({"total": len(rows), "page": page, "records": [log.entry(r["number"]) for r in rows[page*30:(page+1)*30]]})
                elif route == "/api/record":
                    log = viewer.native(key)
                    number = int(query["line"])
                    if number < 1 or number > len(log.rows):
                        raise ValueError("Missing message")
                    self.send(log.entry(number))
                elif route == "/api/native-image":
                    event = viewer.native(key).raw(int(query["line"]))
                    encoded = query["pointer"]
                    pointer = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
                    if not isinstance(pointer, list) or len(pointer) > 32 or not pointer or pointer[-1] != "data":
                        raise ValueError("Invalid image selector")
                    source = event
                    for item in pointer[:-1]:
                        source = source[item]
                    mime = source.get("media_type") or source.get("mimeType") or ""
                    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                        raise ValueError("Not an image")
                    self.send(base64.b64decode(source["data"], validate=True), mime)
                elif route == "/file":
                    path = viewer.file(key, query["path"])
                    self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                else:
                    self.send_error(404)
            except (KeyError, IndexError, ValueError, OSError, TypeError) as exc:
                self.send_error(404, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--port", type=int, default=18792)
    parser.add_argument("--run-glob", default="*", help="Run directory names to list, e.g. test-v3-task*")
    args = parser.parse_args()
    if "/" in args.run_glob or "\\" in args.run_glob or ".." in args.run_glob:
        parser.error("--run-glob must match directory names only")
    print(f"Workspace trajectories: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(Viewer(args.runs, args.run_glob))).serve_forever()


if __name__ == "__main__":
    main()
