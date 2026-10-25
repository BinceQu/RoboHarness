"""Run unmodified Claude Code directly against the configured Messages API."""

import argparse
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "Qwen3.8-Flash-Next-MLX-oQ6-MTP"),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "EMBODIED_ANTHROPIC_BASE_URL", "http://127.0.0.1:31000"
        ),
    )
    parser.add_argument("claude_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not (root / ".session.json").is_file():
        raise SystemExit(
            "Create a fresh session with session.py before launching an agent"
        )
    origin = urlsplit(args.base_url)
    if (
        origin.scheme not in ("http", "https")
        or not origin.hostname
        or origin.path.rstrip("/")
        or origin.query
        or origin.fragment
    ):
        parser.error("Use the direct Messages API origin without /v1")
    binary = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not binary:
        raise SystemExit("Claude Code executable is not installed")
    env = dict(os.environ)
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        env.pop(key, None)
    env.update(
        ANTHROPIC_BASE_URL=args.base_url.rstrip("/"),
        ANTHROPIC_AUTH_TOKEN=env.get("ANTHROPIC_AUTH_TOKEN", "local"),
        ANTHROPIC_API_KEY=env.get("ANTHROPIC_API_KEY", "local-qwen"),
        ANTHROPIC_MODEL=args.model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-sonnet-4-6",
        ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6",
        ANTHROPIC_DEFAULT_OPUS_MODEL="claude-sonnet-4-6",
        CLAUDE_CODE_EFFORT_LEVEL="xhigh",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_TELEMETRY="1",
        DISABLE_ERROR_REPORTING="1",
        ENABLE_TOOL_SEARCH="false",
        PYTHONDONTWRITEBYTECODE="1",
        PATH=str(root / ".venv/bin") + os.pathsep + env.get("PATH", ""),
        CLAUDE_CONFIG_DIR=str(root / ".runtime/claude-home"),
    )
    extra = args.claude_arguments
    if extra[:1] == ["--"]:
        extra = extra[1:]
    marker = root / ".runtime/agent-started"
    resuming = any(item in {"--resume", "--continue", "-r", "-c"} for item in extra)
    if marker.exists() and not resuming:
        parser.error(
            "Create a fresh workspace for a new session; use --resume for this exact session"
        )
    marker.touch(exist_ok=True)
    os.chdir(root)
    print(
        f"Starting ARX_X5 Claude Code directly via {args.base_url}",
        file=sys.stderr,
        flush=True,
    )
    os.execve(
        binary,
        [
            binary,
            "--model",
            args.model,
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Bash,Read,Glob,Grep,Skill,Write,Edit",
            "--allowedTools",
            "Bash,Read,Glob,Grep,Skill,Write,Edit",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "project",
            *extra,
        ],
        env,
    )


if __name__ == "__main__":
    main()
