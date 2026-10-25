"""Launch unmodified Claude Code in one initialized robot workspace."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

from .settings import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--url", default=os.environ.get("EMBODIED_ANTHROPIC_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen3.8-Flash-Next-MLX-oQ6-MTP"))
    parser.add_argument("claude_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    origin = urlsplit(args.url or "")
    if origin.scheme not in ("http", "https") or not origin.hostname or origin.path.rstrip("/") or origin.query or origin.fragment:
        parser.error("Set EMBODIED_ANTHROPIC_BASE_URL to the direct Messages origin, without /v1")
    os.environ["ROBOT_WORKSPACE"] = str(args.workspace.resolve())
    config = configure()
    binary = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not binary:
        parser.error("Claude Code is not installed")
    marker = Path(config["state_dir"]) / "agent-started"
    tail = args.claude_args[1:] if args.claude_args[:1] == ["--"] else args.claude_args
    resuming = any(item in {"--resume", "--continue", "-r", "-c"} for item in tail)
    if marker.exists() and not resuming:
        parser.error("Initialize a fresh workspace for every new session; --resume continues this exact session")
    if not marker.exists():
        marker.touch(exist_ok=False)
    env = dict(os.environ)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    env.update(ANTHROPIC_BASE_URL=args.url.rstrip("/"), ANTHROPIC_MODEL=args.model,
        ANTHROPIC_API_KEY=env.get("ANTHROPIC_API_KEY", "local-qwen"),
        ANTHROPIC_AUTH_TOKEN=env.get("ANTHROPIC_AUTH_TOKEN", "local-qwen"),
        ANTHROPIC_DEFAULT_OPUS_MODEL="claude-sonnet-4-6", ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6",
        ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-sonnet-4-6", CLAUDE_CODE_EFFORT_LEVEL="xhigh",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1",
        ENABLE_TOOL_SEARCH="false", CLAUDE_CONFIG_DIR=str(Path(config["state_dir"]) / "claude-home"),
        PYTHONDONTWRITEBYTECODE="1", PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"])
    command = [binary, "--model", args.model, "--permission-mode", "dontAsk",
        "--tools", "Bash,Read,Glob,Grep,Skill", "--allowedTools", "Bash", "Read", "Glob", "Grep", "Skill",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "project", *tail]
    print(f"Starting workspace Claude Code at {config['workspace']} via {args.url} ({args.model})", file=sys.stderr, flush=True)
    os.chdir(config["workspace"])
    os.execvpe(binary, command, env)


if __name__ == "__main__":
    main()
