"""Fail-closed Claude Code launcher for the embodied plugin."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .qwen_bridge import DEFAULT_MODEL, DEFAULT_UPSTREAM, normalize_upstream_url
from . import skill_state
from .config import OwnedPortRedirectHandler, validate_owned_origin


MAX_PROMPT_IMAGES = 8
MAX_PROMPT_IMAGE_BYTES = 8 * 1024 * 1024
EMBODIED_TOOL_GLOBS = (
    "mcp__behavior-v2__*",
    "mcp__plugin_embodied-claude-code_behavior-v2__*",
)
# Claude Code 2.1 scans the joined allow-list for model-id substrings.
# "behavior-v2" contains "gemini", so that glob rejects a Gemini --model
# before any request.  behavior-robot is the same MCP process under a name
# that does not collide.  Qwen keeps the original globs.
GEMINI_TOOL_GLOBS = (
    "mcp__behavior-robot__*",
    "mcp__plugin_embodied-claude-code_behavior-robot__*",
)


def _is_gemini_origin(model: str, upstream: str) -> bool:
    """Gemini cannot emit a tool_use whose name contains ``mcp__``.

    Qwen 27B and Flash speak /v1/messages with the original tool names.
    They must not enter the Gemini name-shortening proxy.
    """
    return "gemini" in f"{model} {upstream}".lower()


def _needs_xhigh_effort(model: str, upstream: str) -> bool:
    """27B vLLM accepts xhigh, medium, and low. Claude's default is high."""
    blob = f"{model} {upstream}"
    return "27B-FP8" in blob or ":30000" in blob


BLOCKED_CLAUDE_OPTIONS = {
    "--allowedTools",
    "--allowed-tools",
    "--bare",
    "--dangerously-skip-permissions",
    "--disable-slash-commands",
    "--disallowedTools",
    "--disallowed-tools",
    "--mcp-config",
    "--model",
    "--permission-mode",
    "--plugin-dir",
    "--setting-sources",
    "--settings",
    "--strict-mcp-config",
    "--tools",
}


class UsageError(RuntimeError):
    pass


def plugin_root() -> Path:
    # The launcher runs from a local copy under /tmp so its import does not
    # sit on NFS.  Two parents of that copy is /tmp, and Claude then looks
    # for profiles/claude-settings.json there.  The real plugin root is the
    # one the wrapper recorded.
    recorded = os.environ.get("EMBODIED_PLUGIN_ROOT", "").strip()
    if recorded:
        root = Path(recorded).expanduser()
        if (root / "profiles" / "claude-settings.json").is_file():
            return root.resolve()
    return Path(__file__).resolve().parents[2]


def _usage() -> str:
    return """Usage: embodied-claude-code [WRAPPER_OPTIONS] [--] [CLAUDE_ARGS...]

Start an isolated Claude Code session with the embodied plugin, bind its
BEHAVIOR MCP server to the selected local interface, and route Anthropic
Messages requests to an OpenAI-compatible Qwen endpoint.

Wrapper options:
  --port PORT             BEHAVIOR API port (default: 15060)
  --prompt-image FILE     Attach a first-turn image; repeat up to eight times
  --qwen-url URL          OpenAI /v1 or /v1/chat/completions endpoint
  --qwen-model MODEL      Upstream model name (default: Qwen3.8-27B)
  --claude-bin FILE       Claude Code executable
  -h, --help              Show this help

Compatibility:
  `exec PROMPT` and `exec -` are translated to Claude Code print mode.

Examples:
  embodied-claude-code --port 15063
  embodied-claude-code --port 15063 -- exec - < prompt.txt
  embodied-claude-code --port 15063 -- --print "Inspect the scene."
"""


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    raw_port = os.environ.get("BEHAVIOR_PORT", "15060")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise UsageError(f"invalid BEHAVIOR_PORT: {raw_port}") from exc
    images: list[str] = []
    qwen_url = (
        os.environ.get("QWEN_CHAT_COMPLETIONS_URL")
        or os.environ.get("QWEN_OPENAI_URL")
        or os.environ.get("QWEN_OPENAI_BASE_URL")
        or DEFAULT_UPSTREAM
    )
    qwen_model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
    claude_bin = os.environ.get("CLAUDE_BIN", "")
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--":
            index += 1
            break
        if value in {"-h", "--help"}:
            print(_usage(), end="")
            raise SystemExit(0)
        if value == "--port":
            index += 1
            if index >= len(argv):
                raise UsageError("--port requires a value")
            try:
                port = int(argv[index])
            except ValueError as exc:
                raise UsageError(f"invalid port: {argv[index]}") from exc
        elif value.startswith("--port="):
            try:
                port = int(value.split("=", 1)[1])
            except ValueError as exc:
                raise UsageError(f"invalid port: {value.split('=', 1)[1]}") from exc
        elif value == "--prompt-image":
            index += 1
            if index >= len(argv):
                raise UsageError("--prompt-image requires a file path")
            images.append(argv[index])
        elif value.startswith("--prompt-image="):
            images.append(value.split("=", 1)[1])
        elif value == "--qwen-url":
            index += 1
            if index >= len(argv):
                raise UsageError("--qwen-url requires a value")
            qwen_url = argv[index]
        elif value.startswith("--qwen-url="):
            qwen_url = value.split("=", 1)[1]
        elif value == "--qwen-model":
            index += 1
            if index >= len(argv):
                raise UsageError("--qwen-model requires a value")
            qwen_model = argv[index]
        elif value.startswith("--qwen-model="):
            qwen_model = value.split("=", 1)[1]
        elif value == "--claude-bin":
            index += 1
            if index >= len(argv):
                raise UsageError("--claude-bin requires a file path")
            claude_bin = argv[index]
        elif value.startswith("--claude-bin="):
            claude_bin = value.split("=", 1)[1]
        else:
            break
        index += 1
    if not 1 <= port <= 65535:
        raise UsageError(f"port must be between 1 and 65535: {port}")
    namespace = argparse.Namespace(
        port=port,
        images=images,
        qwen_url=normalize_upstream_url(qwen_url),
        qwen_model=str(qwen_model).strip() or DEFAULT_MODEL,
        claude_bin=claude_bin,
    )
    return namespace, argv[index:]


def _validate_session_id(value: str) -> None:
    if not value:
        return
    if len(value) > 64:
        raise UsageError(
            f"BEHAVIOR_SESSION_ID is {len(value)} chars; official max is 64"
        )
    if not value.isascii() or not value[0].isalnum() or any(
        not (char.isalnum() or char in "._-") for char in value
    ):
        raise UsageError(
            "BEHAVIOR_SESSION_ID must start with a letter or digit and contain "
            "only letters, digits, '.', '_' or '-'"
        )


def _image_kind(path: Path) -> str:
    data = path.read_bytes()[:16]
    suffix = path.suffix.lower()
    signatures = {
        ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": data.startswith(b"\xff\xd8\xff"),
        ".jpeg": data.startswith(b"\xff\xd8\xff"),
        ".gif": data.startswith((b"GIF87a", b"GIF89a")),
        ".webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }
    if suffix not in signatures or not signatures[suffix]:
        raise UsageError(
            f"prompt image is not a valid PNG/JPEG/GIF/WEBP file: {path}"
        )
    return suffix


def _validate_images(cli_images: list[str]) -> list[Path]:
    values = list(cli_images)
    extra = os.environ.get("BEHAVIOR_PROMPT_IMAGES", "")
    if extra:
        values.extend(part for part in extra.split(":") if part)
    result: list[Path] = []
    for raw in values:
        if not raw or raw == "-" or "\n" in raw:
            raise UsageError("prompt image path is invalid")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise UsageError(f"prompt image not found: {path}")
        if path.stat().st_size > MAX_PROMPT_IMAGE_BYTES:
            raise UsageError(f"prompt image too large (>8MiB): {path}")
        _image_kind(path)
        if path not in result:
            result.append(path)
    if len(result) > MAX_PROMPT_IMAGES:
        raise UsageError(f"too many prompt images (max {MAX_PROMPT_IMAGES})")
    return result


def _no_proxy_opener():
    return build_opener(ProxyHandler({}), OwnedPortRedirectHandler())


def _get_json(url: str) -> Any:
    request = Request(url, method="GET")
    last: Exception | None = None
    # 官方评测口 busy 时 /api/state、/api/v2/tools 一次 30s 仍可能超时。
    for attempt in range(6):
        try:
            with _no_proxy_opener().open(request, timeout=30.0) as response:
                if not 200 <= int(response.status) < 300:
                    raise RuntimeError(f"HTTP {response.status}")
                return json.loads(response.read())
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            last = exc
            if attempt < 5:
                time.sleep(2)
    raise RuntimeError(f"BEHAVIOR endpoint is not healthy: {url}: {last}") from last


def _post_monitor(
    base_url: str,
    prompt: str,
    session_id: str,
    images: list[Path],
    *,
    resume: bool = False,
) -> None:
    if not prompt.strip():
        return
    body: dict[str, Any] = {
        "prompt": prompt.strip(),
        "user_prompt": prompt.strip(),
        "loaded_skills": ["behavior-v2-baseline"],
        "new_attempt": not resume,
    }
    if resume:
        state = skill_state.read_state(session_id)
        if state and state.active_task_skill:
            body["loaded_skills"].append(state.active_task_skill)
    if session_id:
        body["session_id"] = session_id
    if images:
        body["prompt_images"] = [str(path) for path in images]
    request = Request(
        base_url + "/api/agent_monitor/prompt",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _no_proxy_opener().open(request, timeout=5.0) as response:
            if not 200 <= int(response.status) < 300:
                raise RuntimeError(f"HTTP {response.status}")
    except (URLError, TimeoutError, OSError, RuntimeError) as exc:
        print(
            f"embodied-claude-code: prompt publish failed: {exc}",
            file=sys.stderr,
        )


def _stamp_runtime(
    base_url: str,
    session_id: str,
    prompt: str,
    images: list[Path],
    *,
    resume: bool = False,
) -> None:
    if session_id and not resume:
        skill_state.save_active(None, session_id)
    folders = [Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "embodied-claude-code"]
    fallback = Path("/tmp/embodied-claude-code")
    if fallback not in folders and not os.environ.get("BEHAVIOR_EVAL_OWNER_PORT"):
        folders.append(fallback)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "behavior_base_url").write_text(base_url + "\n", encoding="utf-8")
        if prompt.strip():
            (folder / "user_prompt").write_text(prompt, encoding="utf-8")
            if session_id:
                (folder / f"user_prompt.{session_id}").write_text(
                    prompt, encoding="utf-8"
                )
        if images:
            value = "".join(str(path) + "\n" for path in images)
            (folder / "prompt_images").write_text(value, encoding="utf-8")
            if session_id:
                (folder / f"prompt_images.{session_id}").write_text(
                    value, encoding="utf-8"
                )
    if prompt.strip():
        primary = folders[0] / (
            f"user_prompt.{session_id}" if session_id else "user_prompt"
        )
        os.environ["BEHAVIOR_USER_PROMPT_FILE"] = str(primary)


def _claude_executable(explicit: str) -> str:
    candidates = [
        explicit,
        shutil.which("claude") or "",
        str(Path.home() / ".local" / "bin" / "claude"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError(
        "Claude Code is not installed; expected `claude` on PATH or "
        "~/.local/bin/claude"
    )


def _assert_safe_passthrough(args: list[str]) -> None:
    for value in args:
        option = value.split("=", 1)[0]
        if option in BLOCKED_CLAUDE_OPTIONS:
            raise UsageError(
                f"{option} is managed by embodied-claude-code and cannot be overridden"
            )


def _translate_exec(args: list[str]) -> tuple[list[str], str | None]:
    if not args or args[0] != "exec":
        return args, None
    translated = ["--print"]
    prompt_from_stdin: str | None = None
    for value in args[1:]:
        if value == "-":
            if prompt_from_stdin is None:
                prompt_from_stdin = sys.stdin.read()
            continue
        translated.append(value)
    return translated, prompt_from_stdin


def _prompt_from_args(args: list[str], stdin_prompt: str | None) -> str:
    if stdin_prompt is not None:
        return stdin_prompt
    print_mode = "--print" in args or "-p" in args
    if not print_mode:
        return ""
    skip_next = False
    candidates: list[str] = []
    options_with_value = {
        "--add-dir",
        "--agent",
        "--betas",
        "--debug-file",
        "--fallback-model",
        "--input-format",
        "--json-schema",
        "--max-budget-usd",
        "--output-format",
        "--resume",
        "--session-id",
        "--system-prompt",
    }
    for value in args:
        if skip_next:
            skip_next = False
            continue
        if value in options_with_value:
            skip_next = True
        elif value in {"--print", "-p", "--verbose"} or value.startswith("-"):
            continue
        else:
            candidates.append(value)
    return candidates[-1] if candidates else ""


def _wait_for_ready(process: subprocess.Popen[Any], ready_file: Path) -> int:
    # The bridge writes one port file after it binds.  Its interpreter
    # is on NFS, so the import can sit in rpc_wait well past 30s.  A short
    # deadline makes every lane exit and start another import.
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if ready_file.is_file():
            try:
                port = int(ready_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                port = 0
            if 1 <= port <= 65535:
                return port
        if process.poll() is not None:
            raise RuntimeError(
                f"Qwen protocol bridge exited early with status {process.returncode}"
            )
        time.sleep(0.05)
    raise RuntimeError("Qwen protocol bridge did not become ready")


def _append_no_proxy(env: dict[str, str], values: list[str]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = [item for item in env.get(key, "").split(",") if item]
        for value in values:
            if value not in current:
                current.append(value)
        env[key] = ",".join(current)


def launch(argv: list[str]) -> int:
    options, passthrough = _parse(argv)
    images = _validate_images(options.images)
    _assert_safe_passthrough(passthrough)
    claude_args, stdin_prompt = _translate_exec(passthrough)
    prompt = _prompt_from_args(claude_args, stdin_prompt)
    resume = any(value.split("=", 1)[0] in {"--resume", "-r", "--continue", "-c"}
                 for value in claude_args)
    session_id = os.environ.get("BEHAVIOR_SESSION_ID", "").strip()
    if resume and not session_id:
        raise UsageError("Resuming requires the original BEHAVIOR_SESSION_ID so hooks and MCP share its Skill state.")
    session_id = session_id or "claude-" + uuid.uuid4().hex[:16]
    _validate_session_id(session_id)

    root = plugin_root()
    base_url = f"http://127.0.0.1:{options.port}"
    validate_owned_origin(base_url, session_id)
    if os.environ.get("BEHAVIOR_EVAL_OWNER_PORT") and not os.environ.get("XDG_RUNTIME_DIR"):
        raise UsageError("Official agent requires its isolated XDG_RUNTIME_DIR.")
    # /api/state is the full browser snapshot.  On a lane whose tool call is
    # inside IK it does not return, and the six 30s retries hold this process
    # with no bridge child (15430, 15441).  The constant-size idle probe is
    # the readiness signal; tools are loaded again by MCP after Claude starts.
    try:
        probe = _get_json(base_url + "/__official__/idle_probe")
    except RuntimeError:
        probe = None
    if not isinstance(probe, dict) or probe.get("diagnostic_ok") is not True:
        _get_json(base_url + "/api/state")
        _get_json(base_url + "/api/v2/tools")
    _stamp_runtime(base_url, session_id, prompt, images, resume=resume)
    _post_monitor(base_url, prompt, session_id, images, resume=resume)

    claude_bin = _claude_executable(options.claude_bin)
    env = os.environ.copy()
    env.setdefault("CLAUDE_CONFIG_DIR", str(root / ".claude-home"))
    if not env.get("CLAUDE_PLUGIN_DATA", "").strip():
        env["CLAUDE_PLUGIN_DATA"] = str(
            (
                Path(env["CLAUDE_CONFIG_DIR"]).expanduser()
                / "plugins"
                / "data"
                / "embodied-claude-code-inline"
            ).resolve()
        )
    env.update(
        {
            "BEHAVIOR_BASE_URL": base_url,
            "BEHAVIOR_SESSION_ID": session_id,
            "EMBODIED_PLUGIN_ROOT": str(root),
            "CLAUDE_PLUGIN_ROOT": str(root),
            "QWEN_OPENAI_URL": options.qwen_url,
            "QWEN_MODEL": options.qwen_model,
            "QWEN_PROMPT_IMAGES_JSON": json.dumps([str(path) for path in images]),
            # A native Anthropic proxy (Gemini) authenticates with the
            # caller's token.  The OpenAI bridge path still uses local-qwen
            # because that bridge never forwards the key upstream.
            "ANTHROPIC_API_KEY": (
                os.environ.get("ANTHROPIC_API_KEY", "").strip() or "local-qwen"
            )
            if os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip()
            else "local-qwen",
            "ANTHROPIC_AUTH_TOKEN": (
                os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
                or os.environ.get("ANTHROPIC_API_KEY", "").strip()
                or "local-qwen"
            )
            if os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip()
            else "local-qwen",
            "ANTHROPIC_MODEL": options.qwen_model,
            # Claude Code 2.1 checks opus/sonnet/haiku aliases against its
            # own catalog before any request.  A Gemini id there is rejected
            # locally (api_error 400, duration_api_ms 0) even though --model
            # with that same id reaches the proxy.  Keep the aliases on a
            # catalog name.  The selected model stays --model / ANTHROPIC_MODEL.
            "ANTHROPIC_DEFAULT_OPUS_MODEL": (
                "claude-sonnet-4-6"
                if os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip()
                else options.qwen_model
            ),
            "ANTHROPIC_DEFAULT_SONNET_MODEL": (
                "claude-sonnet-4-6"
                if os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip()
                else options.qwen_model
            ),
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": (
                "claude-sonnet-4-6"
                if os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip()
                else options.qwen_model
            ),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
            # Qwen's OpenAI-compatible endpoint does not implement Anthropic
            # tool_reference blocks. Keep every BEHAVIOR tool schema in the
            # first request so a direct tool name is never deferred or lost.
            "ENABLE_TOOL_SEARCH": "false",
            # Port 15063 can return a multi-megabyte initial state. Prevent
            # noninteractive Claude from snapshotting tools before MCP is ready.
            "MCP_TIMEOUT": "120000",
            "MCP_CONNECT_TIMEOUT_MS": "120000",
            "MCP_CONNECTION_NONBLOCKING": "0",
            "PYTHONPATH": str(root / "src")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    upstream_host = urlparse(options.qwen_url).hostname or ""
    _append_no_proxy(env, ["127.0.0.1", "localhost", upstream_host])

    settings = root / "profiles" / "claude-settings.json"
    command = [
        claude_bin,
        *claude_args,
        "--plugin-dir",
        str(root),
        "--settings",
        str(settings),
        "--model",
        options.qwen_model,
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        *EMBODIED_TOOL_GLOBS,
        "--tools",
        "",
        "--disable-slash-commands",
    ]
    if os.environ.get("BEHAVIOR_EVAL_OWNER_PORT"):
        # Only the explicit current plugin settings; no user/project history.
        command.extend(["--setting-sources", ""])
    direct_anthropic = os.environ.get("EMBODIED_ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    gemini_origin = _is_gemini_origin(options.qwen_model, direct_anthropic)
    if direct_anthropic and gemini_origin:
        # Gemini must not also load the plugin MCP (that publishes the
        # original Qwen coordinate strings).  strict-mcp-config keeps a
        # single server under the name the skill text already uses.
        # The wrapper rewrites those coordinate strings for this process
        # only.  Qwen does not take this branch.
        config_dir = Path(env.get("CLAUDE_CONFIG_DIR") or "/tmp")
        config_dir.mkdir(parents=True, exist_ok=True)
        mcp_path = config_dir / "behavior-v2-mcp.json"
        plugin_root_dir = env.get("CLAUDE_PLUGIN_ROOT") or str(root)
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "behavior-v2": {
                            "command": sys.executable,
                            "args": [
                                "-m",
                                "embodied_claude_code.gemini_mcp_wrap",
                                str(Path(plugin_root_dir) / "scripts" / "embodied-claude-code-mcp"),
                            ],
                            "alwaysLoad": True,
                            "env": {"EMBODIED_PLUGIN_ROOT": plugin_root_dir},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        try:
            mcp_path.chmod(0o644)
        except OSError:
            pass
        command.extend(["--strict-mcp-config", "--mcp-config", str(mcp_path)])
    stdin_data = stdin_prompt
    # 上游已是 Anthropic Messages（如 31000）时，Claude 直连，不再经 OpenAI bridge。
    # ANTHROPIC_BASE_URL 不要带 /v1，Claude 会自己拼 /v1/messages。
    # Qwen 与官方评测同一条路：ANTHROPIC_BASE_URL 就是模型地址。
    # Gemini 才会再套一层本地代理，用来缩短 mcp__ 工具名或注入附图。
    if direct_anthropic:
        parsed_direct = urlparse(direct_anthropic)
        if parsed_direct.scheme not in {"http", "https"} or not parsed_direct.netloc:
            raise RuntimeError("EMBODIED_ANTHROPIC_BASE_URL must be an absolute http(s) URL")
        if parsed_direct.path.rstrip("/") in {"/v1", "/v1/messages", "/v1/chat/completions"}:
            raise RuntimeError(
                "EMBODIED_ANTHROPIC_BASE_URL must not include /v1; "
                "Claude Code appends /v1/messages itself"
            )
        _append_no_proxy(env, [parsed_direct.hostname or ""])
        if _needs_xhigh_effort(options.qwen_model, direct_anthropic):
            env["CLAUDE_CODE_EFFORT_LEVEL"] = "xhigh"
        if not gemini_origin and not images:
            env["ANTHROPIC_BASE_URL"] = direct_anthropic
            print(
                f"Starting Embodied Claude Code on {base_url} with "
                f"{options.qwen_model} via {direct_anthropic}",
                file=sys.stderr,
            )
            completed = subprocess.run(
                command,
                env=env,
                input=stdin_data,
                text=True,
                check=False,
            )
            return int(completed.returncode)
        proxy_module = (
            "embodied_claude_code.anthropic_prompt_inject"
            if images
            else "embodied_claude_code.gemini_tool_proxy"
        )
        with tempfile.TemporaryDirectory(prefix="embodied-gemini-proxy-") as temporary:
            ready_file = Path(temporary) / "port"
            injector = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    proxy_module,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--ready-file",
                    str(ready_file),
                    "--upstream",
                    direct_anthropic,
                ],
                env=env,
            )
            try:
                inject_port = _wait_for_ready(injector, ready_file)
                env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{inject_port}"
                print(
                    f"Starting Embodied Claude Code on {base_url} with "
                    f"{options.qwen_model} via local {proxy_module} -> "
                    f"{direct_anthropic}",
                    file=sys.stderr,
                )
                completed = subprocess.run(
                    command,
                    env=env,
                    input=stdin_data,
                    text=True,
                    check=False,
                )
                return int(completed.returncode)
            finally:
                injector.terminate()
                try:
                    injector.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    injector.kill()
                    injector.wait(timeout=5.0)

    # gpt-6-luna's gateway rejects /v1/messages.  The local bridge turns
    # Claude's Messages request into chat completions, which that gateway
    # accepts.  A native Messages URL still uses the branch above.
    # The ready file must be on local disk.  tempfile follows TMPDIR, and
    # a ready file on NFS is not visible inside the 30s wait.
    with tempfile.TemporaryDirectory(prefix="embodied-claude-bridge-", dir="/tmp") as temporary:
        ready_file = Path(temporary) / "port"
        bridge = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "embodied_claude_code.qwen_bridge",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--ready-file",
                str(ready_file),
                "--upstream",
                options.qwen_url,
                "--model",
                options.qwen_model,
            ],
            env=env,
        )
        try:
            bridge_port = _wait_for_ready(bridge, ready_file)
            env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{bridge_port}"
            print(
                f"Starting Embodied Claude Code on {base_url} with "
                f"{options.qwen_model}",
                file=sys.stderr,
            )
            completed = subprocess.run(
                command,
                env=env,
                input=stdin_data,
                text=True,
                check=False,
            )
            return int(completed.returncode)
        finally:
            bridge.terminate()
            try:
                bridge.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait(timeout=5.0)


def main(argv: list[str] | None = None) -> None:
    try:
        status = launch(list(sys.argv[1:] if argv is None else argv))
    except UsageError as exc:
        print(f"embodied-claude-code: {exc}", file=sys.stderr)
        print("Try `embodied-claude-code --help` for usage.", file=sys.stderr)
        raise SystemExit(2) from exc
    except (RuntimeError, ValueError) as exc:
        print(f"embodied-claude-code: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
