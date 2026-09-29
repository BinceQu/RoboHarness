"""Own and reclaim large OmniGibson temporary files.

OmniGibson creates a process-global ``tempfile.mkdtemp()`` directory for
decrypted USD files.  Its normal cleanup is not reached on SIGTERM, SIGKILL,
or the interface task-switch ``os._exit`` path.  Keeping those files below a
project-owned root gives us enough provenance to remove only dead processes'
directories.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Sequence


ROOT_ENV = "BEHAVIOR_INTERFACE_RUNTIME_TMP_ROOT"
DIR_ENV = "BEHAVIOR_INTERFACE_RUNTIME_TMP_DIR"
ORPHAN_GRACE_ENV = "BEHAVIOR_INTERFACE_RUNTIME_TMP_ORPHAN_GRACE_S"
REAP_INTERVAL_ENV = "BEHAVIOR_INTERFACE_RUNTIME_TMP_REAP_S"
LEAF_MAX_ENV = "BEHAVIOR_INTERFACE_RUNTIME_TMP_LEAF_MAX_BYTES"
OWNER_FILE = ".owner.json"
_DIR_NAME_RE = re.compile(r"^port[0-9A-Za-z_.-]+_pid[0-9]+_")
# NVRTC 报 "name of directory for temporary files is too long" 的经验上限是 128。
# 叶子路径必须留余量给编译器再拼文件名；超过 100 就拒绝启动。
NVRTC_TMPDIR_MAX_CHARS = 100
DEFAULT_REAP_INTERVAL_S = 60.0
DEFAULT_LIVE_LEAF_MAX_BYTES = 2 * 1024 ** 3
_SCRATCH_DIR_PREFIXES = (
    "torchinductor",
    "wp_",
    "cuda",
)

_runtime_dir: Optional[str] = None
_runtime_token: Optional[str] = None
_cleanup_registered = False
_signal_handlers: Dict[int, Any] = {}
_reaper_lock = threading.Lock()
_reaper_stop: Optional[threading.Event] = None
_reaper_thread: Optional[threading.Thread] = None


def _cli_flag_value(argv: Sequence[str], name: str) -> str:
    """读取 ``--name value`` 或 ``--name=value``。"""
    flag = name if name.startswith("--") else f"--{name}"
    prefix = flag + "="
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 < len(tokens):
                return str(tokens[index + 1]).strip()
            return ""
        if token.startswith(prefix):
            return str(token[len(prefix) :]).strip()
    return ""


def process_runtime_port_hint(
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """给 TMPDIR 叶子打端口标签。官方栈认 ``--http-port`` / ``BEHAVIOR_EVAL_TEST_PORT``。

    旧 server 入口仍认 ``--port`` / ``PORT``。只有这些都没有时才回退 ``5000``，
    避免多个官方口 import ``server.py`` 时全部写成 ``port5000_`` 并互相 prune。
    """
    tokens = list(sys.argv if argv is None else argv)
    values = os.environ if env is None else env
    for candidate in (
        _cli_flag_value(tokens, "--http-port"),
        str(values.get("BEHAVIOR_EVAL_TEST_PORT", "")).strip(),
        _cli_flag_value(tokens, "--port"),
        str(values.get("PORT", "")).strip(),
    ):
        if candidate:
            return candidate
    return "5000"


def runtime_tmp_root(
    env: Optional[Mapping[str, str]] = None,
    port: object = None,
) -> str:
    values = os.environ if env is None else env
    configured = str(values.get(ROOT_ENV, "")).strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    user = values.get("USER") or str(os.getuid())
    label = _port_label(
        port if port is not None else process_runtime_port_hint(env=values)
    )
    # 官方 HTTP 口各自一份 root，15065 启动时的 prune 看不到 15063 的叶子。
    if label not in {"", "5000", "unknown"}:
        return f"/tmp/behavior_interface_runtime_{user}_p{label}"
    return f"/tmp/behavior_interface_runtime_{user}"


def _validate_root(root: str) -> str:
    root = os.path.abspath(os.path.expanduser(root))
    if root in {"", os.path.sep}:
        raise ValueError(f"unsafe runtime temp root: {root!r}")
    os.makedirs(root, mode=0o700, exist_ok=True)
    if not os.path.isdir(root) or os.path.islink(root):
        raise ValueError(f"runtime temp root must be a real directory: {root}")
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def assert_nvrtc_safe_tmpdir(path: str) -> None:
    """拦 eval_runs 长路径；其它超长叶子只告警。

    occupancy 编译走 ``nvrtc_compile_tmpdir``，不依赖进程 TMPDIR 短。
    15068 NAS 恢复口的 USD 必须留在长叶子里，这里不能一刀切拒绝。
    ``eval_runs`` 是误配，必须启动失败。
    """

    normalized = path.replace("\\", "/")
    if "/eval_runs/" in normalized:
        raise RuntimeError(
            "TMPDIR 不能放在 eval_runs 下 "
            f"({path!r})。必须用 /tmp/behavior_interface_runtime_${{USER}}_p${{PORT}}。"
        )
    if len(path) <= NVRTC_TMPDIR_MAX_CHARS:
        return
    sys.stderr.write(
        "TMPDIR 超过 NVRTC 经验安全长度 "
        f"({len(path)} > {NVRTC_TMPDIR_MAX_CHARS}): {path}。"
        "occupancy 编译会切到 /tmp/behavior_nvrtc_*；"
        "USD 可以留在当前叶子。\n"
    )


@contextmanager
def nvrtc_compile_tmpdir() -> Iterator[str]:
    """只在 Warp/NVRTC 编译窗口切到短目录，不永久改 OmniGibson 的 USD TMPDIR。

    NVRTC 读 TMPDIR；目录名经验上限 128。OG 解密 USD 可以留在长叶子里。
    """

    current = [
        os.environ.get("TMPDIR") or "",
        os.environ.get("TMP") or "",
        os.environ.get("TEMP") or "",
        str(tempfile.gettempdir() or ""),
    ]
    if current and all(len(item) <= NVRTC_TMPDIR_MAX_CHARS for item in current if item):
        yield str(tempfile.gettempdir())
        return
    port = process_runtime_port_hint()
    safe = f"/tmp/behavior_nvrtc_{os.getuid()}_p{_port_label(port)}"
    if len(safe) > NVRTC_TMPDIR_MAX_CHARS:
        safe = f"/tmp/behavior_nvrtc_{os.getuid()}"
    os.makedirs(safe, mode=0o700, exist_ok=True)
    saved = {name: os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP")}
    saved_tempdir = tempfile.tempdir
    try:
        for name in ("TMPDIR", "TMP", "TEMP"):
            os.environ[name] = safe
        tempfile.tempdir = safe
        yield safe
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        tempfile.tempdir = saved_tempdir


def _scratch_dir_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _SCRATCH_DIR_PREFIXES)


def enforce_live_leaf_budget(
    path: Optional[str] = None,
    *,
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """长开时清编译缓存，不动 OmniGibson 正在用的解密 USD。"""

    leaf = path or _runtime_dir
    report: Dict[str, Any] = {
        "path": leaf,
        "bytes": 0,
        "freed_bytes": 0,
        "ok": True,
        "removed": [],
    }
    if not leaf or not os.path.isdir(leaf):
        return report
    try:
        limit = (
            int(max_bytes)
            if max_bytes is not None
            else max(1, int(os.environ.get(LEAF_MAX_ENV, str(DEFAULT_LIVE_LEAF_MAX_BYTES))))
        )
    except (TypeError, ValueError):
        limit = DEFAULT_LIVE_LEAF_MAX_BYTES
    size = _tree_bytes(leaf)
    report["bytes"] = size
    if size <= limit:
        return report
    for name in os.listdir(leaf):
        if not _scratch_dir_name(name):
            continue
        target = os.path.join(leaf, name)
        if os.path.islink(target) or not os.path.isdir(target):
            continue
        freed = _tree_bytes(target)
        try:
            shutil.rmtree(target)
        except OSError:
            continue
        report["removed"].append(name)
        report["freed_bytes"] += freed
    report["bytes"] = _tree_bytes(leaf)
    report["ok"] = report["bytes"] <= limit
    return report


def reclaim_runtime_tmp_root(
    root: Optional[str] = None,
    *,
    orphan_grace_s: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """删掉死进程叶子；root 空了就拆掉，避免 stop 后留空目录。"""

    report = prune_stale_runtime_dirs(
        root,
        orphan_grace_s=orphan_grace_s,
        dry_run=dry_run,
    )
    report["root_removed"] = False
    target = str(report.get("root") or "")
    if dry_run or not target or target in {"", os.path.sep, "/tmp", "/var/tmp"}:
        return report
    try:
        remaining = [
            entry.name
            for entry in os.scandir(target)
            if entry.name not in {".", ".."}
        ]
    except OSError as exc:
        report["errors"].append(f"scandir {target}: {exc}")
        return report
    if remaining:
        return report
    try:
        os.rmdir(target)
    except OSError as exc:
        report["errors"].append(f"rmdir {target}: {exc}")
        return report
    report["root_removed"] = True
    return report


def _reaper_interval_s() -> float:
    try:
        return max(5.0, float(os.environ.get(REAP_INTERVAL_ENV, str(DEFAULT_REAP_INTERVAL_S))))
    except ValueError:
        return DEFAULT_REAP_INTERVAL_S


def _reaper_loop(root: str, stop: threading.Event) -> None:
    while not stop.wait(_reaper_interval_s()):
        try:
            prune_stale_runtime_dirs(root)
            enforce_live_leaf_budget()
        except Exception:
            continue


def _stop_reaper() -> None:
    global _reaper_thread, _reaper_stop
    with _reaper_lock:
        stop = _reaper_stop
        thread = _reaper_thread
        _reaper_stop = None
        _reaper_thread = None
    if stop is not None:
        stop.set()
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)


def _start_reaper(root: str) -> None:
    global _reaper_thread, _reaper_stop
    with _reaper_lock:
        if _reaper_thread is not None and _reaper_thread.is_alive():
            return
        stop = threading.Event()
        thread = threading.Thread(
            target=_reaper_loop,
            args=(root, stop),
            name="behavior-runtime-tmp-reaper",
            daemon=True,
        )
        _reaper_stop = stop
        _reaper_thread = thread
        thread.start()


def _proc_start_ticks(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as stream:
            stat = stream.read().strip()
    except (OSError, ValueError):
        return None
    # comm (field 2) may contain spaces or ')'; fields after its final ') '
    # start at state (field 3). starttime is field 22, hence index 19 here.
    end = stat.rfind(") ")
    if end < 0:
        return None
    fields = stat[end + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def _lock_path(root: str) -> str:
    runtime = f"/run/user/{os.getuid()}"
    if not os.path.isdir(runtime) or not os.access(runtime, os.W_OK):
        runtime = "/dev/shm"
    if not os.path.isdir(runtime) or not os.access(runtime, os.W_OK):
        runtime = "/tmp"
    digest = hashlib.sha256(root.encode("utf-8", "surrogateescape")).hexdigest()[:16]
    return os.path.join(runtime, f"behavior-runtime-tmp-{os.getuid()}-{digest}.lock")


@contextmanager
def _root_lock(root: str) -> Iterator[None]:
    lock_path = _lock_path(root)
    with open(lock_path, "a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_owner(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(path, OWNER_FILE), encoding="utf-8") as stream:
            owner = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    return owner if isinstance(owner, dict) else None


def _owner_is_alive(owner: Dict[str, Any]) -> bool:
    try:
        pid = int(owner["pid"])
        expected = str(owner["start_ticks"])
    except (KeyError, TypeError, ValueError):
        return False
    current = _proc_start_ticks(pid)
    if current is None:
        # /proc/<pid>/stat 读不到时，只要进程目录还在就当活着。
        # 以前这里会立刻 rmtree，把别的官方口还在用的叶子删掉。
        return os.path.isdir(f"/proc/{pid}")
    return current == expected


def _safe_owned_dir(root: str, path: str) -> bool:
    path = os.path.abspath(path)
    try:
        inside = os.path.commonpath((root, path)) == root
    except ValueError:
        return False
    return (
        inside
        and os.path.dirname(path) == root
        and bool(_DIR_NAME_RE.match(os.path.basename(path)))
        and os.path.isdir(path)
        and not os.path.islink(path)
    )


def _tree_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                stat = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
                total += stat.st_blocks * 512
            except OSError:
                continue
    return total


def prune_stale_runtime_dirs(
    root: Optional[str] = None,
    *,
    orphan_grace_s: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Remove marked directories whose owning PID/starttime no longer exists.

    A valid PID plus matching kernel starttime is treated as active. Unmarked
    or malformed directories receive a grace period because they may belong to
    a process that was interrupted between mkdir and writing its marker.
    """

    root = _validate_root(root or runtime_tmp_root())
    if orphan_grace_s is None:
        try:
            orphan_grace_s = max(0.0, float(os.environ.get(ORPHAN_GRACE_ENV, "3600")))
        except ValueError:
            orphan_grace_s = 3600.0
    report: Dict[str, Any] = {
        "root": root,
        "removed": 0,
        "kept": 0,
        "freed_bytes": 0,
        "errors": [],
    }
    now = time.time()
    with _root_lock(root):
        try:
            entries = list(os.scandir(root))
        except OSError as exc:
            report["errors"].append(f"scandir {root}: {exc}")
            return report
        for entry in entries:
            path = entry.path
            if not entry.is_dir(follow_symlinks=False) or not _safe_owned_dir(root, path):
                continue
            owner = _read_owner(path)
            if owner is not None and _owner_is_alive(owner):
                report["kept"] += 1
                continue
            if owner is None:
                try:
                    age_s = now - entry.stat(follow_symlinks=False).st_mtime
                except OSError:
                    age_s = 0.0
                if age_s < orphan_grace_s:
                    report["kept"] += 1
                    continue
            size = _tree_bytes(path)
            try:
                if not dry_run:
                    shutil.rmtree(path)
            except OSError as exc:
                report["errors"].append(f"rmtree {path}: {exc}")
                continue
            report["removed"] += 1
            report["freed_bytes"] += size
    return report


def _write_owner(path: str, owner: Dict[str, Any]) -> None:
    marker = os.path.join(path, OWNER_FILE)
    with open(marker, "x", encoding="utf-8") as stream:
        json.dump(owner, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _port_label(port: object) -> str:
    label = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(port or "unknown"))
    return label[:64] or "unknown"


def _install_signal_cleanup() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)
        _signal_handlers.setdefault(sig, previous)

        def handler(signum, frame, *, _previous=previous):
            try:
                cleanup_process_runtime_tmp()
                if callable(_previous):
                    _previous(signum, frame)
            finally:
                # A custom handler may return or raise a catchable exception.
                # The managed directory is already gone, so this process must
                # not continue running in either case.
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(sig, handler)


def _reset_runtime_state_after_fork() -> None:
    """Drop a parent's leaf from a forked child before it can be reused."""

    global _runtime_dir, _runtime_token
    _stop_reaper()
    inherited = _runtime_dir
    _runtime_dir = None
    _runtime_token = None
    if inherited is None:
        return
    os.environ.pop(DIR_ENV, None)
    root = runtime_tmp_root()
    for name in ("TMPDIR", "TMP", "TEMP"):
        if os.environ.get(name) == inherited:
            os.environ[name] = root
    if tempfile.tempdir == inherited:
        tempfile.tempdir = root


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_runtime_state_after_fork)


def configure_process_runtime_tmp(
    port: object = "unknown",
    *,
    root: Optional[str] = None,
    register_cleanup: bool = True,
) -> str:
    """Create and select a process-owned TMPDIR before OmniGibson imports."""

    global _cleanup_registered, _runtime_dir, _runtime_token
    if _runtime_dir and os.path.isdir(_runtime_dir):
        owner = _read_owner(_runtime_dir)
        try:
            owned_here = (
                owner is not None
                and int(owner.get("pid", -1)) == os.getpid()
                and str(owner.get("start_ticks")) == _proc_start_ticks(os.getpid())
                and owner.get("token") == _runtime_token
            )
        except (TypeError, ValueError):
            owned_here = False
        if owned_here:
            return _runtime_dir
        _reset_runtime_state_after_fork()

    root = _validate_root(root or runtime_tmp_root(port=port))
    prune_stale_runtime_dirs(root)
    pid = os.getpid()
    start_ticks = _proc_start_ticks(pid)
    if start_ticks is None:
        raise RuntimeError(f"cannot read /proc/{pid}/stat for runtime temp ownership")
    token = secrets.token_hex(16)
    path = tempfile.mkdtemp(
        prefix=f"port{_port_label(port)}_pid{pid}_",
        dir=root,
    )
    try:
        assert_nvrtc_safe_tmpdir(path)
        _write_owner(
            path,
            {
                "pid": pid,
                "start_ticks": start_ticks,
                "port": str(port),
                "created_at": time.time(),
                "token": token,
            },
        )
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise

    _runtime_dir = path
    _runtime_token = token
    os.environ[ROOT_ENV] = root
    os.environ[DIR_ENV] = path
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = path
    # tempfile caches its selected directory after the first call.
    tempfile.tempdir = path

    if register_cleanup and not _cleanup_registered:
        atexit.register(cleanup_process_runtime_tmp)
        _install_signal_cleanup()
        _cleanup_registered = True
    if register_cleanup:
        _start_reaper(root)
    return path


def ensure_process_runtime_tmp(
    port: object = None,
    *,
    root: Optional[str] = None,
    register_cleanup: bool = True,
) -> str:
    """叶子被别的进程删掉时，重建本进程 TMPDIR，不要继续指向幽灵路径。"""

    hint = process_runtime_port_hint() if port is None else port
    if tempfile.tempdir and not os.path.isdir(str(tempfile.tempdir)):
        tempfile.tempdir = None
    return configure_process_runtime_tmp(
        hint,
        root=root,
        register_cleanup=register_cleanup,
    )


def prepare_child_environment(
    env: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
    """Point a child at the managed root, never at its parent's leaf TMPDIR."""

    values = os.environ.copy() if env is None else env
    root = _validate_root(runtime_tmp_root(values))
    values[ROOT_ENV] = root
    values.pop(DIR_ENV, None)
    for name in ("TMPDIR", "TMP", "TEMP"):
        values[name] = root
    return values


def cleanup_process_runtime_tmp() -> bool:
    """Remove this process's runtime directory after verifying its token."""

    global _runtime_dir, _runtime_token
    path = _runtime_dir or os.environ.get(DIR_ENV)
    if not path:
        return False
    root = runtime_tmp_root()
    if not _safe_owned_dir(root, path):
        return False
    owner = _read_owner(path)
    if not owner or owner.get("token") != _runtime_token:
        return False
    if int(owner.get("pid", -1)) != os.getpid():
        return False
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    _runtime_dir = None
    _runtime_token = None
    os.environ.pop(DIR_ENV, None)
    for name in ("TMPDIR", "TMP", "TEMP"):
        if os.environ.get(name) == path:
            os.environ[name] = root
    tempfile.tempdir = root
    return True


def _main() -> int:
    parser = argparse.ArgumentParser(description="Manage BEHAVIOR process-owned TMPDIRs")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--orphan-grace-s", type=float, default=None)
    parser.add_argument(
        "--reclaim",
        action="store_true",
        help="prune 死叶子；root 空了就删除",
    )
    args = parser.parse_args()
    if args.reclaim:
        report = reclaim_runtime_tmp_root(
            args.root,
            orphan_grace_s=args.orphan_grace_s,
            dry_run=args.dry_run,
        )
    else:
        report = prune_stale_runtime_dirs(
            args.root,
            orphan_grace_s=args.orphan_grace_s,
            dry_run=args.dry_run,
        )
    print(json.dumps(report, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
