"""Agent-run 会话管理（server 侧权威存储）。

设计要点（见 embodied-agent/BEHAVIOR_AGENT_DESIGN.md §4）：
- server 是 session 的**权威方**：capture 出的图片、深度/分割/法向/相机内外参（层 B 反投影资源）
  以及 plan_* 产出的 EefPlanRecord（move_NNNN）都落在 server 文件系统并由 server 编号。
- 远程 agent（笔记本经 SSH 端口转发）只拿到 server 分配的字符串编号（img_NNNN / move_NNNN）
  与主视图 base64，后续 plan_*(img_id) / exec_move(plan_id) 只回传编号，重数据始终在 server。

目录结构：
    $BEHAVIOR_AGENT_RUNS/<session_id>/
    ├── images/ img_0001.png / img_0001.depth.npy / img_0001.seg.npy /
    │           img_0001.normal.npy / img_0001.meta.json
    ├── plans/  move_0001.png / move_0001.json
    └── session.json   # 计数器
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import json
import os
import socket
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

# Durable recording root.  Keep an explicit environment override for tests and
# deployments that intentionally use a different filesystem; ordinary runs
# should survive /tmp cleanup and stay together under the NAS archive folder.
DEFAULT_RUNS_ROOT = os.path.expanduser("~/.local/share/roboharness/recordings")
RUNS_ROOT = os.environ.get("BEHAVIOR_AGENT_RUNS", DEFAULT_RUNS_ROOT)

# plan_grasp / plan_eef 的会话产物根目录，与 RUNS_ROOT 一起纳入保留策略
GRASP_SESSION_ROOT = os.environ.get(
    "PLAN_GRASP_SESSION_ROOT", "/tmp/plan_grasp_sessions"
)

_LOCKS_DIR = ".locks"
_ACTIVE_LEASES_DIR = ".active_leases"
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, list[Any]] = {}
_THREAD_LOCK_STATE = threading.local()


def _reset_process_locks_after_fork() -> None:
    global _PROCESS_LOCKS_GUARD, _PROCESS_LOCKS, _THREAD_LOCK_STATE
    _PROCESS_LOCKS_GUARD = threading.Lock()
    _PROCESS_LOCKS = {}
    _THREAD_LOCK_STATE = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_locks_after_fork)


def _validate_path_component(value: str, label: str) -> str:
    """验证来自 API 的 id 是单一文件名，而不是路径。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空字符串")
    if value in {".", ".."} or os.path.isabs(value):
        raise ValueError(f"非法 {label}: {value!r}")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ValueError(f"{label} 不能包含路径分隔符: {value!r}")
    return value


def _contained_path(root: str, *parts: str) -> str:
    """拼接后验证 realpath 仍位于 root 内，阻止 symlink/.. 逃逸。"""
    root_abs = os.path.abspath(root)
    candidate = os.path.abspath(os.path.join(root_abs, *parts))
    root_real = os.path.realpath(root_abs)
    candidate_real = os.path.realpath(candidate)
    try:
        contained = os.path.commonpath((root_real, candidate_real)) == root_real
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"路径逃逸管理根目录: {candidate!r} not under {root_abs!r}")
    return candidate


def managed_session_dir(root: str, session_id: str) -> str:
    """返回 root 下经过边界校验的一级会话目录。"""
    component = _validate_path_component(session_id, "session_id")
    if component in {_LOCKS_DIR, _ACTIVE_LEASES_DIR}:
        raise ValueError(f"保留的 session_id: {component!r}")
    return _contained_path(root, component)


def _normalized_session_key(root: str, path: str) -> str:
    root_real = os.path.realpath(os.path.abspath(root))
    path_real = os.path.realpath(os.path.abspath(path))
    try:
        contained = os.path.commonpath((root_real, path_real)) == root_real
    except ValueError:
        contained = False
    if not contained or path_real == root_real:
        raise ValueError(f"会话路径不在管理根目录内: {path!r}")
    return path_real


@contextmanager
def _process_session_lock(key: str):
    """同进程仅串行化同一规范化会话；不同会话可以并行。"""
    with _PROCESS_LOCKS_GUARD:
        entry = _PROCESS_LOCKS.get(key)
        if entry is None:
            entry = [threading.RLock(), 0]
            _PROCESS_LOCKS[key] = entry
        entry[1] += 1
        lock = entry[0]
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _PROCESS_LOCKS_GUARD:
            entry[1] -= 1
            if entry[1] == 0 and _PROCESS_LOCKS.get(key) is entry:
                del _PROCESS_LOCKS[key]


def _session_lock_path(root: str, path: str) -> str:
    key = _normalized_session_key(root, path)
    digest = hashlib.sha256(
        key.encode("utf-8", errors="surrogateescape")
    ).hexdigest()
    return _contained_path(root, _LOCKS_DIR, f"{digest}.lock")


@contextmanager
def _locked_session(root: str, path: str):
    """串行化一个会话的计数器更新与删除。

    锁文件放在会话目录之外，避免清理线程删掉锁 inode 后，
    新写入者在同一路径创建了另一把锁。
    """
    key = _normalized_session_key(root, path)
    with _process_session_lock(key):
        held = getattr(_THREAD_LOCK_STATE, "held", None)
        if held is None:
            held = {}
            _THREAD_LOCK_STATE.held = held
        if key in held:
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return

        lock_path = _session_lock_path(root, path)
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held[key] = 1
            yield
        finally:
            try:
                held.pop(key, None)
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _atomic_json_write(path: str, value: Dict[str, Any]) -> None:
    """在同目录写唯一临时文件后原子替换 JSON。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent
    )
    try:
        try:
            mode = os.stat(path).st_mode & 0o777
        except OSError:
            mode = 0o664
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        tmp = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────────
# 路径
# ──────────────────────────────────────────────────────────────────────────

def run_dir(session_id: str) -> str:
    return managed_session_dir(RUNS_ROOT, session_id)


def images_dir(session_id: str) -> str:
    return _contained_path(run_dir(session_id), "images")


def plans_dir(session_id: str) -> str:
    return _contained_path(run_dir(session_id), "plans")


def _session_json(session_id: str) -> str:
    return _contained_path(run_dir(session_id), "session.json")


# ──────────────────────────────────────────────────────────────────────────
# active lease
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ActiveSessionLease:
    root: str
    session_id: str
    session_path: str
    token: str
    pid: int
    process_starttime: str
    lease_path: str
    fd: int = field(repr=False)
    released: bool = False
    _guard: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


def _process_starttime(pid: int) -> Optional[str]:
    """读取 Linux /proc stat 的 starttime，避免 PID 复用误判。"""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as stream:
            raw = stream.read()
        close_paren = raw.rfind(")")
        if close_paren < 0:
            return None
        fields = raw[close_paren + 1:].split()
        # fields[0] 是 stat 第 3 项 state，starttime 是第 22 项。
        return fields[19] if len(fields) > 19 else None
    except (OSError, TypeError, ValueError):
        return None


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "active lease write returned zero bytes")
        offset += written


def acquire_session_lease(
    session_id: str,
    *,
    root: Optional[str] = None,
) -> ActiveSessionLease:
    """取得跨进程 active lease；调用方必须持有返回对象直至会话结束。"""
    lease_root = os.path.abspath(root or RUNS_ROOT)
    session_path = managed_session_dir(lease_root, session_id)
    pid = os.getpid()
    process_starttime = _process_starttime(pid)
    if process_starttime is None:
        raise RuntimeError(f"无法读取进程 {pid} 的 starttime，拒绝创建不可靠 lease")
    token = uuid.uuid4().hex
    lease_dir = _contained_path(session_path, _ACTIVE_LEASES_DIR)
    lease_path = _contained_path(lease_dir, f"{token}.json")
    fd = -1
    with _locked_session(lease_root, session_path):
        if not os.path.isdir(session_path):
            raise FileNotFoundError(f"会话不存在，无法取得 active lease: {session_id}")
        os.makedirs(lease_dir, exist_ok=True)
        fd = os.open(lease_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            payload = json.dumps(
                {
                    "pid": pid,
                    "process_starttime": process_starttime,
                    "token": token,
                    "hostname": socket.gethostname(),
                    "created_at": time.time(),
                },
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            _write_all(fd, payload)
            os.fsync(fd)
        except BaseException:
            try:
                if fd >= 0:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
            finally:
                fd = -1
                try:
                    os.unlink(lease_path)
                except OSError:
                    pass
            raise
    return ActiveSessionLease(
        root=lease_root,
        session_id=session_id,
        session_path=session_path,
        token=token,
        pid=pid,
        process_starttime=process_starttime,
        lease_path=lease_path,
        fd=fd,
    )


def _lease_file_is_live(path: str) -> bool:
    """调用方持有 session lock；不确定时保守地视为活跃。"""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True

    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            return True

        try:
            with os.fdopen(os.dup(fd), "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            token = str(payload.get("token") or "")
            if os.path.basename(path) != f"{token}.json":
                return False
            if str(payload.get("hostname") or "") != socket.gethostname():
                return False
            pid = int(payload.get("pid"))
            expected_starttime = str(payload.get("process_starttime") or "")
            return bool(
                expected_starttime
                and _process_starttime(pid) == expected_starttime
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return False
    finally:
        try:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _session_has_live_lease(session_path: str) -> bool:
    try:
        lease_dir = _contained_path(session_path, _ACTIVE_LEASES_DIR)
    except ValueError:
        return True
    if not os.path.isdir(lease_dir):
        return False
    try:
        entries = list(os.scandir(lease_dir))
    except OSError:
        return True
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        if _lease_file_is_live(entry.path):
            return True
        try:
            os.unlink(entry.path)
        except FileNotFoundError:
            pass
        except OSError:
            return True
    return False


def session_has_live_lease(
    session_id: str,
    *,
    root: Optional[str] = None,
) -> bool:
    lease_root = os.path.abspath(root or RUNS_ROOT)
    session_path = managed_session_dir(lease_root, session_id)
    with _locked_session(lease_root, session_path):
        return _session_has_live_lease(session_path)


def release_session_lease(lease: ActiveSessionLease) -> None:
    """释放 lease；token 校验防止误删同路径下其他持有者的文件。"""
    with lease._guard:
        if lease.released:
            return
        try:
            with _locked_session(lease.root, lease.session_path):
                try:
                    with open(lease.lease_path, encoding="utf-8") as stream:
                        payload = json.load(stream)
                    if str(payload.get("token") or "") == lease.token:
                        os.unlink(lease.lease_path)
                except (AttributeError, OSError, TypeError, ValueError):
                    pass
        finally:
            try:
                if lease.fd >= 0:
                    try:
                        fcntl.flock(lease.fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lease.fd)
            finally:
                lease.fd = -1
                lease.released = True


# ──────────────────────────────────────────────────────────────────────────
# session.json（计数器）
# ──────────────────────────────────────────────────────────────────────────

def _load_meta(session_id: str) -> Dict[str, Any]:
    path = _session_json(session_id)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"session_id": session_id, "image_counter": 0, "plan_counter": 0}


def _save_meta(session_id: str, meta: Dict[str, Any]) -> None:
    _atomic_json_write(_session_json(session_id), meta)


def ensure_session(session_id: str) -> str:
    """确保 session 目录存在，返回 session_id。"""
    path = run_dir(session_id)
    with _locked_session(RUNS_ROOT, path):
        os.makedirs(images_dir(session_id), exist_ok=True)
        os.makedirs(plans_dir(session_id), exist_ok=True)
        meta = _load_meta(session_id)
        _save_meta(session_id, meta)
    return session_id


def next_image_id(session_id: str) -> str:
    """分配下一个 img_NNNN 编号（自增、落盘）。"""
    with _locked_session(RUNS_ROOT, run_dir(session_id)):
        meta = _load_meta(session_id)
        meta["image_counter"] = int(meta.get("image_counter", 0)) + 1
        _save_meta(session_id, meta)
        return f"img_{meta['image_counter']:04d}"


def next_plan_id(session_id: str) -> str:
    """分配下一个 move_NNNN 编号（自增、落盘）。"""
    with _locked_session(RUNS_ROOT, run_dir(session_id)):
        meta = _load_meta(session_id)
        meta["plan_counter"] = int(meta.get("plan_counter", 0)) + 1
        _save_meta(session_id, meta)
        return f"move_{meta['plan_counter']:04d}"


# ──────────────────────────────────────────────────────────────────────────
# image bundle（capture 落盘）
# ──────────────────────────────────────────────────────────────────────────

def image_path(session_id: str, image_id: str, suffix: str = ".png") -> str:
    component = _validate_path_component(image_id, "image_id")
    if (
        not isinstance(suffix, str)
        or "\x00" in suffix
        or "/" in suffix
        or "\\" in suffix
    ):
        raise ValueError(f"非法 image suffix: {suffix!r}")
    return _contained_path(images_dir(session_id), f"{component}{suffix}")


def image_meta_path(session_id: str, image_id: str) -> str:
    return image_path(session_id, image_id, ".meta.json")


def save_image_meta(session_id: str, image_id: str, meta: Dict[str, Any]) -> None:
    """落盘 img_NNNN.meta.json：相机内外参 + tro + robot 状态 + 各模态文件路径。"""
    with _locked_session(RUNS_ROOT, run_dir(session_id)):
        _atomic_json_write(image_meta_path(session_id, image_id), meta)


def load_image_meta(session_id: str, image_id: str) -> Dict[str, Any]:
    """plan_*(img_id) 时取回该图的相机内外参/模态文件路径（用于反投影）。"""
    path = image_meta_path(session_id, image_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image meta 不存在: {session_id}/{image_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_capture_image_id(
    session_id: str,
    *,
    require_camera: bool = False,
    require_depth: bool = False,
) -> Optional[str]:
    """返回当前 session 编号最新、且满足指定模态要求的 capture image_id。"""
    image_dir = images_dir(session_id)
    if not os.path.isdir(image_dir):
        return None

    candidates = []
    for name in os.listdir(image_dir):
        if not (name.startswith("img_") and name.endswith(".meta.json")):
            continue
        image_id = name[:-len(".meta.json")]
        try:
            counter = int(image_id.removeprefix("img_"))
        except ValueError:
            counter = -1
        try:
            mtime = float(os.path.getmtime(os.path.join(image_dir, name)))
        except OSError:
            mtime = 0.0
        candidates.append((counter, mtime, image_id))

    for _counter, _mtime, image_id in sorted(candidates, reverse=True):
        try:
            meta = load_image_meta(session_id, image_id)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
        if require_camera:
            camera = meta.get("camera") or {}
            if not camera.get("pos") or not camera.get("quat"):
                continue
        if require_depth:
            modalities = meta.get("modalities") or {}
            depth_name = str(
                modalities.get("depth")
                or modalities.get("depth_linear")
                or f"{image_id}.depth.npy"
            )
            depth_path = os.path.join(image_dir, depth_name)
            try:
                if not os.path.isfile(depth_path) or os.path.getsize(depth_path) <= 0:
                    continue
            except OSError:
                continue
        return image_id
    return None


# ──────────────────────────────────────────────────────────────────────────
# EefPlanRecord（plan_* 落盘 / exec_move 读取）
# ──────────────────────────────────────────────────────────────────────────

def plan_path(session_id: str, plan_id: str, suffix: str = ".json") -> str:
    component = _validate_path_component(plan_id, "plan_id")
    if (
        not isinstance(suffix, str)
        or "\x00" in suffix
        or "/" in suffix
        or "\\" in suffix
    ):
        raise ValueError(f"非法 plan suffix: {suffix!r}")
    return _contained_path(plans_dir(session_id), f"{component}{suffix}")


def save_plan_record(session_id: str, plan_id: str, record: Dict[str, Any]) -> None:
    with _locked_session(RUNS_ROOT, run_dir(session_id)):
        _atomic_json_write(plan_path(session_id, plan_id), record)


def load_plan_record(session_id: str, plan_id: str) -> Dict[str, Any]:
    path = plan_path(session_id, plan_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"plan record 不存在: {session_id}/{plan_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# 磁盘保留策略
#
# capture / plan_* 每次调用都会落盘 png+npy（单次 rollout 约 40 MiB），而会话
# 目录此前只增不减，长期跑必然写满分区。写满后 np.save 会留下截断文件，下游
# 把坏数据当好数据读，比直接报错更危险，所以这里按年龄回收陈旧会话。
# ──────────────────────────────────────────────────────────────────────────

def disk_free_mib(path: str = RUNS_ROOT) -> float:
    """path 所在文件系统剩余空间（MiB）；取不到时返回 -1。"""
    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free / (1024.0 * 1024.0)
    except OSError:
        return -1.0


def _session_activity_ts(path: str) -> float:
    """会话最近活动时间。

    只看目录自身 mtime 会误判：session.json 是覆盖写，不会更新父目录 mtime，
    活跃会话可能看起来很旧。images/ 每次 capture 新建文件必然更新其 mtime，
    因此取这几个位置的最大值作为活动信号。
    """
    newest = 0.0
    for rel in (
        "",
        "session.json",
        "images",
        "plans",
        "events.jsonl",
        "recording.json",
        "turns",
    ):
        try:
            newest = max(newest, os.stat(os.path.join(path, rel)).st_mtime)
        except OSError:
            continue
    turns = os.path.join(path, "turns")
    if os.path.isdir(turns):
        for dirpath, dirnames, filenames in os.walk(turns):
            for name in [*dirnames, *filenames]:
                try:
                    newest = max(
                        newest, os.stat(os.path.join(dirpath, name)).st_mtime
                    )
                except OSError:
                    continue
    return newest


def prune_stale_sessions(
    root: str,
    max_age_h: float,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """删除 root 下最近活动早于 max_age_h 的一级会话目录。

    以活动时间而非创建时间为准，正在写入的会话不会被回收。
    """
    result: Dict[str, Any] = {
        "root": root, "removed": 0, "freed_mib": 0.0, "kept": 0, "errors": [],
    }
    if max_age_h <= 0 or not os.path.isdir(root):
        return result
    cutoff = time.time() - max_age_h * 3600.0
    victims: List[tuple[str, str]] = []
    try:
        entries = list(os.scandir(root))
    except OSError as e:
        result["errors"].append(f"scandir {root}: {e}")
        return result
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        if entry.name == _LOCKS_DIR:
            continue
        if _session_activity_ts(entry.path) >= cutoff:
            result["kept"] += 1
            continue
        victims.append((entry.name, entry.path))
    for _name, path in victims:
        size_mib = 0.0
        try:
            with _locked_session(root, path):
                # 会话可能在首次扫描后重新开始写入。删除前必须
                # 在与计数器/lease 更新相同的跨进程锁内重新判定。
                if not os.path.isdir(path):
                    continue
                if _session_has_live_lease(path):
                    result["kept"] += 1
                    continue
                if _session_activity_ts(path) >= cutoff:
                    result["kept"] += 1
                    continue
                for dirpath, _dirnames, filenames in os.walk(path):
                    for name in filenames:
                        try:
                            size_mib += os.path.getsize(os.path.join(dirpath, name))
                        except OSError:
                            pass
                size_mib /= 1024.0 * 1024.0
                if not dry_run:
                    shutil.rmtree(path)
        except (OSError, ValueError) as e:
            result["errors"].append(f"rmtree {path}: {e}")
            continue
        result["removed"] += 1
        result["freed_mib"] += size_mib
    return result


def prune_all_stale_sessions(
    max_age_h: float, *, dry_run: bool = False
) -> Dict[str, Any]:
    """对 agent runs 与 plan_grasp 两个会话根一起执行保留策略。"""
    roots = [RUNS_ROOT, GRASP_SESSION_ROOT]
    reports = [prune_stale_sessions(r, max_age_h, dry_run=dry_run) for r in roots]
    return {
        "removed": sum(r["removed"] for r in reports),
        "freed_mib": sum(r["freed_mib"] for r in reports),
        "errors": [e for r in reports for e in r["errors"]],
        "roots": reports,
    }


def file_to_data_url(path: str, mime: str = "image/png") -> Optional[str]:
    """把本地文件编码成 data URL（供 ToolOutputImage / JSON 内联返回）。"""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"
