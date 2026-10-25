"""Storage preflight shared by interface and official evaluator launchers."""

from __future__ import annotations

import argparse
import atexit
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

if __package__:
    from behavior_interface.runtime_tmp import (
        DIR_ENV as RUNTIME_TMP_DIR_ENV,
        prune_stale_runtime_dirs,
        runtime_tmp_root,
    )
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from behavior_interface.runtime_tmp import (
        DIR_ENV as RUNTIME_TMP_DIR_ENV,
        prune_stale_runtime_dirs,
        runtime_tmp_root,
    )


SOFT_FREE_ENV = "BEHAVIOR_INTERFACE_LOCAL_MIN_FREE_GIB"
HARD_FREE_ENV = "BEHAVIOR_INTERFACE_START_MIN_FREE_GIB"
CACHE_MAX_ENV = "BEHAVIOR_INTERFACE_TEXCACHE_MAX_GIB"
LOCK_ENV = "BEHAVIOR_TEXTURECACHE_LOCK_FILE"
AUTO_CLEAN_ENV = "BEHAVIOR_INTERFACE_AUTO_STORAGE_CLEANUP"
ROOTS_ENV = "BEHAVIOR_INTERFACE_MANAGED_APPDATA_ROOTS"
RESERVATION_DIR_ENV = "BEHAVIOR_TEXTURECACHE_RESERVATION_DIR"
PROTECTED_APPDATA_ENV = "BEHAVIOR_INTERFACE_PROTECTED_APPDATA_PATHS"
WATCHDOG_INTERVAL_ENV = "BEHAVIOR_INTERFACE_STORAGE_WATCHDOG_INTERVAL_S"
WATCHDOG_KILL_GRACE_ENV = "BEHAVIOR_INTERFACE_STORAGE_WATCHDOG_KILL_GRACE_S"
WATCHDOG_SAMPLE_ERROR_LIMIT_ENV = (
    "BEHAVIOR_INTERFACE_STORAGE_WATCHDOG_SAMPLE_ERROR_LIMIT"
)

_owned_reservations: Dict[str, str] = {}
_reservation_cleanup_registered = False
_watchdog_cleanup_registered = False
_watchdog_registry_lock = threading.Lock()
_process_storage_watchdog: Optional["StorageWatchdog"] = None
_OFFICIAL_POLICY_MODULE = b"behavior_interface_eval_test.official_policy_interface"


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def storage_lock_path() -> str:
    configured = os.environ.get(LOCK_ENV, "").strip()
    if configured:
        if not os.path.isabs(configured):
            raise ValueError(f"{LOCK_ENV} must be an absolute path: {configured!r}")
        return os.path.abspath(configured)
    runtime = f"/run/user/{os.getuid()}"
    if not (os.path.isdir(runtime) and os.access(runtime, os.W_OK)):
        runtime = "/dev/shm"
    if not (os.path.isdir(runtime) and os.access(runtime, os.W_OK)):
        runtime = "/tmp"
    return os.path.join(
        runtime, f"behavior_texturecache_maintenance_{os.getuid()}.lock"
    )


@contextmanager
def storage_lock() -> Iterator[None]:
    path = storage_lock_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _normalize(path: str) -> Set[str]:
    absolute = os.path.abspath(os.path.expanduser(path))
    return {absolute, os.path.realpath(absolute)}


def _proc_start_ticks(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as stream:
            stat = stream.read()
    except (OSError, ValueError):
        return None
    end = stat.rfind(") ")
    fields = stat[end + 2 :].split() if end >= 0 else []
    return fields[19] if len(fields) > 19 else None


def _reservation_dir() -> str:
    configured = os.environ.get(RESERVATION_DIR_ENV, "").strip()
    if configured:
        if not os.path.isabs(configured):
            raise ValueError(
                f"{RESERVATION_DIR_ENV} must be an absolute path: {configured!r}"
            )
        path = os.path.abspath(configured)
    else:
        path = os.path.join(
            os.path.dirname(storage_lock_path()),
            f"behavior_texturecache_reservations_{os.getuid()}",
        )
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not os.path.isdir(path) or os.path.islink(path):
        raise ValueError(f"reservation root must be a real directory: {path}")
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _cleanup_owned_reservations() -> None:
    _stop_process_storage_watchdog()
    for marker in list(_owned_reservations):
        _release_owned_reservation(marker)


def _release_owned_reservation(marker: str) -> None:
    token = _owned_reservations.get(marker)
    if token is None:
        return
    try:
        with open(marker, encoding="utf-8") as stream:
            value = json.load(stream)
        if (
            int(value.get("pid", -1)) == os.getpid()
            and value.get("token") == token
        ):
            os.unlink(marker)
    except (OSError, TypeError, ValueError):
        pass
    finally:
        _owned_reservations.pop(marker, None)


def _reservation_records() -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    try:
        entries = list(os.scandir(_reservation_dir()))
    except OSError:
        return records
    for entry in entries:
        if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
            continue
        try:
            with open(entry.path, encoding="utf-8") as stream:
                value = json.load(stream)
            pid = int(value["pid"])
            start_ticks = str(value["start_ticks"])
            reserved_bytes = max(0, int(value["reserved_bytes"]))
            appdata = os.path.realpath(str(value["appdata"]))
            device = int(value["device"])
            if _proc_start_ticks(pid) != start_ticks:
                raise ValueError("stale reservation owner")
        except (OSError, KeyError, TypeError, ValueError):
            try:
                os.unlink(entry.path)
            except OSError:
                pass
            continue
        records.append(
            {
                "path": entry.path,
                "pid": pid,
                "start_ticks": start_ticks,
                "reserved_bytes": reserved_bytes,
                "appdata": appdata,
                "device": device,
            }
        )
    return records


def _write_reservation(appdata: str, reserved_bytes: int, pid: int) -> str:
    global _reservation_cleanup_registered

    start_ticks = _proc_start_ticks(pid)
    if start_ticks is None:
        raise RuntimeError(f"cannot reserve storage for dead pid {pid}")
    try:
        if os.stat(f"/proc/{pid}", follow_symlinks=False).st_uid != os.getuid():
            raise RuntimeError(f"refusing storage reservation for another uid: pid {pid}")
    except FileNotFoundError as exc:
        raise RuntimeError(f"cannot reserve storage for dead pid {pid}") from exc
    appdata_real = os.path.realpath(appdata)
    digest = hashlib.sha256(appdata_real.encode("utf-8", "surrogateescape")).hexdigest()[:20]
    marker = os.path.join(_reservation_dir(), f"{digest}_pid{pid}_{start_ticks}.json")
    token = os.urandom(16).hex()
    value = {
        "pid": pid,
        "start_ticks": start_ticks,
        "token": token,
        "appdata": appdata_real,
        "device": os.stat(appdata).st_dev,
        "reserved_bytes": max(0, int(reserved_bytes)),
        "created_at": time.time(),
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(marker)}.", dir=_reservation_dir())
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        temporary = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    if pid == os.getpid():
        _owned_reservations[marker] = token
        if not _reservation_cleanup_registered:
            atexit.register(_cleanup_owned_reservations)
            _reservation_cleanup_registered = True
    return marker


def _ancestor_pids(pid: int) -> Set[int]:
    result: Set[int] = set()
    current = int(pid)
    while current > 1 and current not in result:
        result.add(current)
        try:
            with open(f"/proc/{current}/stat", encoding="utf-8") as stream:
                stat = stream.read()
            end = stat.rfind(") ")
            fields = stat[end + 2 :].split()
            current = int(fields[1])  # ppid: field 4, index 1 after state
        except (OSError, ValueError, IndexError):
            break
    return result


def _process_may_use_appdata(pid: int) -> bool:
    """Conservatively classify an unreadable same-UID process.

    Some ordinary session processes (notably sshd and systemd's pam helper)
    deliberately make ``environ`` unreadable. They must not disable storage
    maintenance globally. Unreadable Python, Kit, OG, or interface processes
    remain fail-closed because they may own a managed cache.
    """

    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as stream:
            stat = stream.read()
    except (OSError, ValueError):
        return False
    end = stat.rfind(") ")
    fields = stat[end + 2 :].split() if end >= 0 else []
    if fields and fields[0] == "Z":
        return False
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as stream:
            cmdline = stream.read().replace(b"\0", b" ").lower()
    except OSError:
        return True
    if not cmdline:
        return True
    executable = os.path.basename(cmdline.split(None, 1)[0]).decode(
        "utf-8", "ignore"
    )
    if executable in {"(sd-pam)", "sd-pam"} or executable.startswith("sshd"):
        return False
    if executable in {"codex", "bwrap", "node", "electron"} and b"codex" in cmdline:
        return False
    # Unknown unreadable processes stay fail-closed. This covers launchers such
    # as uv or a renamed Python executable without maintaining a fragile list.
    return True


def _official_policy_cmdline(pid: int) -> Optional[bool]:
    """Return whether *pid* is the official policy process, or None if unknown."""

    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as stream:
            argv = [value for value in stream.read().split(b"\0") if value]
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return any(
        argv[index] == b"-m" and argv[index + 1] == _OFFICIAL_POLICY_MODULE
        for index in range(len(argv) - 1)
    )


def _protected_appdata_from_environ(
    values: Iterable[bytes],
) -> Tuple[Set[str], int, bool]:
    """Return normalized protected paths, variable count, and validity."""

    prefix = f"{PROTECTED_APPDATA_ENV}=".encode()
    protected: Set[str] = set()
    count = 0
    valid = True
    for value in values:
        if not value.startswith(prefix):
            continue
        count += 1
        raw = value[len(prefix) :].decode("utf-8", "surrogateescape")
        paths = raw.split(os.pathsep)
        if not raw or any(not path or not os.path.isabs(path) for path in paths):
            valid = False
            continue
        for path in paths:
            protected.update(_normalize(path))
    if count > 1:
        valid = False
    if not valid:
        protected.clear()
    return protected, count, valid


def _active_appdata_snapshot_details(
    ignore_pids: Iterable[int] = (),
    ignore_protected_paths: Iterable[str] = (),
) -> Tuple[Set[str], bool, bool]:
    """Return active paths, scan completeness, and unknown legacy policy state."""

    ignored = {int(pid) for pid in ignore_pids}
    ignored_protected: Set[str] = set()
    for path in ignore_protected_paths:
        ignored_protected.update(_normalize(path))
    active: Set[str] = set()
    complete = True
    legacy_policy_unknown = False
    try:
        proc_entries = os.scandir("/proc")
    except OSError:
        return active, False, False
    with proc_entries:
        for entry in proc_entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in ignored:
                continue
            try:
                if entry.stat(follow_symlinks=False).st_uid != os.getuid():
                    continue
                with open(f"/proc/{pid}/environ", "rb") as stream:
                    values = stream.read().split(b"\0")
            except FileNotFoundError:
                continue
            except OSError as exc:
                if (
                    exc.errno not in {errno.ENOENT, errno.ESRCH}
                    and _process_may_use_appdata(pid)
                ):
                    complete = False
                continue
            protected, protected_count, protected_valid = (
                _protected_appdata_from_environ(values)
            )
            active.update(protected - ignored_protected)

            prefix = b"OMNIGIBSON_APPDATA_PATH="
            for value in values:
                if not value.startswith(prefix):
                    continue
                try:
                    path = value[len(prefix) :].decode("utf-8", "surrogateescape")
                except UnicodeDecodeError:
                    continue
                if path:
                    active.update(_normalize(path))
            if protected_count != 1 or not protected_valid:
                is_official_policy = _official_policy_cmdline(pid)
                if is_official_policy is None:
                    if _process_may_use_appdata(pid):
                        complete = False
                elif is_official_policy:
                    # Legacy policy processes outlive evaluator restarts but do not
                    # identify the appdata that the next evaluator will reuse.
                    legacy_policy_unknown = True
                elif protected_count:
                    complete = False
    return active, complete, legacy_policy_unknown


def active_appdata_snapshot(
    ignore_pids: Iterable[int] = (),
) -> Tuple[Set[str], bool]:
    """Return active appdata paths and whether destructive cleanup is safe."""

    active, complete, legacy_policy_unknown = _active_appdata_snapshot_details(
        ignore_pids
    )
    return active, complete and not legacy_policy_unknown


def active_appdata_paths(ignore_pids: Iterable[int] = ()) -> Set[str]:
    """Compatibility wrapper returning exact active appdata paths."""

    return active_appdata_snapshot(ignore_pids)[0]


def _preflight_active_appdata_snapshot(
    ignore_pids: Iterable[int],
    ignore_protected_paths: Iterable[str],
    *,
    reserve_for_pid: Optional[int],
) -> Tuple[Set[str], bool, bool, int]:
    """Take the launch-time process snapshot, retrying only a transient gap.

    A process can be between ``fork``/``exec`` while its ``/proc`` entries are
    being enumerated.  The normal safety policy remains fail-closed after the
    bounded retry window, but a successful first scan keeps the historical
    one-pass fast path with no added delay.  Retries are limited to evaluator
    launches (``reserve_for_pid``), where a false negative currently aborts the
    whole startup before Isaac is imported.
    """

    ignored = tuple(int(pid) for pid in ignore_pids)
    ignored_paths = tuple(str(path) for path in ignore_protected_paths)
    active, complete, legacy_unknown = _active_appdata_snapshot_details(
        ignored,
        ignored_paths,
    )
    attempts = 1
    if reserve_for_pid is None or complete:
        return active, complete, legacy_unknown, attempts

    # Keep this deliberately short: it covers fork/exec races without turning
    # a genuinely unreadable process into an unbounded launch wait.
    for _ in range(2):
        time.sleep(0.05)
        active, complete, legacy_unknown = _active_appdata_snapshot_details(
            ignored,
            ignored_paths,
        )
        attempts += 1
        if complete:
            break
    return active, complete, legacy_unknown, attempts


def _add_reserved_appdata(
    active: Set[str],
    records: Iterable[Dict[str, Any]],
    ignored_pids: Set[int],
    current_appdata: str,
) -> None:
    """Treat live reservations as activity, including post-exec env changes."""

    current_aliases = _normalize(current_appdata)
    for record in records:
        aliases = _normalize(str(record["appdata"]))
        # The current process (and a task-switch ancestor) may legitimately
        # reserve the target before this preflight. Ignore only that exact
        # target; reservations for its other appdata paths remain protected.
        if int(record["pid"]) in ignored_pids and aliases & current_aliases:
            continue
        active.update(aliases)


def directory_size_bytes(path: str, *, deadline_s: float = 8.0) -> int:
    """Physical bytes, counting a hardlinked inode once within this tree.

    A texture cache lives on the shared NFS volume. One stalled ``stat`` used
    to sit inside the exclusive storage lock and freeze every other lane.
    Past the deadline this returns the bytes counted so far; callers that
    only need a free-space decision must not wait on a full walk.
    """

    total = 0
    seen: Set[Tuple[int, int]] = set()
    deadline = time.monotonic() + max(0.0, deadline_s)
    for dirpath, _dirnames, filenames in os.walk(path):
        if time.monotonic() >= deadline:
            return total
        for name in filenames:
            if time.monotonic() >= deadline:
                return total
            try:
                stat = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
            except OSError:
                continue
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            total += stat.st_blocks * 512
    return total


def _free_bytes(path: str) -> int:
    return shutil.disk_usage(_storage_probe(path)).free


def _storage_probe(path: str) -> str:
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return probe


def _storage_device(path: str) -> int:
    return int(os.stat(_storage_probe(path)).st_dev)


def watched_storage_targets(appdata_path: str) -> list[Dict[str, Any]]:
    """Return one stable probe for each appdata/runtime-temp filesystem."""

    appdata = os.path.abspath(os.path.expanduser(appdata_path))
    runtime_dir = os.environ.get(RUNTIME_TMP_DIR_ENV, "").strip()
    runtime_path = os.path.abspath(runtime_dir or runtime_tmp_root())
    targets_by_device: Dict[int, Dict[str, Any]] = {}
    for kind, path in (("appdata", appdata), ("runtime_tmp", runtime_path)):
        probe = _storage_probe(path)
        device = int(os.stat(probe).st_dev)
        target = targets_by_device.get(device)
        if target is None:
            targets_by_device[device] = {
                "path": probe,
                "device": device,
                "kinds": [kind],
            }
        elif kind not in target["kinds"]:
            target["kinds"].append(kind)
    return list(targets_by_device.values())


class StorageWatchdog:
    """Terminate this process if a filesystem it writes crosses the hard floor."""

    def __init__(
        self,
        targets: Iterable[Dict[str, Any]],
        *,
        hard_free_gib: float,
        check_interval_s: Optional[float] = None,
        kill_grace_s: Optional[float] = None,
        sample_error_limit: Optional[int] = None,
        free_bytes_fn: Optional[Callable[[str], int]] = None,
        device_fn: Optional[Callable[[str], int]] = None,
        signal_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.owner_pid = os.getpid()
        self.owner_start_ticks = _proc_start_ticks(self.owner_pid)
        if self.owner_start_ticks is None:
            raise RuntimeError(
                f"cannot start storage watchdog for dead pid {self.owner_pid}"
            )
        self.owner_token = os.urandom(16).hex()
        self.hard_free_bytes = int(max(0.0, float(hard_free_gib)) * (1024**3))
        configured_interval = (
            _env_float(WATCHDOG_INTERVAL_ENV, 1.0)
            if check_interval_s is None
            else max(0.0, float(check_interval_s))
        )
        self.check_interval_s = max(0.05, configured_interval)
        self.kill_grace_s = (
            _env_float(WATCHDOG_KILL_GRACE_ENV, 30.0)
            if kill_grace_s is None
            else max(0.0, float(kill_grace_s))
        )
        if sample_error_limit is None:
            try:
                configured_error_limit = int(
                    os.environ.get(WATCHDOG_SAMPLE_ERROR_LIMIT_ENV, "3")
                )
            except ValueError:
                configured_error_limit = 3
            self.sample_error_limit = max(1, configured_error_limit)
        else:
            self.sample_error_limit = max(1, int(sample_error_limit))
        self._free_bytes_fn = free_bytes_fn or _free_bytes
        self._device_fn = device_fn or _storage_device
        self._signal_callback = signal_callback or self._send_signal
        self._log_callback = log_callback or self._log
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._triggered_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._targets: Dict[int, Dict[str, Any]] = {}
        self.last_free_bytes: Dict[int, int] = {}
        self.last_error: Optional[str] = None
        self.trigger_report: Optional[Dict[str, Any]] = None
        self.update_targets(targets, hard_free_gib=hard_free_gib)

    @staticmethod
    def _log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def _owner_is_current(self) -> bool:
        return (
            os.getpid() == self.owner_pid
            and _proc_start_ticks(self.owner_pid) == self.owner_start_ticks
        )

    def _send_signal(self, pid: int, signum: int, _report: Dict[str, Any]) -> None:
        if pid != os.getpid() or not self._owner_is_current():
            return
        os.kill(pid, signum)

    def update_targets(
        self,
        targets: Iterable[Dict[str, Any]],
        *,
        hard_free_gib: Optional[float] = None,
    ) -> None:
        if not self._owner_is_current():
            return
        normalized: Dict[int, Dict[str, Any]] = {}
        for raw in targets:
            path = os.path.abspath(os.path.expanduser(str(raw["path"])))
            device = int(raw["device"])
            actual_device = int(self._device_fn(path))
            if actual_device != device:
                raise RuntimeError(
                    f"storage target changed device before watchdog start: "
                    f"{path} expected={device} actual={actual_device}"
                )
            kinds = sorted({str(kind) for kind in raw.get("kinds", [])})
            normalized[device] = {
                "path": path,
                "device": device,
                "kinds": kinds,
            }
        if not normalized:
            raise ValueError("storage watchdog requires at least one target")
        with self._state_lock:
            self._targets.update(normalized)
            if hard_free_gib is not None:
                self.hard_free_bytes = max(
                    self.hard_free_bytes,
                    int(max(0.0, float(hard_free_gib)) * (1024**3)),
                )

    @property
    def triggered(self) -> bool:
        return self._triggered_event.is_set()

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def status(self) -> Dict[str, Any]:
        return {
            "pid": self.owner_pid,
            "start_ticks": self.owner_start_ticks,
            "token": self.owner_token,
            "hard_free_bytes": self.hard_free_bytes,
            "devices": sorted(
                int(target["device"]) for target in self._targets_snapshot()
            ),
            "running": self.running,
            "triggered": self.triggered,
        }

    def start(self) -> bool:
        # A forked child inherits this object but not its thread. Never touch an
        # inherited Event or Lock before rejecting the parent's identity.
        if not self._owner_is_current() or not self.hard_free_bytes:
            return False
        with self._state_lock:
            if self._triggered_event.is_set() or (
                self._thread is not None and self._thread.is_alive()
            ):
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"storage-watchdog-{self.owner_pid}",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, join_timeout_s: float = 2.0) -> None:
        if os.getpid() != self.owner_pid:
            return
        self._stop_event.set()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(0.0, float(join_timeout_s)))

    def _targets_snapshot(self) -> list[Dict[str, Any]]:
        with self._state_lock:
            return [dict(target) for target in self._targets.values()]

    def _record_error(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if message == self.last_error:
            return
        self.last_error = message
        self._log_callback(f"[storage-watchdog] sample failed: {message}")

    def _run(self) -> None:
        consecutive_sample_errors = 0
        while not self._stop_event.is_set() and self._owner_is_current():
            low_target: Optional[Dict[str, Any]] = None
            failed_target: Optional[Dict[str, Any]] = None
            sample_errors = []
            for target in self._targets_snapshot():
                try:
                    actual_device = int(self._device_fn(target["path"]))
                    if actual_device != int(target["device"]):
                        failed_target = dict(target)
                        failed_target["sample_error"] = (
                            f"storage target changed device: {target['path']} "
                            f"expected={target['device']} actual={actual_device}"
                        )
                        failed_target["sample_error_immediate"] = True
                        sample_errors.append(failed_target["sample_error"])
                        break
                    free_bytes = int(self._free_bytes_fn(target["path"]))
                    self.last_free_bytes[actual_device] = free_bytes
                    if free_bytes < self.hard_free_bytes:
                        low_target = dict(target)
                        low_target["free_bytes"] = free_bytes
                        break
                except Exception as exc:
                    message = f"{target['path']}: {type(exc).__name__}: {exc}"
                    sample_errors.append(message)
                    if failed_target is None:
                        failed_target = dict(target)
                        failed_target["sample_error"] = message
            if sample_errors:
                consecutive_sample_errors += 1
                self._record_error(RuntimeError("; ".join(sample_errors)))
            else:
                consecutive_sample_errors = 0
                self.last_error = None

            if low_target is not None:
                self._trigger(low_target)
                return
            if failed_target is not None and (
                failed_target.get("sample_error_immediate")
                or consecutive_sample_errors >= self.sample_error_limit
            ):
                failed_target["free_bytes"] = 0
                failed_target["sample_error_count"] = consecutive_sample_errors
                self._trigger(failed_target)
                return
            if self._stop_event.wait(self.check_interval_s):
                return

    def _trigger(self, low_target: Dict[str, Any]) -> None:
        if (
            self._stop_event.is_set()
            or self._triggered_event.is_set()
            or not self._owner_is_current()
        ):
            return
        self._triggered_event.set()
        report = {
            "pid": self.owner_pid,
            "start_ticks": self.owner_start_ticks,
            "token": self.owner_token,
            "path": low_target["path"],
            "device": int(low_target["device"]),
            "kinds": list(low_target.get("kinds", [])),
            "free_bytes": int(low_target["free_bytes"]),
            "hard_free_bytes": self.hard_free_bytes,
            "checked_at": time.time(),
        }
        if low_target.get("sample_error"):
            report["sample_error"] = str(low_target["sample_error"])
            report["sample_error_count"] = int(
                low_target.get("sample_error_count", 1)
            )
        self.trigger_report = report
        if report.get("sample_error"):
            self._log_callback(
                "[storage-watchdog] storage sampling is unreliable; "
                "terminating writer "
                f"pid={self.owner_pid} device={report['device']} "
                f"kinds={','.join(report['kinds']) or 'unknown'} "
                f"error={report['sample_error']}"
            )
        else:
            self._log_callback(
                "[storage-watchdog] hard floor crossed; terminating writer "
                f"pid={self.owner_pid} device={report['device']} "
                f"kinds={','.join(report['kinds']) or 'unknown'} "
                f"free={report['free_bytes'] / (1024**3):.1f}GiB "
                f"required={self.hard_free_bytes / (1024**3):.1f}GiB"
            )
        if self._stop_event.is_set() or not self._owner_is_current():
            return
        try:
            self._signal_callback(self.owner_pid, signal.SIGTERM, report)
        except Exception as exc:
            self._record_error(exc)
        if self._stop_event.wait(self.kill_grace_s):
            return
        if not self._owner_is_current():
            return
        self._log_callback(
            "[storage-watchdog] writer survived SIGTERM grace; escalating "
            f"pid={self.owner_pid} signal=SIGKILL"
        )
        try:
            self._signal_callback(self.owner_pid, signal.SIGKILL, report)
        except Exception as exc:
            self._record_error(exc)


def _reset_storage_watchdog_after_fork() -> None:
    global _process_storage_watchdog, _watchdog_cleanup_registered
    global _watchdog_registry_lock

    _process_storage_watchdog = None
    _watchdog_cleanup_registered = False
    _watchdog_registry_lock = threading.Lock()


def _stop_process_storage_watchdog() -> None:
    global _process_storage_watchdog

    if os.getpid() != getattr(_process_storage_watchdog, "owner_pid", os.getpid()):
        return
    with _watchdog_registry_lock:
        watchdog = _process_storage_watchdog
        _process_storage_watchdog = None
    if watchdog is not None:
        watchdog.stop()


def start_storage_watchdog(
    targets: Iterable[Dict[str, Any]],
    *,
    hard_free_gib: float,
) -> StorageWatchdog:
    """Start or update the single storage watchdog owned by this process."""

    global _process_storage_watchdog, _watchdog_cleanup_registered

    with _watchdog_registry_lock:
        watchdog = _process_storage_watchdog
        if watchdog is not None and watchdog._owner_is_current():
            if watchdog.triggered:
                raise RuntimeError(
                    "storage watchdog already triggered; refusing to reuse it"
                )
            watchdog.update_targets(targets, hard_free_gib=hard_free_gib)
            if not watchdog.running and not watchdog.start():
                raise RuntimeError("storage watchdog could not be restarted")
            return watchdog
        watchdog = StorageWatchdog(targets, hard_free_gib=hard_free_gib)
        _process_storage_watchdog = watchdog
        try:
            if not _watchdog_cleanup_registered:
                atexit.register(_stop_process_storage_watchdog)
                _watchdog_cleanup_registered = True
            if not watchdog.start():
                raise RuntimeError("storage watchdog did not start")
        except BaseException:
            _process_storage_watchdog = None
            raise
        return watchdog


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_storage_watchdog_after_fork)


def _managed_roots(appdata_path: str) -> list[str]:
    configured = os.environ.get(ROOTS_ENV, "").strip()
    roots = configured.split(os.pathsep) if configured else ["/var/tmp"]
    parent = os.path.dirname(os.path.abspath(appdata_path))
    if parent not in roots:
        roots.append(parent)
    try:
        appdata_device = os.stat(appdata_path).st_dev
    except OSError:
        appdata_device = None

    result = []
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(os.path.expanduser(root))
        try:
            if appdata_device is not None and os.stat(root).st_dev != appdata_device:
                continue
        except OSError:
            continue
        result.append(root)
    return result


def _managed_appdata_dirs(roots: Iterable[str]) -> Iterator[str]:
    user = os.environ.get("USER") or str(os.getuid())
    prefix = f"og_appdata_{user}_"
    seen: Set[str] = set()
    for root in roots:
        try:
            entries = os.scandir(root)
        except OSError:
            continue
        with entries:
            for entry in entries:
                if not entry.name.startswith(prefix):
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_uid != os.getuid():
                        continue
                except OSError:
                    continue
                path = os.path.abspath(entry.path)
                if path not in seen:
                    seen.add(path)
                    yield path
                # Optional task-switch isolation stores retry appdata below the
                # stable per-port appdata. Enumerate only that fixed hierarchy;
                # an unrestricted recursive walk would broaden deletion scope.
                retry_root = os.path.join(path, "task_switch_retries")
                try:
                    targets = list(os.scandir(retry_root))
                except OSError:
                    continue
                for target in targets:
                    if not target.is_dir(follow_symlinks=False):
                        continue
                    try:
                        supervisors = list(os.scandir(target.path))
                    except OSError:
                        continue
                    for supervisor in supervisors:
                        if (
                            not supervisor.name.startswith("supervisor_")
                            or not supervisor.is_dir(follow_symlinks=False)
                        ):
                            continue
                        try:
                            attempts = list(os.scandir(supervisor.path))
                        except OSError:
                            continue
                        for attempt in attempts:
                            if (
                                not attempt.name.startswith("attempt_")
                                or not attempt.is_dir(follow_symlinks=False)
                            ):
                                continue
                            try:
                                if attempt.stat(follow_symlinks=False).st_uid != os.getuid():
                                    continue
                            except OSError:
                                continue
                            nested = os.path.abspath(attempt.path)
                            if nested not in seen:
                                seen.add(nested)
                                yield nested


def _is_protected_main(appdata: str) -> bool:
    return any(
        re.search(r"_interface_15050(?:_|$)", os.path.basename(path)) is not None
        for path in _normalize(appdata)
    )


def _cache_path(appdata: str) -> str:
    return os.path.join(os.path.abspath(appdata), "global", "cache", "texturecache")


def _is_retry_appdata(appdata: str) -> bool:
    attempt = os.path.abspath(appdata)
    supervisor = os.path.dirname(attempt)
    target = os.path.dirname(supervisor)
    retry_root = os.path.dirname(target)
    return (
        os.path.basename(attempt).startswith("attempt_")
        and os.path.basename(supervisor).startswith("supervisor_")
        and os.path.basename(retry_root) == "task_switch_retries"
    )


def _remove_cache(appdata: str, *, dry_run: bool) -> int:
    cache = _cache_path(appdata)
    if not os.path.isdir(cache) or os.path.islink(cache):
        return 0
    size = directory_size_bytes(cache)
    if not dry_run:
        shutil.rmtree(cache)
    return size


def preflight_storage(
    appdata_path: str,
    *,
    soft_free_gib: Optional[float] = None,
    hard_free_gib: Optional[float] = None,
    cache_max_gib: Optional[float] = None,
    dry_run: bool = False,
    ignore_pids: Iterable[int] = (),
    reserve_for_pid: Optional[int] = None,
) -> Dict[str, Any]:
    """Bound the current cache, reclaim inactive caches, and enforce reserve."""

    appdata = os.path.abspath(os.path.expanduser(appdata_path))
    if not appdata or appdata == os.path.sep:
        raise ValueError(f"unsafe appdata path: {appdata_path!r}")
    os.makedirs(appdata, exist_ok=True)
    soft = _env_float(SOFT_FREE_ENV, 200.0) if soft_free_gib is None else max(0.0, soft_free_gib)
    hard = _env_float(HARD_FREE_ENV, 64.0) if hard_free_gib is None else max(0.0, hard_free_gib)
    maximum = _env_float(CACHE_MAX_ENV, 16.0) if cache_max_gib is None else max(0.0, cache_max_gib)
    if soft and hard > soft:
        raise ValueError(f"hard reserve {hard} GiB exceeds soft target {soft} GiB")
    reserve_owner_is_current = (
        reserve_for_pid is not None and int(reserve_for_pid) == os.getpid()
    )
    report: Dict[str, Any] = {
        "appdata": appdata,
        "removed": [],
        "freed_bytes": 0,
        "free_before": _free_bytes(appdata),
        "free_after": 0,
        "soft_free_gib": soft,
        "hard_free_gib": hard,
        "cache_max_gib": maximum,
    }

    try:
        tmp_report = prune_stale_runtime_dirs(dry_run=dry_run)
        report["runtime_tmp_removed"] = tmp_report["removed"]
        report["runtime_tmp_freed_bytes"] = tmp_report["freed_bytes"]
    except (OSError, ValueError) as exc:
        report["runtime_tmp_error"] = str(exc)

    ignored = _ancestor_pids(os.getpid()) | {int(pid) for pid in ignore_pids}
    # The process scan walks /proc and can block on a stale NFS cwd.  Doing
    # that inside the exclusive lock froze every other lane's launch behind
    # one disk-sleep evaluator (15441, rpc_wait_bit_killable, 49 minutes).
    (
        active,
        process_scan_complete,
        legacy_policy_unknown,
        scan_attempts,
    ) = _preflight_active_appdata_snapshot(
        ignored,
        (appdata,),
        reserve_for_pid=reserve_for_pid,
    )
    with storage_lock():
        active_scan_complete = process_scan_complete and not legacy_policy_unknown
        reservation_records = _reservation_records()
        _add_reserved_appdata(active, reservation_records, ignored, appdata)
        report["active_scan_complete"] = active_scan_complete
        report["active_scan_attempts"] = scan_attempts
        report["legacy_policy_appdata_unknown"] = legacy_policy_unknown
        appdata_aliases = _normalize(appdata)
        current_active_elsewhere = bool(appdata_aliases & active)
        if current_active_elsewhere:
            raise RuntimeError(f"appdata is already used by another process: {appdata}")
        current_cache = _cache_path(appdata)
        current_size = directory_size_bytes(current_cache) if os.path.isdir(current_cache) else 0
        report["current_cache_bytes"] = current_size
        if maximum and current_size > maximum * (1024**3) and not _is_protected_main(appdata):
            if not active_scan_complete:
                raise RuntimeError(
                    "refusing destructive cache cleanup because /proc activity scan was incomplete"
                )
            # A process can start after the first /proc snapshot. Re-read while
            # holding the maintenance lock immediately before destructive work.
            refreshed, refreshed_process_complete, refreshed_legacy_unknown = (
                _active_appdata_snapshot_details(ignored, (appdata,))
            )
            refreshed_complete = (
                refreshed_process_complete and not refreshed_legacy_unknown
            )
            refreshed_records = _reservation_records()
            _add_reserved_appdata(refreshed, refreshed_records, ignored, appdata)
            if not refreshed_complete:
                raise RuntimeError(
                    "refusing destructive cache cleanup because /proc activity rescan was incomplete"
                )
            if appdata_aliases & refreshed:
                raise RuntimeError(
                    f"refusing to clear cache used by another process: {appdata}"
                )
            size = _remove_cache(appdata, dry_run=dry_run)
            report["removed"].append({"path": current_cache, "bytes": size, "reason": "over_limit"})
            report["freed_bytes"] += size

        free_now = _free_bytes(appdata)
        if (
            soft
            and free_now < soft * (1024**3)
            and _env_enabled(AUTO_CLEAN_ENV)
            and active_scan_complete
        ):
            candidates = []
            appdata_device = os.stat(appdata).st_dev
            for candidate in _managed_appdata_dirs(_managed_roots(appdata)):
                if (_normalize(candidate) & appdata_aliases) or _is_protected_main(candidate):
                    continue
                cache = _cache_path(candidate)
                retry_appdata = _is_retry_appdata(candidate)
                reclaim_path = candidate if retry_appdata else cache
                if not os.path.isdir(reclaim_path) or (_normalize(candidate) & active):
                    continue
                try:
                    if (
                        os.stat(candidate, follow_symlinks=False).st_dev != appdata_device
                        or os.stat(reclaim_path, follow_symlinks=False).st_dev != appdata_device
                    ):
                        continue
                except OSError:
                    continue
                candidates.append(
                    (directory_size_bytes(reclaim_path), candidate, retry_appdata)
                )
            for _size, candidate, retry_appdata in sorted(candidates, reverse=True):
                if _free_bytes(appdata) >= soft * (1024**3):
                    break
                # Refresh immediately before deletion to close the process-start race.
                refreshed, refreshed_process_complete, refreshed_legacy_unknown = (
                    _active_appdata_snapshot_details(ignored, (appdata,))
                )
                refreshed_complete = (
                    refreshed_process_complete and not refreshed_legacy_unknown
                )
                refreshed_records = _reservation_records()
                _add_reserved_appdata(refreshed, refreshed_records, ignored, appdata)
                if not refreshed_complete or (_normalize(candidate) & refreshed):
                    continue
                if retry_appdata:
                    size = directory_size_bytes(candidate)
                    if not dry_run:
                        shutil.rmtree(candidate)
                    removed_path = candidate
                    reason = "inactive_retry_appdata"
                else:
                    size = _remove_cache(candidate, dry_run=dry_run)
                    removed_path = _cache_path(candidate)
                    reason = "low_space"
                report["removed"].append(
                    {"path": removed_path, "bytes": size, "reason": reason}
                )
                report["freed_bytes"] += size
        elif soft and free_now < soft * (1024**3) and not active_scan_complete:
            report["cleanup_skipped"] = "incomplete_active_process_scan"

        report["free_after"] = _free_bytes(appdata)

        maximum_bytes = int(maximum * (1024**3))
        current_size_after = (
            directory_size_bytes(current_cache) if os.path.isdir(current_cache) else 0
        )
        report["current_cache_bytes_after"] = current_size_after
        requested_reservation = (
            max(0, maximum_bytes - current_size_after) if maximum else 0
        )
        device = os.stat(appdata).st_dev
        reserved_by_appdata: Dict[str, int] = {}
        for record in reservation_records:
            if record["device"] != device:
                continue
            key = record["appdata"]
            reserved_by_appdata[key] = max(
                reserved_by_appdata.get(key, 0), record["reserved_bytes"]
            )
        current_real = os.path.realpath(appdata)
        reserved_by_appdata.pop(current_real, None)

        # Include older processes that started before reservation support.
        for active_path in {os.path.realpath(path) for path in active}:
            if active_path == current_real or not os.path.isdir(active_path):
                continue
            try:
                if os.stat(active_path).st_dev != device:
                    continue
                active_size = directory_size_bytes(_cache_path(active_path))
            except OSError:
                continue
            reserved_by_appdata[active_path] = max(
                reserved_by_appdata.get(active_path, 0),
                max(0, maximum_bytes - active_size),
            )

        other_reserved = sum(reserved_by_appdata.values())
        projected_free = report["free_after"] - other_reserved - requested_reservation
        report["other_reserved_bytes"] = other_reserved
        report["requested_reservation_bytes"] = requested_reservation
        report["projected_free_bytes"] = projected_free
        if reserve_for_pid is not None and not process_scan_complete:
            raise RuntimeError(
                "refusing storage reservation because /proc activity scan was incomplete"
            )
        if hard and projected_free < hard * (1024**3):
            raise OSError(
                errno.ENOSPC,
                "storage reservations would cross the hard reserve: "
                f"projected={projected_free / (1024**3):.1f} GiB, "
                f"required={hard:.1f} GiB",
                appdata,
            )
        if reserve_owner_is_current and hard and not dry_run:
            watched = watched_storage_targets(appdata)
            hard_bytes = int(hard * (1024**3))
            for target in watched:
                target["free_bytes"] = _free_bytes(target["path"])
                if target["free_bytes"] < hard_bytes:
                    kinds = ",".join(target["kinds"]) or "unknown"
                    raise OSError(
                        errno.ENOSPC,
                        "storage watchdog target is already below the hard reserve: "
                        f"device={target['device']} kinds={kinds} "
                        f"free={target['free_bytes'] / (1024**3):.1f} GiB, "
                        f"required={hard:.1f} GiB",
                        target["path"],
                    )
            report["watched_storage"] = watched
        if reserve_for_pid is not None and not dry_run:
            report["reservation"] = _write_reservation(
                appdata, requested_reservation, int(reserve_for_pid)
            )

    if not dry_run and hard and report["free_after"] < hard * (1024**3):
        raise OSError(
            errno.ENOSPC,
            "storage preflight could not restore the hard reserve: "
            f"free={report['free_after'] / (1024**3):.1f} GiB, required={hard:.1f} GiB",
            appdata,
        )
    if reserve_owner_is_current and hard and not dry_run:
        try:
            watchdog = start_storage_watchdog(
                report["watched_storage"],
                hard_free_gib=hard,
            )
        except BaseException:
            marker = report.get("reservation")
            if marker:
                _release_owned_reservation(marker)
            raise
        report["watchdog"] = watchdog.status()
    return report


def lane_appdata_free(appdata_path: str) -> bool:
    """True when this lane's appdata directory exists and has free space.

    The full preflight walks every process on the machine to decide which
    appdata is still in use.  That walk stalls in D state once a campaign
    has hundreds of Claude processes, and every scene change waits on it.
    A lane that already owns its appdata directory does not need that scan
    to start the next evaluator.
    """
    appdata = os.path.abspath(os.path.expanduser(appdata_path))
    if not appdata or appdata == os.path.sep or not os.path.isdir(appdata):
        return False
    try:
        free = _free_bytes(appdata)
    except OSError:
        return False
    hard = _env_float(HARD_FREE_ENV, 64.0)
    return free >= hard * (1024**3)


def _main() -> int:
    parser = argparse.ArgumentParser(description="BEHAVIOR runtime storage preflight")
    parser.add_argument("--appdata", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-pid", action="append", type=int, default=[])
    parser.add_argument("--reserve-for-pid", type=int, default=None)
    parser.add_argument(
        "--lane-check",
        action="store_true",
        help="Check only this appdata directory. Do not scan /proc.",
    )
    args = parser.parse_args()
    if args.lane_check:
        return 0 if lane_appdata_free(args.appdata) else 75
    try:
        report = preflight_storage(
            args.appdata,
            dry_run=args.dry_run,
            ignore_pids=args.ignore_pid,
            reserve_for_pid=args.reserve_for_pid,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[storage-preflight] ERROR: {exc}", flush=True)
        return 75
    print(
        "[storage-preflight] "
        f"free={report['free_before'] / (1024**3):.1f}"
        f"->{report['free_after'] / (1024**3):.1f} GiB "
        f"removed={len(report['removed'])} "
        f"freed={report['freed_bytes'] / (1024**3):.1f} GiB "
        f"tmp_removed={report.get('runtime_tmp_removed', 0)}",
        flush=True,
    )
    if report["removed"]:
        print(json.dumps(report["removed"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
