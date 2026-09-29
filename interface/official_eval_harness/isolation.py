"""One port, one owner; clean up by process identity, never by a shared GPU.

PID files and process groups alone are insufficient: the leader may have exited,
children may call setsid(), and a stored PID may have been reused. Match exact
port metadata, include descendants, and validate PID start time before signalling.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .catalog import (
    OFFICIAL_PORT_TASK,
    mapped_task_for_port,
    official_policy_port,
    idle_gate_port,
    official_port_for_task,
    resolve_task,
)

LOCK_ROOT = Path('/tmp')
OWNER_PORT = 'BEHAVIOR_EVAL_OWNER_PORT'
OWNER_RUN = 'BEHAVIOR_EVAL_OWNER_RUN'
OWNER_ROLE = 'BEHAVIOR_EVAL_OWNER_ROLE'
SID_RE = re.compile(r'^t(\d{2})p(\d+)i\d+-[A-Za-z0-9_.-]+$')
ENV_KEYS = {OWNER_PORT, OWNER_RUN, OWNER_ROLE, 'BEHAVIOR_PORT',
            'BEHAVIOR_EVAL_TEST_PORT', 'BEHAVIOR_SESSION_ID', 'BEHAVIOR_BASE_URL'}


def require_official_port(port: int) -> int:
    if mapped_task_for_port(int(port)) is None:
        raise ValueError(f'只能操作官方口 15010–15045 或 15060–15069: {port}')
    return int(port)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path: Path) -> dict:
    try:
        body = json.loads(path.read_text(encoding='utf-8'))
        return body if isinstance(body, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start: str
    state: str
    argv: tuple[str, ...]
    env: dict[str, str]


def read_process(pid: int) -> Process | None:
    root = Path(f'/proc/{int(pid)}')
    try:
        if root.stat().st_uid != os.getuid():
            return None
        stat = (root / 'stat').read_text()
        fields = stat[stat.rfind(') ') + 2:].split()
        argv = tuple(x.decode('utf-8', 'replace') for x in (root / 'cmdline').read_bytes().split(b'\0') if x)
        env = {}
        for item in (root / 'environ').read_bytes().split(b'\0'):
            if b'=' in item:
                key, value = item.split(b'=', 1)
                name = key.decode('utf-8', 'replace')
                if name in ENV_KEYS:
                    env[name] = value.decode('utf-8', 'replace')
        return Process(int(pid), int(fields[1]), fields[19], fields[0], argv, env)
    except (OSError, ValueError, IndexError):
        return None


def process_running(pid: int, start: str | None = None) -> bool:
    p = read_process(pid)
    return p is not None and p.state not in {'Z', 'X'} and (start is None or p.start == start)


def snapshot() -> dict[int, Process]:
    found = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            p = read_process(int(entry.name))
            if p and p.state not in {'Z', 'X'}:
                found[p.pid] = p
    return found


def option(argv: tuple[str, ...], key: str) -> str:
    values = []
    for i, word in enumerate(argv):
        if word == key and i + 1 < len(argv):
            values.append(argv[i + 1])
        elif word.startswith(key + '='):
            values.append(word[len(key) + 1:])
    # Ambiguous legacy commands must not be guessed at during cleanup.
    return values[0] if len(set(values)) == 1 else ''


def module(p: Process, name: str) -> bool:
    return any(a == '-m' and b == name for a, b in zip(p.argv, p.argv[1:]))


def ownership(p: Process) -> tuple[set[int], str]:
    ports = set()
    for key in (OWNER_PORT, 'BEHAVIOR_PORT', 'BEHAVIOR_EVAL_TEST_PORT'):
        if p.env.get(key, '').isdigit():
            ports.add(int(p.env[key]))
    sid = SID_RE.fullmatch(p.env.get('BEHAVIOR_SESSION_ID', ''))
    if sid:
        ports.add(int(sid[2]))
    base = urlparse(p.env.get('BEHAVIOR_BASE_URL', ''))
    try:
        if base.port:
            ports.add(base.port)
    except ValueError:
        pass
    role = ''
    # Structured argv, not substring matching against shell commands / prompts.
    if module(p, 'official_eval_harness.run') or module(p, 'official_eval_harness'):
        if '--yes' not in p.argv or '--dry-run' in p.argv or option(p.argv, '--stop-run'):
            return set(), ''  # read-only previews / stop clients are not old owners
        role = 'harness'
        # Harness config comes from argv, not inherited agent environment.
        # In particular, --task task07 (without --port) still owns only 15067.
        ports = set()
        raw = option(p.argv, '--port')
        if raw.isdigit():
            ports.add(int(raw))
        elif option(p.argv, '--task'):
            try:
                task = resolve_task(option(p.argv, '--task'))
                mapped_port = official_port_for_task(task.index)
                if mapped_port is not None:
                    ports.add(mapped_port)
            except ValueError:
                pass
    elif module(p, 'official_idle_step_gate') or module(p, 'official_idle_step_gate.proxy'):
        role = 'stack'
        try:
            port = urlparse(option(p.argv, '--backend-http-url')).port
            if port:
                ports.add(port)
        except ValueError:
            pass
    elif module(p, 'behavior_interface_eval_test.official_policy_interface'):
        role = 'stack'
        raw = option(p.argv, '--http-port')
        if raw.isdigit():
            ports.add(int(raw))
    elif module(p, 'behavior_interface_eval_test.official_evaluator_entrypoint') or module(p, 'omnigibson.eval.eval'):
        role = 'stack'
        raw = option(p.argv, '--port')
        if raw.isdigit():
            for http_port in OFFICIAL_PORT_TASK:
                if int(raw) in {official_policy_port(http_port), idle_gate_port(http_port)}:
                    ports.add(http_port)
    elif p.env.get('BEHAVIOR_SESSION_ID') and p.env.get('BEHAVIOR_PORT'):
        role = 'session'
    elif any(Path(arg).name in {'launch_official_policy_interface.sh', 'launch_official_evaluator_v391.sh'} for arg in p.argv):
        role = 'stack'
    elif p.env.get('BEHAVIOR_EVAL_TEST_PORT') and any(
        arg.startswith(('from multiprocessing.spawn ', 'from multiprocessing.forkserver ', 'from multiprocessing.resource_tracker ')) for arg in p.argv
    ):
        role = 'stack'
    if role != 'harness' and p.env.get(OWNER_ROLE) in {'session', 'stack', 'guardian'}:
        role = p.env[OWNER_ROLE]
    return ports, role


def select_processes(procs: dict[int, Process], port: int, roles: Iterable[str], *, run_id: str | None = None) -> list[Process]:
    require_official_port(port)
    excluded = {1}
    current = os.getpid()
    while current > 1 and current not in excluded:
        excluded.add(current)
        current = procs[current].ppid if current in procs else 1
    roles = set(roles)
    selected = {}
    for p in procs.values():
        ports, role = ownership(p)
        if p.pid in excluded or role not in roles or port not in ports:
            continue
        if ports != {port}:
            raise RuntimeError(f'pid={p.pid} 的端口归属冲突 {sorted(ports)}，拒绝清理/启动')
        if run_id is not None and p.env.get(OWNER_RUN) != run_id:
            continue
        selected[p.pid] = p
    # Descendants that lost their environment are still owned, but a child
    # explicitly assigned to a different port aborts cleanup, never crosses it.
    while True:
        added = False
        for p in procs.values():
            if p.pid in selected or p.pid in excluded or p.ppid not in selected:
                continue
            ports, role = ownership(p)
            if role == 'guardian':
                continue
            if ports and ports != {port}:
                raise RuntimeError(f'子进程 pid={p.pid} 属于其它口 {sorted(ports)}，拒绝清理/启动')
            selected[p.pid] = p
            added = True
        if not added:
            return list(selected.values())


def signal_process(p: Process, sig: int) -> None:
    # pidfd prevents a PID-reuse race between validation and kill.  The fallback
    # still verifies starttime; no PID loaded from an old file is blindly killed.
    fd = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            fd = os.pidfd_open(p.pid)
        if not process_running(p.pid, p.start):
            return
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(p.pid, sig)
    except ProcessLookupError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def cleanup_port(port: int, *, roles: Iterable[str] = ('harness', 'session', 'stack'), run_id: str | None = None, grace_s: float = 8.0) -> list[int]:
    """Re-scan until no live target remains, including orphaned setsid children.

    Zombies do not execute actions and are left for their parent/init to reap.
    Cleanup failure is fatal: a new agent must never be started over survivors.
    """
    seen: dict[tuple[int, str], Process] = {}
    term_sent = set()
    started = time.monotonic()
    deadline = started + grace_s + 5
    empty = 0
    while True:
        for p in select_processes(snapshot(), port, roles, run_id=run_id):
            seen[p.pid, p.start] = p
        alive = [p for p in seen.values() if process_running(p.pid, p.start)]
        if not alive:
            empty += 1
            if empty >= 2:
                return sorted({p.pid for p in seen.values()})
        else:
            empty = 0
        for p in alive:
            key = (p.pid, p.start)
            if time.monotonic() - started >= grace_s:
                signal_process(p, signal.SIGKILL)
            elif key not in term_sent:
                signal_process(p, signal.SIGTERM)
                term_sent.add(key)
        # Reap only our children; never wait on an unrelated PID.
        for p in seen.values():
            try:
                os.waitpid(p.pid, os.WNOHANG)
            except (OSError, ChildProcessError):
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f'端口 {port} 残留进程未退出: {[p.pid for p in alive]}')
        time.sleep(0.1)


class PortBusyError(RuntimeError):
    pass


class PortLease:
    def __init__(self, port: int, run_dir: Path | None = None):
        self.port = require_official_port(port)
        self.run_dir = run_dir
        self.stream = None

    @property
    def path(self) -> Path:
        return LOCK_ROOT / f'behavior_eval_harness_port_p{self.port}.lock'

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+', encoding='utf-8')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read(4096)
            stream.close()
            raise PortBusyError(f'端口 {self.port} 已有评测 owner，拒绝第二条；owner={owner}') from exc
        self.stream = stream
        self.update()
        return self

    def update(self, **extra):
        p = read_process(os.getpid())
        body = dict(pid=os.getpid(), start=p.start if p else '', port=self.port,
                    run_dir=str(self.run_dir or ''), **extra)
        self.stream.seek(0)
        self.stream.truncate()
        json.dump(body, self.stream, ensure_ascii=False)
        self.stream.flush()

    def __exit__(self, *args):
        # Never unlink a flock file, and do not unlock the shared descriptor:
        # the guardian inherits it and retains ownership through emergency cleanup.
        self.stream.close()
        self.stream = None


def start_guardian(lease: PortLease, run_id: str, log_path: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if not k.startswith('BEHAVIOR_')}
    env.update({OWNER_PORT: str(lease.port), OWNER_RUN: run_id, OWNER_ROLE: 'guardian'})
    with log_path.open('ab') as log:
        return subprocess.Popen(
            [sys.executable, '-m', 'official_eval_harness.isolation',
             '--watch', str(lease.port), '--run-id', run_id],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            env=env,
            pass_fds=(lease.stream.fileno(),), start_new_session=True,
        )


def finish_guardian(proc: subprocess.Popen) -> None:
    proc.stdin.close()  # EOF also arrives if the harness is SIGKILLed.
    proc.wait(timeout=45)
    if proc.returncode:
        raise RuntimeError('端口 guardian 清理失败，检查 guardian.log')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    sys.stdin.buffer.read()
    cleanup_port(args.watch, roles=('session', 'stack'), run_id=args.run_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
