"""Read-only liveness checks of the workspace's real evaluator connection."""
import hashlib
import json
from pathlib import Path
import socket
import time


def workspace_status(workspace, timeout=2):
    address = '\0arx-x5-' + hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(address)
        connection.sendall(b'{"method":"status"}\n')
        with connection.makefile('rb') as stream:
            value = json.loads(stream.readline(1024*1024))
    if not value.get('ok') or not isinstance(value.get('evaluator'), dict):
        raise RuntimeError('Invalid workspace health response')
    return value


class ConnectionHealth:
    def __init__(self):
        self.unavailable_since = None

    def check(self, snapshot, *, now=None, active=False):
        """Idle inference still polls get_action; expensive planning gets more time.

        Observation age alone is not a failure: zero-tick reads and inference
        intentionally freeze physics while the evaluator heartbeat continues.
        """
        now = time.monotonic() if now is None else now
        if not active:
            self.unavailable_since = None
            return None
        if snapshot is None:
            if self.unavailable_since is None:
                self.unavailable_since = now
            if now-self.unavailable_since >= 120:
                return 'Workspace health socket unresponsive for 120 seconds'
            return None
        self.unavailable_since = None
        evaluator = snapshot['evaluator']
        age = evaluator.get('heartbeat_age_s')
        busy = bool(snapshot.get('scheduler', {}).get('current_tool'))
        limit = 900 if busy else 180
        if age is not None and age > limit:
            return f'Evaluator heartbeat stopped for {age:.1f}s (limit {limit}s, planning={busy})'
        return None
