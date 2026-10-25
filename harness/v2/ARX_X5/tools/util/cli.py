"""Plain JSON CLI into the local workspace's persistent action worker.

The worker preserves RGB-D/track/plan state between separate Bash processes.
It imports the exact visible tools/NAME.py algorithm. Only its yielded joint
commands cross the evaluator boundary. There is no remote macro-tool service.
"""

import argparse
import json
import socket
import sys
from pathlib import Path
from tools.util.settings import settings
from tools.util.receipt import without_ok


def request(payload, timeout=3660):
    root = Path(settings()["workspace"])
    import hashlib

    address = "\0arx-x5-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(address)
        connection.sendall(json.dumps(payload, allow_nan=False).encode() + b"\n")
        stream = connection.makefile("rb")
        line = stream.readline(32 * 1024 * 1024)
        if not line.endswith(b"\n"):
            raise RuntimeError("Incomplete workspace worker response")
    result = json.loads(line)
    return without_ok(result) if payload.get("method") == "call" else result


def run_cli(name, function=None):
    parser = argparse.ArgumentParser(
        description=(function.__doc__ if function else name)
    )
    parser.add_argument(
        "arguments", nargs="?", default="{}", help="JSON object, or - for stdin"
    )
    parser.add_argument(
        "--schema", action="store_true", help="Show this Python function signature"
    )
    args = parser.parse_args()
    if args.schema:
        import inspect

        print(
            json.dumps(
                {
                    "tool": name,
                    "signature": str(inspect.signature(function)),
                    "description": function.__doc__,
                }
            )
        )
        return
    arguments = json.loads(
        sys.stdin.read() if args.arguments == "-" else args.arguments
    )
    if not isinstance(arguments, dict):
        parser.error("arguments must be a JSON object")
    result = request({"method": "call", "tool": name, "arguments": arguments})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result.get("error"):
        raise SystemExit(1)
