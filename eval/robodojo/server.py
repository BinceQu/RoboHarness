"""Official evaluator transport adapter; all robot code belongs to its workspace."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--robodojo', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    temp = root/'.runtime/tmp'
    temp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(temp)
    tempfile.tempdir = str(temp)
    # XPolicyLab is the official evaluator protocol implementation. Its
    # transport remains evaluator-side; no agent tool imports this repository.
    sys.path.extend([str(args.robodojo.resolve()), str(args.robodojo.resolve()/'XPolicyLab')])
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    import cv2
    # OpenCV otherwise starts a worker for every available host CPU (168 on
    # 12023), each with private allocator arenas, in every policy process.
    cv2.setNumThreads(2)
    from tools.util.runtime import Runtime
    from lifecycle import BenchmarkControl
    class WorkspaceRuntime(BenchmarkControl, Runtime):
        pass
    runtime = WorkspaceRuntime(root)
    runtime.start()
    server = PolicyServer(runtime, PolicyServerConfig(host='127.0.0.1', port=args.port))
    print(json.dumps({'workspace':str(root), 'policy_port':args.port, 'pid':os.getpid()}), flush=True)
    try:
        asyncio.run(server.serve_forever())
    finally:
        runtime.close()

if __name__ == '__main__':
    main()
