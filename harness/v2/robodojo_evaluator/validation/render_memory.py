"""Measure real software-renderer retention across short-lived request threads.

Run with ARX_X5/.venv/bin/python. --source-file can select an archived render.py
to reproduce the pre-fix behavior with --mode fresh. No simulator is started.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

from tools.util import render


def memory(label):
    values = {
        key: int(value.split()[0])
        for key, value in (
            line.split(":", 1)
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines()[1:]
        )
    }
    return {"label": label, "pss_kib": values["Pss"],
            "swap_pss_kib": values["SwapPss"],
            "threads": len(list(Path("/proc/self/task").iterdir()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--mode", choices=("fresh", "persistent"), default="persistent")
    parser.add_argument("--source-file", type=Path)
    args = parser.parse_args()
    module = render
    if args.source_file:
        spec = importlib.util.spec_from_file_location("tools.util.render_audit", args.source_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    root = Path(__file__).resolve().parents[2] / "ARX_X5/robot_config"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_path = args.output.with_suffix(".png")
    rows = [memory("before")]
    hashes, errors = [], []
    persistent = module.BodyViewRenderer(root) if args.mode == "persistent" else None

    def request(index):
        try:
            # Exercise changed joints, then identical pixels after resetting.
            positions = {"right_joint2": 0.2} if index % 2 else {}
            if persistent:
                persistent.save(positions, image_path, caption="Memory regression", sequence=0)
            else:
                renderer = module.RobotRenderer(root)
                try:
                    renderer.three_views(positions, caption="Memory regression", sequence=0).save(image_path)
                finally:
                    renderer.close()
            hashes.append(hashlib.sha256(image_path.read_bytes()).hexdigest())
        except BaseException as exc:
            errors.append(repr(exc))

    try:
        for index in range(args.iterations):
            worker = threading.Thread(target=request, args=(index,))
            worker.start()
            worker.join()
            if errors:
                raise RuntimeError(errors)
            rows.append(memory(f"after {index + 1}"))
            if index < 4 or (index + 1) % 8 == 0:
                print(json.dumps(rows[-1]), flush=True)
    finally:
        if persistent:
            persistent.close()
    gc.collect()
    rows.append(memory("closed"))
    # Deliberately no malloc_trim: the steady-state check measures normal use.
    report = {"mode": args.mode, "iterations": args.iterations,
              "memory": rows, "image_hashes": hashes}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
