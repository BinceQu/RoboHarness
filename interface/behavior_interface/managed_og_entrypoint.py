"""Run an OmniGibson script with managed temp and storage safeguards."""

from __future__ import annotations

import os
import re
import runpy
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _path_component(value: object, fallback: str) -> str:
    component = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "")).strip("._")
    return component or fallback


def _default_appdata_path() -> str:
    configured_roots = os.environ.get(
        "BEHAVIOR_INTERFACE_MANAGED_APPDATA_ROOTS", ""
    ).split(os.pathsep)
    root = next((item for item in configured_roots if item.strip()), "/var/tmp")
    root = os.path.abspath(os.path.expanduser(root.strip()))
    user = _path_component(os.environ.get("USER") or os.getuid(), str(os.getuid()))
    gpu = _path_component(os.environ.get("OMNIGIBSON_GPU_ID", "0"), "0")
    return os.path.join(root, f"og_appdata_{user}_official_gpu{gpu}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # This local module is stdlib-only. Select TMPDIR before importing anything
    # that can transitively import OmniGibson and create decrypted USD files.
    from behavior_interface.runtime_tmp import configure_process_runtime_tmp

    runtime_dir = configure_process_runtime_tmp("managed-og")

    if not args:
        print(
            "usage: managed_og_entrypoint.py TARGET_SCRIPT [TARGET_ARGS ...]",
            file=sys.stderr,
        )
        return 2
    target = os.path.abspath(os.path.expanduser(args.pop(0)))
    if not os.path.isfile(target):
        print(f"[managed-og] target script not found: {target}", file=sys.stderr)
        return 2

    appdata = os.environ.get("OMNIGIBSON_APPDATA_PATH", "").strip()
    if not appdata:
        appdata = _default_appdata_path()
    appdata = os.path.abspath(os.path.expanduser(appdata))
    os.environ["OMNIGIBSON_APPDATA_PATH"] = appdata

    from behavior_interface.runtime_storage import preflight_storage

    try:
        storage_report = preflight_storage(
            appdata,
            reserve_for_pid=os.getpid(),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[managed-og] storage preflight failed: {exc}", file=sys.stderr)
        return 75

    watchdog = storage_report.get("watchdog")
    print(
        f"[managed-og] appdata={appdata} runtime_tmp={runtime_dir} "
        f"watchdog={'on' if watchdog else 'off'}",
        file=sys.stderr,
        flush=True,
    )

    # Match direct script execution closely: preserve all Hydra overrides and
    # make sibling imports resolve relative to the target script.
    sys.argv = [target, *args]
    target_dir = os.path.dirname(target)
    if not sys.path or sys.path[0] != target_dir:
        sys.path.insert(0, target_dir)
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
