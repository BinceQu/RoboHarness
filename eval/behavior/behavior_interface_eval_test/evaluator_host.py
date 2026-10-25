"""Private OmniGibson evaluator host for the separated compatibility stack.

The host owns the task, simulator, reset lifecycle, physics loop, observations,
and legacy skill execution. Its HTTP service must be bound to loopback; the
public Human/Model endpoint is behavior_interface_eval_test.interface_gateway.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from behavior_interface.gpu_diag import (
    apply_requested_cpu_affinity,
    enforce_fixed_gpu_environment,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Private OmniGibson Evaluator Host for Behavior Interface test."
    )
    parser.add_argument("--task", default=os.environ.get("TASK", "make_microwave_popcorn"))
    parser.add_argument("--scene", default=os.environ.get("SCENE", "house_double_floor_lower"))
    parser.add_argument("--robot", default=os.environ.get("ROBOT"))
    parser.add_argument(
        "--robot-dof",
        type=int,
        choices=(7, 8),
        default=int(os.environ.get("BEHAVIOR_ROBOT_DOF", "8")),
    )
    parser.add_argument("--robot-config", default=os.environ.get("BEHAVIOR_ROBOT_CONFIG"))
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BEHAVIOR_EVAL_TEST_EVALUATOR_PORT", "18080")),
    )
    parser.add_argument("--main-w", type=int, default=960)
    parser.add_argument("--main-h", type=int, default=540)
    parser.add_argument("--sub-w", type=int, default=320)
    parser.add_argument("--sub-h", type=int, default=240)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--physics-per-render", type=int, default=4)
    parser.add_argument(
        "--tool-version",
        choices=("v0", "v1", "v1_shortcut", "v2", "v3"),
        default=os.environ.get("INTERFACE_TOOL_VERSION", "v2"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, unknown_kit_args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "Evaluator Host must bind to loopback. "
            "Expose behavior_interface_eval_test.interface_gateway instead."
        )

    os.environ["INTERFACE_TOOL_VERSION"] = args.tool_version
    os.environ["BEHAVIOR_ROBOT_DOF"] = str(args.robot_dof)

    # When this private host is launched for an official HTTP port, enforce
    # the same physical owner before importing server/OmniGibson.  Standalone
    # legacy hosts with only port=18080 remain unchanged and must be launched
    # through launch_evaluator_host.sh with an explicit single-card CVD.
    port_hint = os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "").strip()
    if port_hint.isdigit():
        try:
            enforce_fixed_gpu_environment(int(port_hint))
        except ValueError as exc:
            raise SystemExit(
                f"evaluator host GPU ownership validation failed: {exc}"
            ) from exc
    try:
        apply_requested_cpu_affinity()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"evaluator host CPU affinity validation failed: {exc}"
        ) from exc

    from behavior_interface import server as legacy_server

    legacy_server.enable_faulthandler()
    legacy_server._ensure_runtime_env_defaults()

    import behavior_interface.skills as skills_pkg

    skills_pkg = importlib.reload(skills_pkg)
    loaded = skills_pkg.load_all_skills()

    evaluator = legacy_server.BehaviorInterface(
        task=args.task,
        robot=args.robot,
        robot_dof=args.robot_dof,
        scene_model=args.scene,
        main_size=(args.main_w, args.main_h),
        sub_size=(args.sub_w, args.sub_h),
        target_hz=args.hz,
        dry_run=args.dry_run,
        physics_per_render=args.physics_per_render,
        activity_instance_id=args.instance_id,
        robot_config_path=args.robot_config,
        cli_argv=sys.argv,
    )
    legacy_server.enable_faulthandler(evaluator.log)
    evaluator.log(
        "EVAL_TEST architecture: evaluator host owns OmniGibson; "
        "public interface runs in a separate gateway process"
    )
    evaluator.log(
        f"EVAL_TEST tool_version={args.tool_version} skills loaded: "
        f"{loaded} -> {[item['name'] for item in skills_pkg.list_skills()]}"
    )
    if unknown_kit_args:
        evaluator.log(f"EVAL_TEST passing through kit args: {unknown_kit_args}")

    evaluator.init_simulation()

    from behavior_interface.web import run_web_in_thread

    run_web_in_thread(evaluator, host=args.host, port=args.port)
    evaluator.log(
        f"EVAL_TEST private evaluator endpoint=http://{args.host}:{args.port}; "
        "do not expose this port to Human/Model clients"
    )
    evaluator.run_forever()


if __name__ == "__main__":
    main()
