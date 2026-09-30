"""Launch the stock evaluator with a deterministic Carbonite tasking limit."""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

from behavior_interface.gpu_diag import (
    apply_requested_cpu_affinity,
    enforce_fixed_gpu_environment,
)


DEFAULT_TASKING_THREADS = 8
ACTION_TRACE_ENV = "BEHAVIOR_EVAL_TEST_ACTION_TRACE_PATH"
GPU_DYNAMICS_ENV = "BEHAVIOR_EVAL_TEST_GPU_DYNAMICS"
_SYNCHRONOUS_RENDER_ARGS = (
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
    "--/omni/replicator/asyncRendering=false",
    # Emergency workers increase cross-thread fiber resumption and reproduce
    # the BaseMutex ownership crash. The normal worker pool remains large
    # enough for Replicator's render graph to initialize.
    "--/plugins/carb.tasking.plugin/stuckCheckSeconds=0",
    # These settings disable extension and MDL reload only. OmniClient still
    # installs texture watches: a byte-identical JPEG mtime change reproduced
    # the carb.assets mutex abort on 2026-09-30. main() additionally applies
    # the exact-binary native subscription workaround before Kit startup.
    "--/app/extensions/fsWatcherEnabled=false",
    "--/app/material/disableMdlReload=true",
)


def configure_host_thread_defaults() -> None:
    """Bound numerical-library pools before Kit/Omni imports.

    PhysX remains CPU-backed by the stock evaluator.  This only prevents
    inherited OpenMP/BLAS/TBB defaults from multiplying across evaluator
    processes; an operator-provided value is left untouched.
    """
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def configure_gpu_dynamics() -> bool:
    """Opt in to GPU PhysX for isolated A/B experiments only."""
    raw = os.environ.get(GPU_DYNAMICS_ENV, "0").strip().lower()
    if raw not in {"0", "1"}:
        raise ValueError(f"{GPU_DYNAMICS_ENV} must be 0 or 1")
    enabled = raw == "1"
    if enabled:
        from omnigibson.macros import gm

        # The stock evaluator sets this macro to False at module import. Set
        # it after importing the evaluator but before official_main creates
        # the Environment/Simulator. Default remains the stock CPU path.
        gm.USE_GPU_DYNAMICS = True
    return enabled


def configure_omnigibson_gpu_id() -> tuple[int, int]:
    """Set OmniGibson's macro GPU before ``Simulator`` is constructed.

    The stock OmniGibson simulator reads ``gm.GPU_ID`` when it builds its
    ``SimulationApp`` config.  Patching the Isaac class defaults alone is not
    sufficient because the evaluator creates the config from this macro.
    The RTX renderer/Kit ordinal is the physical host ordinal.  The
    ``SimulationApp`` shim below separately maps PhysX to local CUDA ordinal
    0 when a route is masked.
    """
    physical_gpu, local_gpu = evaluator_gpu_mapping()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    simulator_gpu = physical_gpu
    from omnigibson.macros import gm

    gm.GPU_ID = simulator_gpu
    print(
        "[official-test] OmniGibson GPU mapping: "
        f"physical={physical_gpu} local={local_gpu} "
        f"visible={visible or '<unmasked>'} simulator_gpu={simulator_gpu}",
        flush=True,
    )
    return physical_gpu, simulator_gpu


def resident_action_active() -> bool:
    """Return whether the managed interface has a real active action/session.

    The resident evaluator uses this compact probe only to arm recording.
    Initialization and grasp preparation are deliberately excluded.
    """
    port = os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "").strip()
    if not port.isdigit():
        return False
    url = f"http://127.0.0.1:{int(port)}/__official__/idle_probe"
    try:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "Cache-Control": "no-cache"},
        )
        with urllib.request.urlopen(request, timeout=1.5) as response:
            probe = json.load(response)
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return False
    if not isinstance(probe, dict) or probe.get("ok") is not True:
        return False
    if (probe.get("episode_initialization") or {}).get("ready") is not True:
        return False
    if probe.get("idle") is False:
        return True
    return probe.get("action_source") in {"human/tool", "downstream-model"}


def evaluator_action_trace_path() -> str | None:
    """Return the opt-in evaluator-side action trace path."""
    raw = os.environ.get(ACTION_TRACE_ENV, "").strip()
    if not raw:
        return None
    return os.path.abspath(os.path.expanduser(raw))


def _trace_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _trace_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def install_evaluator_action_trace(robot_cls, controller_view_cls, trace_path: str) -> bool:
    """Log actions and controller targets without changing evaluator behavior."""
    original_apply_action = robot_cls.apply_action
    if getattr(original_apply_action, "_eval_test_action_trace", False):
        return False

    path = os.path.abspath(os.path.expanduser(str(trace_path)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trace_lock = threading.Lock()
    trace_sequence = 0
    trace_errors = 0

    @functools.wraps(original_apply_action)
    def traced_apply_action(robot, action):
        nonlocal trace_sequence, trace_errors
        result = original_apply_action(robot, action)
        try:
            controllers = []
            action_offset = 0
            for name, (group_key, controller_idx) in robot.controllers.items():
                command_dim = int(
                    controller_view_cls.get_command_dim(group_key)
                )
                command = action[action_offset : action_offset + command_dim]
                entry = {
                    "name": str(name),
                    "action_start": action_offset,
                    "action_stop": action_offset + command_dim,
                    "command": _trace_jsonable(command),
                }
                if str(name).startswith("gripper_"):
                    entry["goal_after_update"] = _trace_jsonable(
                        controller_view_cls.get_goal(group_key, controller_idx)
                    )
                    entry["last_deployed_control"] = _trace_jsonable(
                        controller_view_cls.get_control(group_key, controller_idx)
                    )
                controllers.append(entry)
                action_offset += command_dim

            gripper_qpos = {}
            joint_positions = robot.get_joint_positions()
            for arm, indices in robot.gripper_control_idx.items():
                gripper_qpos[str(arm)] = _trace_jsonable(
                    joint_positions[indices]
                )

            with trace_lock:
                trace_sequence += 1
                record = {
                    "sequence": trace_sequence,
                    "time_ns": time.time_ns(),
                    "robot": str(getattr(robot, "name", "")),
                    "action": _trace_jsonable(action),
                    "controllers": controllers,
                    "gripper_qpos_before_physics": gripper_qpos,
                }
                with open(path, "a", encoding="utf-8") as trace_file:
                    trace_file.write(
                        json.dumps(
                            record,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        except Exception as exc:
            if trace_errors == 0:
                print(
                    "[official-test] evaluator action trace failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            trace_errors += 1
        return result

    traced_apply_action._eval_test_action_trace = True
    robot_cls.apply_action = traced_apply_action
    return True


def evaluator_gpu_mapping() -> tuple[int, int]:
    """Return (physical RTX GPU, CUDA-visible PhysX ordinal)."""
    physical_raw = os.environ.get(
        "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU",
        os.environ.get("OMNIGIBSON_GPU_ID", "0"),
    )
    local_raw = os.environ.get("OMNIGIBSON_GPU_ID", "0")
    try:
        physical = int(physical_raw)
        local = int(local_raw)
    except ValueError as exc:
        raise ValueError("evaluator GPU ids must be non-negative integers") from exc
    if physical < 0 or local < 0:
        raise ValueError("evaluator GPU ids must be non-negative integers")

    # Isaac Kit's renderer setting is addressed in physical host ordinals,
    # while PhysX is addressed in the CUDA-visible ordinal (normally 0).
    # Reject a contradictory mask here as well as in the shell launcher so a
    # direct ``python -m`` invocation cannot silently render on another card.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        if not visible.isdigit():
            raise ValueError(
                "masked evaluator requires a single numeric "
                f"CUDA_VISIBLE_DEVICES value, got {visible!r}"
            )
        if int(visible) != physical:
            raise ValueError(
                "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU must match the single "
                f"CUDA_VISIBLE_DEVICES GPU (physical={physical}, visible={visible})"
            )
    return physical, local


def kit_renderer_gpu_ordinal(physical_gpu: int) -> int:
    """Map a physical GPU to the Isaac Kit Vulkan ordinal.

    The managed 4090D hosts expose two Vulkan ICD entries for every physical
    card.  The duplicate entries have identical UUID and bus-id pairs, so
    Kit's ``activeGpu`` ordinal is ``2 * physical`` while CUDA / PhysX keeps
    the normal physical or masked-local ordinal.  The stride is configurable
    for a host with a different ICD layout, but the challenge fleet default
    is the observed two-entry layout.
    """
    # ``vulkaninfo`` is not installed on every host, so auto detection is an
    # opt-in fleet setting with a conservative duplicate-ICD fallback.  Match
    # Vulkan NVIDIA device UUIDs to nvidia-smi UUIDs.  A host with an extra
    # non-NVIDIA Vulkan device (the local llvmpipe entry) must compact the
    # NVIDIA ordinals because Kit filters that device before assigning
    # ``activeGpu``.  A host with repeated NVIDIA ICD entries (the remote
    # six-card host) must retain the first ordinal of each UUID pair.
    mapping_raw = os.environ.get("BEHAVIOR_EVAL_TEST_KIT_GPU_MAP", "").strip()
    if mapping_raw:
        if mapping_raw.lower() == "auto":
            detected = _detect_kit_gpu_map()
            if detected is not None and physical_gpu < len(detected):
                return detected[physical_gpu]
        else:
            try:
                mapping = tuple(int(item.strip()) for item in mapping_raw.split(","))
            except ValueError as exc:
                raise ValueError(
                    "BEHAVIOR_EVAL_TEST_KIT_GPU_MAP must be comma-separated integers"
                ) from exc
            if any(item < 0 for item in mapping):
                raise ValueError(
                    "BEHAVIOR_EVAL_TEST_KIT_GPU_MAP values must be non-negative"
                )
            if physical_gpu < len(mapping):
                return mapping[physical_gpu]
            raise ValueError(
                "BEHAVIOR_EVAL_TEST_KIT_GPU_MAP has no entry for physical GPU "
                f"{physical_gpu}"
            )

    raw = os.environ.get("BEHAVIOR_EVAL_TEST_KIT_GPU_STRIDE", "2").strip()
    try:
        stride = int(raw)
    except ValueError as exc:
        raise ValueError(
            "BEHAVIOR_EVAL_TEST_KIT_GPU_STRIDE must be a positive integer"
        ) from exc
    if stride < 1:
        raise ValueError(
            "BEHAVIOR_EVAL_TEST_KIT_GPU_STRIDE must be a positive integer"
        )
    return int(physical_gpu) * stride


def _detect_kit_gpu_map() -> tuple[int, ...] | None:
    """Return Vulkan ordinals for NVIDIA CUDA physical GPUs when available."""
    vulkaninfo = shutil.which("vulkaninfo")
    nvidia_smi = shutil.which("nvidia-smi")
    if not vulkaninfo or not nvidia_smi:
        return None
    try:
        summary = subprocess.run(
            [vulkaninfo, "--summary"],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
        uuids_text = subprocess.run(
            [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    cuda_uuids = [
        line.strip().removeprefix("GPU-").lower()
        for line in uuids_text.splitlines()
        if line.strip()
    ]
    if not cuda_uuids:
        return None
    entries: list[tuple[int, str]] = []
    blocks = re.split(r"(?m)^GPU(\d+):\s*$", summary)
    # split() returns [prefix, ordinal, body, ordinal, body, ...]
    for offset in range(1, len(blocks) - 1, 2):
        ordinal = int(blocks[offset])
        body = blocks[offset + 1]
        if "deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" not in body:
            continue
        match = re.search(r"(?m)^\s*deviceUUID\s*=\s*([^\s]+)", body)
        if match:
            entries.append((ordinal, match.group(1).lower()))
    occurrences: dict[str, list[int]] = {}
    for ordinal, uuid in entries:
        occurrences.setdefault(uuid, []).append(ordinal)
    if not all(uuid in occurrences for uuid in cuda_uuids):
        return None
    counts = {len(occurrences[uuid]) for uuid in cuda_uuids}
    if counts == {1}:
        # Kit drops non-discrete Vulkan devices before exposing activeGpu.
        # The remaining NVIDIA UUID order is the CUDA physical order.
        compact = {uuid: index for index, (_, uuid) in enumerate(entries)}
        return tuple(compact[uuid] for uuid in cuda_uuids)
    if len(counts) == 1 and next(iter(counts)) > 1:
        # Duplicate ICDs retain their first ordinal in Kit's device table.
        return tuple(occurrences[uuid][0] for uuid in cuda_uuids)
    return None


def install_split_gpu_mapping(simulation_app_cls) -> None:
    """Map Kit's renderer and PhysX to the process' GPU namespace.

    Managed routes expose one physical card through ``CUDA_VISIBLE_DEVICES``.
    Isaac Kit's Vulkan enumeration on these hosts contains duplicate ICD
    entries, so the renderer ordinal is translated separately from the CUDA
    / PhysX ordinal.  PhysX remains local ``cuda:0`` in masked routes and the
    physical ordinal in an explicitly unmasked diagnostic launch.
    """
    current_init = simulation_app_cls.__init__
    if getattr(current_init, "_eval_test_split_gpu_mapping", False):
        return
    physical_gpu, physics_gpu = evaluator_gpu_mapping()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    renderer_gpu = kit_renderer_gpu_ordinal(physical_gpu)

    def mapped_init(self, launch_config=None, *args, **kwargs):
        mapped_config = dict(launch_config or {})
        mapped_config["active_gpu"] = renderer_gpu
        mapped_config["physics_gpu"] = physics_gpu
        # Opt-in cache budget for a new resident on a shared GPU. This changes
        # texture residency, not camera configuration, physics, or tick limits.
        texture_budget = os.getenv("BEHAVIOR_EVAL_TEST_TEXTURE_MEMORY_FRACTION", "").strip()
        if texture_budget:
            fraction = float(texture_budget)
            if not 0.01 <= fraction <= 0.6:
                raise ValueError("texture memory fraction must be between 0.01 and 0.6")
            mapped_config["extra_args"] = [*mapped_config.get("extra_args", []),
                f"--/rtx-transient/resourcemanager/texturestreaming/memoryBudget={fraction}"]
            print(f"[official-test] texture streaming memory fraction={fraction}", flush=True)
        print(
            "[official-test] SimulationApp GPU config: "
            f"active_gpu={renderer_gpu} physics_gpu={physics_gpu} "
            f"physical_gpu={physical_gpu} visible={visible or '<unmasked>'}",
            flush=True,
        )
        return current_init(self, mapped_config, *args, **kwargs)

    mapped_init._eval_test_split_gpu_mapping = True
    simulation_app_cls.__init__ = mapped_init


def tasking_thread_limit() -> int:
    raw = os.environ.get(
        "BEHAVIOR_EVAL_TEST_TASKING_THREADS",
        str(DEFAULT_TASKING_THREADS),
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "BEHAVIOR_EVAL_TEST_TASKING_THREADS must be an integer"
        ) from exc
    if not 1 <= value <= 32:
        raise ValueError(
            "BEHAVIOR_EVAL_TEST_TASKING_THREADS must be between 1 and 32"
        )
    return value


def evaluator_extra_args(limit: int | None = None) -> list[str]:
    args = list(_SYNCHRONOUS_RENDER_ARGS)
    # ``limit_cpu_threads`` is consumed by Isaac Sim's SimulationApp.  It
    # appends both Carbonite ``threadCount`` and omni.tbb's
    # ``maxThreadCount`` exactly once during _start_app; adding either here
    # would produce duplicate command-line settings whose precedence depends
    # on Kit version.  Keep this helper limited to settings not synthesized by
    # SimulationApp itself.
    use_omni_job = os.environ.get(
        "BEHAVIOR_EVAL_TEST_USE_OMNI_JOB",
        "0",
    )
    if use_omni_job not in {"0", "1"}:
        raise ValueError("BEHAVIOR_EVAL_TEST_USE_OMNI_JOB must be 0 or 1")
    args.append(
        "--/plugins/carb.tasking.plugin/useOmniJob="
        + ("true" if use_omni_job == "1" else "false")
    )
    return args


def configure_simulation_app_defaults(simulation_app_cls, limit: int) -> None:
    """Apply evaluator-only launch defaults before OmniGibson creates Kit."""
    launch_config = dict(simulation_app_cls.DEFAULT_LAUNCHER_CONFIG)
    launch_config["limit_cpu_threads"] = limit
    # Keep USD and material asset completion on the loading path. Leaving
    # these asynchronous lets carb.assets fibers survive into the policy loop,
    # where Kit 107 can resume one on another worker while it owns a
    # thread-affine mutex and abort the evaluator.
    launch_config["sync_loads"] = True
    launch_config["extra_args"] = [
        *launch_config.get("extra_args", []),
        *evaluator_extra_args(limit),
    ]
    simulation_app_cls.DEFAULT_LAUNCHER_CONFIG = launch_config


def install_prepared_usd_cache(usd_object_cls) -> bool:
    """Avoid decrypting/exporting the same object again in prebuild + load."""

    original = usd_object_cls._prepare_to_load
    if getattr(original, "_eval_test_prepared_usd_cache", False):
        return False

    @functools.wraps(original)
    def cached_prepare(obj, *args, **kwargs):
        cached = getattr(obj, "_eval_test_prepared_usd_path", None)
        if cached and os.path.isfile(cached):
            return cached
        prepared = original(obj, *args, **kwargs)
        obj._eval_test_prepared_usd_path = prepared
        return prepared

    cached_prepare._eval_test_prepared_usd_cache = True
    usd_object_cls._prepare_to_load = cached_prepare
    return True


def install_effort_gripper_no_op_compat(
    controller_cls,
    compute_backend,
) -> bool:
    """Provide the neutral goal omitted by stock v3.9.1 effort grippers."""

    original_goal = controller_cls.compute_no_op_goal
    if getattr(original_goal, "_eval_test_effort_no_op_compat", False):
        return False

    original_command = controller_cls._compute_no_op_command

    @functools.wraps(original_goal)
    def compatible_goal(controller, controller_idx):
        if controller._mode != "binary" and controller._motor_type == "effort":
            return {"target": compute_backend.zeros(controller.command_dim)}
        return original_goal(controller, controller_idx)

    @functools.wraps(original_command)
    def compatible_command(controller, controller_idx):
        if controller._mode != "binary" and controller._motor_type == "effort":
            return compute_backend.zeros(controller.command_dim)
        return original_command(controller, controller_idx)

    compatible_goal._eval_test_effort_no_op_compat = True
    compatible_command._eval_test_effort_no_op_compat = True
    controller_cls.compute_no_op_goal = compatible_goal
    controller_cls._compute_no_op_command = compatible_command
    return True


def install_eager_assisted_grasp_quat2mat(transform_utils_module) -> bool:
    """Avoid a first-grasp TorchInductor compile without changing the math."""

    current = transform_utils_module.quat2mat
    if getattr(current, "_eval_test_eager_assisted_grasp", False):
        return False
    eager = getattr(current, "_torchdynamo_orig_callable", None)
    if not callable(eager):
        return False
    eager._eval_test_eager_assisted_grasp = True
    transform_utils_module.quat2mat = eager
    return True


def run_resident_evaluator(eval_module, evaluator_cls=None) -> None:
    """Run the stock evaluator loop with resident-safe video arming.

    The official evaluator starts recording before its first gated action. In
    a resident route that creates empty/boot videos even when no model session
    exists. This copy keeps the official cfg, metrics, JSON, and rollout loop,
    but arms the writer only after the interface reports a real active action.
    """
    from pathlib import Path

    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    from omnigibson.macros import gm
    if evaluator_cls is not None:
        Evaluator = evaluator_cls

    args = eval_module.parse_args()
    gm.HEADLESS = args.headless
    if args.headless and os.getenv("OMNIGIBSON_KEEP_VIEWER_CAMERA", "0") != "1":
        gm.RENDER_VIEWER_CAMERA = False
    seed = seed_everything(DEFAULT_EVAL_SEED)
    eval_module.logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")

    instance_ids = resolve_instance_ids(
        args.task_name, args.instance_indices, mode=args.mode
    )
    eval_module.logger.info(
        f"Resolved {args.mode} instance ids for {args.task_name}: {instance_ids}"
    )

    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = OmegaConf.load(str(robot_config_path))
        eval_module.logger.info(f"Loaded robot config from {robot_config_path}")

    if args.policy == "websocket":
        model_cfg = {
            "_target_": "omnigibson.eval.policies.WebsocketPolicy",
            "host": args.host,
            "port": args.port,
        }
    else:
        model_cfg = {
            "_target_": "omnigibson.eval.policies.LocalPolicy",
            "action_dim": None,
        }

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": args.policy,
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            # Resident routes may spend an unbounded amount of time in
            # simulator-side initialization before a model session exists.
            # Apply the official max-tick budget only after the first real
            # agent action; otherwise a slow grasp-prep phase can emit a
            # false zero-score rollout with no agent.
            "max_steps": 2_000_000_000,
            "write_video": args.write_video,
            "mode": args.mode,
            "seed": seed,
            "num_envs": 1,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    # The resident stack survives completion of an experiment. Each full pass
    # returns to the first configured instance and blocks on the idle gate.
    # Allocate new rollout IDs without changing or overwriting earlier metrics.
    from itertools import cycle
    from collections import deque
    next_rollout = {}
    for item in instance_ids:
        existing = []
        for filename in Path(json_dir).glob(f"{args.task_name}_{item}_*.json"):
            try:
                existing.append(int(filename.stem.rsplit("_", 1)[1]))
            except ValueError:
                pass
        next_rollout[int(item)] = max(existing, default=-1) + 1
    results = deque(maxlen=1000)
    with Evaluator(cfg) as evaluator:
        for instance_id in cycle(instance_ids):
            evaluator.load_task_instance(int(instance_id))
            for rollout_index in range(args.num_rollouts):
                rollout_id = next_rollout[int(instance_id)]
                video_path = os.path.join(
                    video_dir, f"{args.task_name}_{instance_id}_{rollout_id}.mp4"
                )
                recording_started = False
                try:
                    # The official v3.9.3 load_batch has already prepared
                    # the first rollout, including initial Q-score states.
                    # Subsequent rollouts and legacy evaluators still reset.
                    if rollout_index or not getattr(
                        evaluator, "load_task_instance_resets_rollout", False
                    ):
                        evaluator.reset()
                    terminated = truncated = False
                    active_session_seen = False
                    active_steps = 0
                    while True:
                        active_before = resident_action_active()
                        if active_before:
                            active_session_seen = True
                        if active_session_seen and active_steps >= int(args.max_steps):
                            break
                        if (
                            args.write_video
                            and not recording_started
                            and active_before
                        ):
                            evaluator.start_recording(video_path, rate=args.video_fps)
                            recording_started = True

                        # Publish the same active-step counter and limit that
                        # this resident loop enforces; never its idle 2B cap.
                        evaluator._resident_tick_budget = (active_steps, int(args.max_steps))
                        terminated, truncated = evaluator.step()
                        if recording_started and hasattr(evaluator, "record_resident_frame"):
                            evaluator.record_resident_frame()
                        active_after = resident_action_active()
                        if active_after:
                            active_session_seen = True
                        if active_before or active_after:
                            active_steps += 1
                        if (
                            args.write_video
                            and not recording_started
                            and active_after
                        ):
                            evaluator.start_recording(video_path, rate=args.video_fps)
                            recording_started = True

                        if not (terminated or truncated):
                            continue
                        if not active_session_seen:
                            # Initialization/prep ended without a model action.
                            # Keep the resident route alive and discard this
                            # non-rollout; no metrics JSON or video is emitted.
                            eval_module.logger.warning(
                                f"Resident instance={instance_id} ended before an active "
                                "agent session; resetting without submission artifacts."
                            )
                            if not getattr(
                                evaluator, "load_task_instance_resets_rollout", False
                            ):
                                evaluator.reset()
                            evaluator.load_task_instance(int(instance_id))
                            terminated = truncated = False
                            continue
                        break

                    if not active_session_seen:
                        # Defensive guard for a reset/stop race.
                        continue
                    steps = active_steps
                    if hasattr(evaluator, "resident_result"):
                        success, metrics = evaluator.resident_result()
                    else:
                        success = bool(evaluator.env.task.success)
                        metrics = {}
                        for metric in evaluator.metrics:
                            metrics.update(metric.aggregate(evaluator.env))
                    result = {
                        "task": args.task_name,
                        "instance_id": int(instance_id),
                        "rollout_id": rollout_id,
                        "steps": steps,
                        "success": success,
                        **metrics,
                    }
                    out_path = os.path.join(
                        json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json"
                    )
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2, default=float)
                    q_score = metrics.get("q_score", {}).get("final")
                    video_msg = (
                        f" | video -> {video_path}"
                        if recording_started
                        else " | video -> <not armed: no active session>"
                    )
                    eval_module.logger.info(
                        f"Result: instance={instance_id} rollout={rollout_id} "
                        f"steps={steps} success={success} q_score={q_score} "
                        f"-> {out_path}{video_msg}"
                    )
                    results.append(result)
                    next_rollout[int(instance_id)] = rollout_id + 1
                except Exception:
                    eval_module.logger.exception(
                        f"Instance {instance_id} rollout {rollout_id} failed."
                    )
                    raise
                finally:
                    if args.write_video and recording_started:
                        evaluator.stop_recording()

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = (
        sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n
        if n
        else 0.0
    )
    eval_module.logger.info(
        f"Eval summary: {n_success}/{n} success | mean q_score={mean_q:.3f} "
        f"| task={args.task_name}"
    )


def main() -> None:
    port_hint = os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "").strip()
    # The shell launcher has one explicit diagnostic escape hatch that keeps
    # CUDA unmasked and addresses the selected physical card directly.  Do not
    # rewrite that environment back to a local cuda:0 mask; production always
    # leaves this flag at 0 and therefore remains fail-closed.
    unmasked_diagnostic = (
        os.environ.get("BEHAVIOR_EVAL_TEST_UNMASK_CUDA", "0").strip() == "1"
    )
    if port_hint.isdigit() and not unmasked_diagnostic:
        try:
            owned_gpu = enforce_fixed_gpu_environment(int(port_hint))
        except ValueError as exc:
            raise SystemExit(
                f"official evaluator GPU ownership validation failed: {exc}"
            ) from exc
        if owned_gpu is not None:
            print(
                f"Official evaluator port={port_hint} physical_gpu={owned_gpu} "
                "local_cuda=cuda:0",
                flush=True,
            )
    elif port_hint.isdigit() and unmasked_diagnostic:
        physical_gpu = os.environ.get("BEHAVIOR_EVAL_TEST_PHYSICAL_GPU", "").strip()
        if physical_gpu:
            print(
                f"Official evaluator diagnostic-unmasked port={port_hint} "
                f"physical_gpu={physical_gpu} local_cuda=cuda:{os.environ.get('OMNIGIBSON_GPU_ID', physical_gpu)}",
                flush=True,
            )
    try:
        apply_requested_cpu_affinity()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"official evaluator CPU affinity validation failed: {exc}"
        ) from exc
    configure_host_thread_defaults()
    # OmniGibson allocates its decrypted-USD directory at import time. Keep it
    # process-owned so SIGTERM and evaluator crashes cannot leak /tmp/tmp*.
    from behavior_interface.runtime_tmp import configure_process_runtime_tmp

    configure_process_runtime_tmp(
        os.environ.get("BEHAVIOR_EVAL_TEST_POLICY_PORT", "official")
    )
    appdata_path = os.environ.get("OMNIGIBSON_APPDATA_PATH", "").strip()
    if appdata_path:
        from behavior_interface.runtime_storage import preflight_storage

        preflight_storage(appdata_path, reserve_for_pid=os.getpid())
    # OmniGibson imports Python dependencies before Isaac swaps in its bundled
    # runtime libraries. Preserve that ordering, then configure SimulationApp
    # before Evaluator constructs the simulator.
    import omnigibson.eval.eval as eval_module
    from omnigibson.eval.eval import main as official_main
    from omnigibson.eval.evaluator import Evaluator
    is_batched = hasattr(Evaluator, "load_batch") and not hasattr(Evaluator, "step")
    if is_batched:
        from behavior_interface_eval_test.official_evaluator_v393 import route_evaluator_class
        Evaluator = route_evaluator_class(Evaluator)
    from omnigibson.controllers.controller_view import ControllerView
    from omnigibson.controllers.multi_finger_gripper_controller import (
        MultiFingerGripperController,
        cb,
    )
    from omnigibson.robots import Robot
    from omnigibson.objects.usd_object import USDObject
    from omnigibson.utils import transform_utils as transform_utils
    from isaacsim import SimulationApp
    from behavior_interface_eval_test.native_asset_watches import (
        disable_native_asset_watches,
    )

    watch_policy = disable_native_asset_watches()
    print(
        "[official-test] native asset watch policy: "
        + json.dumps(watch_policy, sort_keys=True),
        flush=True,
    )

    configure_omnigibson_gpu_id()
    gpu_dynamics = configure_gpu_dynamics()
    install_split_gpu_mapping(SimulationApp)
    install_prepared_usd_cache(USDObject)
    install_effort_gripper_no_op_compat(
        MultiFingerGripperController,
        cb,
    )
    install_eager_assisted_grasp_quat2mat(transform_utils)
    trace_path = evaluator_action_trace_path()
    if trace_path is not None:
        install_evaluator_action_trace(Robot, ControllerView, trace_path)
        print(
            f"[official-test] evaluator action trace enabled: {trace_path}",
            flush=True,
        )
    limit = tasking_thread_limit()
    configure_simulation_app_defaults(SimulationApp, limit)
    print(
        "[official-test] Carbonite/Replicator tasking threads "
        f"limited to {limit}; synchronous USD/material loading, synchronous "
        "rendering, the configured tasking backend, and split RTX/PhysX GPU "
        f"mapping are applied before launching the stock evaluator; "
        f"gpu_dynamics={'on' if gpu_dynamics else 'off'}.",
        flush=True,
    )

    # 测试口右上角「选 instance + RESET」，以及 harness 的模型收工交卷：
    # 官方 eval.py 不会收 HTTP。在 Evaluator.step 里轮询请求文件，
    # 由 evaluator 自己 reset / 按 truncated 写 JSON 再切 instance。
    operator_port = os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "").strip()
    if operator_port.isdigit():
        from behavior_interface_eval_test.operator_scene_control import (
            install_operator_reset_on_evaluator_class,
        )

        if install_operator_reset_on_evaluator_class(Evaluator, int(operator_port)):
            print(
                "[official-test] operator scene reset listening for "
                f"HTTP port {operator_port}",
                flush=True,
            )

    if os.environ.get("BEHAVIOR_EVAL_TEST_RESIDENT_MODE", "0").strip() == "1":
        run_resident_evaluator(eval_module, Evaluator)
    elif is_batched:
        from behavior_interface_eval_test.official_evaluator_v393 import run_finite_route
        run_finite_route(eval_module, Evaluator)
    else:
        official_main()


if __name__ == "__main__":
    main()
