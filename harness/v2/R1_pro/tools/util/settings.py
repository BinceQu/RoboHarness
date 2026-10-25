"""Resolve the current session; never fall back to another session or server."""
import json
import os
from pathlib import Path


def workspace_path():
    explicit = os.environ.get("ROBOT_WORKSPACE")
    candidates = [Path(explicit)] if explicit else [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / ".session.json").is_file():
            return candidate.resolve()
    raise RuntimeError("This is a template, not a live session. Initialize a fresh workspace before controlling a robot.")


def settings():
    workspace = workspace_path()
    result = json.loads((workspace / ".session.json").read_text())
    if Path(result["workspace"]).resolve() != workspace:
        raise RuntimeError("Session configuration belongs to another workspace")
    if "trajectory_dir" not in result:
        # Sessions created by an older launcher: keep evidence beside, never inside, the workspace.
        result["trajectory_dir"] = str(workspace.parent / "trajectory" if workspace.name == "workspace"
                                       else workspace.with_name(workspace.name + ".trajectory"))
    for name in ("recordings", "calls"):
        (Path(result["trajectory_dir"]) / name).mkdir(parents=True, exist_ok=True)
    return result


def trajectory_path(config):
    """Harness-owned evidence directory; deliberately outside the model's workspace."""
    path = Path(config["trajectory_dir"]).resolve()
    if path.is_relative_to(Path(config["workspace"]).resolve()):
        raise RuntimeError("trajectory_dir must be outside the model workspace")
    return path


def configure():
    config = settings()
    state = Path(config["state_dir"])
    os.environ["ROBOT_WORKSPACE"] = config["workspace"]
    os.environ["ROBOT_RECORDINGS"] = str(trajectory_path(config) / "recordings")
    os.environ["ROBOT_COMPILE_TMP"] = config["compile_tmp"]
    # Numerical subprocesses share this session's explicit GPU assignment;
    # a tool may never inherit a different GPU by default.
    gpu = config.get("gpu")
    if type(gpu) is not int or gpu < 0:
        raise RuntimeError("Session must explicitly assign a nonnegative physical GPU index")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), BEHAVIOR_INTERFACE_PHYSICAL_GPU=str(gpu),
                      BEHAVIOR_EVAL_TEST_PHYSICAL_GPU=str(gpu), OFFICIAL_V2_LITE_OCCUPANCY_DEVICE="cuda:0",
                      BEHAVIOR_PYTHON=config["numerical_python"],
                      BEHAVIOR_EVAL_TEST_ROBOT_PROFILE="r1pro_8dof_hf250",
                      BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL="0")
    # All implementation, interpreter, model assets and state belong to this
    # workspace. Never inherit the previous machine's project/data paths.
    workspace = Path(config["workspace"]).resolve()
    for key in ("mesh_root", "numerical_python", "state_dir", "compile_tmp"):
        if not Path(config[key]).resolve().is_relative_to(workspace):
            raise RuntimeError(f"{key} must be inside this workspace")
    os.environ["EEF_NEAR_SAM2_CKPT"] = str(workspace / "robot_config/assets/sam2.1_hiera_small.pt")
    os.environ.pop("EEF_NEAR_SAM2_REPO", None)
    os.environ.pop("EEF_NEAR_SAM2_DEPS", None)
    cache = state / "cache"
    for variable, directory in {
        "XDG_CACHE_HOME": cache, "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
        "WARP_CACHE_PATH": cache / "warp", "CUDA_CACHE_PATH": cache / "cuda",
        "NUMBA_CACHE_DIR": cache / "numba",
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["PYTHONNOUSERSITE"] = "1"
    return config


def plan_key():
    value = bytes.fromhex((Path(settings()["state_dir"]) / "plan.key").read_text().strip())
    if len(value) != 32:
        raise ValueError("Session plan integrity key is invalid")
    return value
