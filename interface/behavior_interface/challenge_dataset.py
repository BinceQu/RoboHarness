"""Strict path resolution and validation for BEHAVIOR Challenge 2026 data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from typing import Iterable


CHALLENGE_DATASET_NAME = "2026-challenge-task-instances"
CHALLENGE_MODES = {"train", "public_test", "hidden_test"}
PUBLIC_TEST_INSTANCE_IDS = tuple(range(301, 321))
HIDDEN_TEST_INSTANCE_IDS = tuple(range(321, 341))
TRAIN_INSTANCE_IDS = tuple(range(1, 301))


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "public_test").strip()
    if value not in CHALLENGE_MODES:
        raise ValueError(f"invalid challenge mode: {value!r}")
    return value


def mode_dir(mode: str) -> str:
    return {
        "train": "scenes",
        "public_test": os.path.join("scene_test", "public"),
        "hidden_test": os.path.join("scene_test", "private"),
    }[normalize_mode(mode)]


def mode_for_instance(instance_id: int, default_mode: str = "public_test") -> str:
    instance_id = int(instance_id)
    if instance_id in TRAIN_INSTANCE_IDS:
        return "train"
    if instance_id in HIDDEN_TEST_INSTANCE_IDS:
        return "hidden_test"
    if instance_id in PUBLIC_TEST_INSTANCE_IDS:
        return "public_test"
    return normalize_mode(default_mode)


def eval_instance_ids(mode: str, limit: int = 10) -> list[int]:
    ids = {
        "train": TRAIN_INSTANCE_IDS,
        "public_test": PUBLIC_TEST_INSTANCE_IDS,
        "hidden_test": HIDDEN_TEST_INSTANCE_IDS,
    }[normalize_mode(mode)]
    return list(ids[: max(0, int(limit))])


def dataset_root(data_path: str) -> str:
    return os.path.join(os.path.abspath(data_path), CHALLENGE_DATASET_NAME)


def template_path(data_path: str, task_name: str, scene_model: str, mode: str) -> str:
    filename = f"{scene_model}_task_{task_name}_0_0_template.json"
    return os.path.join(
        dataset_root(data_path),
        mode_dir(mode),
        scene_model,
        "json",
        filename,
    )


def stable_path(data_path: str, scene_model: str) -> str:
    """场景基线文件（<scene>_stable.json）。

    该文件自带 versions 声明（behavior-1k-assets 3.7.2rc1），其
    expected_file_hash 与官方发布的同版资产逐一对应。task template 里的
    同名对象则来自另一套未公开的资产基线，hash 全部对不上。

    stable.json 只在 scenes/ 下发布，scene_test/{public,private} 没有；
    但同一场景的房屋本体在三种 mode 下是同一批对象，可以共用。
    """
    return os.path.join(
        dataset_root(data_path),
        "scenes",
        scene_model,
        "json",
        f"{scene_model}_stable.json",
    )


def align_template_asset_hashes(
    template: dict,
    data_path: str,
    scene_model: str,
) -> dict[str, object]:
    """就地用 stable 基线的 expected_file_hash 校正 template 中的同名对象。

    只改 expected_file_hash，不动 state、scale、category、model，因此物体摆放
    与关节开合完全保持 template 原样。仅在 stable 与 template 对该对象的其余
    init args 完全一致时才覆盖，避免把两个不同对象错认成同一个。

    返回统计信息，供调用方写日志。
    """
    result: dict[str, object] = {
        "aligned": 0,
        "already_matching": 0,
        "unmatched": [],
        "stable": None,
        "error": None,
    }
    path = stable_path(data_path, scene_model)
    if not os.path.isfile(path):
        result["error"] = f"stable scene baseline missing: {path}"
        return result
    result["stable"] = path

    try:
        with open(path, "r", encoding="utf-8") as f:
            stable_init = (json.load(f).get("objects_info") or {}).get("init_info") or {}
    except (OSError, ValueError) as exc:
        result["error"] = f"unreadable stable scene baseline {path}: {exc}"
        return result

    template_init = (template.get("objects_info") or {}).get("init_info") or {}
    for name, entry in template_init.items():
        args = (entry or {}).get("args")
        if not isinstance(args, dict) or "expected_file_hash" not in args:
            continue
        stable_args = ((stable_init.get(name) or {}).get("args")) or {}
        stable_hash = stable_args.get("expected_file_hash")
        if not stable_hash:
            # stable 里没有的对象就是任务额外物体，没有权威基线可用，保持原样。
            result["unmatched"].append(name)
            continue
        if args.get("category") != stable_args.get("category") or args.get("model") != stable_args.get("model"):
            result["unmatched"].append(name)
            continue
        if args["expected_file_hash"] == stable_hash:
            result["already_matching"] = int(result["already_matching"]) + 1
            continue
        args["expected_file_hash"] = stable_hash
        result["aligned"] = int(result["aligned"]) + 1
    return result


def _dataset_object_asset_hash(data_path: str, args: dict) -> tuple[str | None, str | None]:
    """Return the on-disk encrypted USD hash for one DatasetObject entry."""
    category = str(args.get("category") or "").strip()
    model = str(args.get("model") or "").strip()
    if not category or not model:
        return None, "missing category/model"
    path = os.path.join(
        os.path.abspath(data_path),
        "behavior-1k-assets",
        "objects",
        category,
        model,
        "usd",
        f"{model}.encrypted.usd",
    )
    if not os.path.isfile(path):
        return None, f"asset missing: {path}"
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), None


def build_stable_backed_task_scene(
    template: dict,
    data_path: str,
    scene_model: str,
) -> tuple[dict, dict[str, object]]:
    """Build a coherent task cache from stable scene data and task-only additions.

    Shared objects use the stable file's matching init/state pair. The challenge
    template contributes task metadata, task systems, and objects absent from the
    stable scene. This prevents states sampled against another asset snapshot from
    being applied to the local scene model.
    """
    path = stable_path(data_path, scene_model)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"stable scene baseline missing: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        stable = json.load(stream)
    if not isinstance(stable, dict) or not isinstance(template, dict):
        raise ValueError("stable scene and task template must both be mappings")

    for label, scene in (("stable", stable), ("template", template)):
        value = (((scene.get("init_info") or {}).get("args") or {}).get("scene_model"))
        if value and str(value) != str(scene_model):
            raise ValueError(
                f"{label} scene_model={value!r} does not match {scene_model!r}"
            )

    stable_init = ((stable.get("objects_info") or {}).get("init_info")) or {}
    template_init = ((template.get("objects_info") or {}).get("init_info")) or {}
    stable_registry = (
        (((stable.get("state") or {}).get("registry") or {}).get("object_registry"))
        or {}
    )
    template_registry = (
        (((template.get("state") or {}).get("registry") or {}).get("object_registry"))
        or {}
    )
    if set(stable_init) != set(stable_registry):
        raise ValueError("stable scene init/state object registries are inconsistent")

    shared = sorted(set(stable_init) & set(template_init))
    incompatible: list[str] = []
    for name in shared:
        stable_entry = stable_init[name] or {}
        template_entry = template_init[name] or {}
        stable_args = stable_entry.get("args") or {}
        template_args = template_entry.get("args") or {}
        if (
            stable_entry.get("class_module") != template_entry.get("class_module")
            or stable_entry.get("class_name") != template_entry.get("class_name")
            or stable_args.get("category") != template_args.get("category")
            or stable_args.get("model") != template_args.get("model")
        ):
            incompatible.append(name)
    if incompatible:
        raise ValueError(
            "task template and stable scene disagree on shared objects: "
            f"{incompatible[:12]}"
        )

    merged = copy.deepcopy(stable)
    merged.setdefault("metadata", {})
    merged["metadata"].update(copy.deepcopy(template.get("metadata") or {}))
    merged.setdefault("objects_info", {}).setdefault("init_info", {})
    merged.setdefault("state", {}).setdefault("registry", {})
    merged_registry = merged["state"]["registry"]
    merged_registry["system_registry"] = copy.deepcopy(
        (((template.get("state") or {}).get("registry") or {}).get("system_registry"))
        or {}
    )
    merged_objects = merged_registry.setdefault("object_registry", {})
    merged_init = merged["objects_info"]["init_info"]

    task_only = sorted(set(template_init) - set(stable_init))
    corrected_hashes: dict[str, str] = {}
    for name in task_only:
        if name not in template_registry:
            raise ValueError(f"task-only object {name!r} has no template state")
        entry = copy.deepcopy(template_init[name])
        args = entry.get("args") or {}
        if "expected_file_hash" in args:
            file_hash, error = _dataset_object_asset_hash(data_path, args)
            if error:
                raise FileNotFoundError(
                    f"cannot resolve task-only object {name!r}: {error}"
                )
            args["expected_file_hash"] = file_hash
            corrected_hashes[name] = str(file_hash)
        merged_init[name] = entry
        merged_objects[name] = copy.deepcopy(template_registry[name])

    init_args = ((merged.get("init_info") or {}).get("args")) or {}
    init_args["scene_file"] = None
    init_args["include_robots"] = False

    if set(merged_init) != set(merged_objects):
        missing_state = sorted(set(merged_init) - set(merged_objects))
        missing_init = sorted(set(merged_objects) - set(merged_init))
        raise ValueError(
            "merged task scene init/state mismatch "
            f"missing_state={missing_state[:12]} missing_init={missing_init[:12]}"
        )

    stats: dict[str, object] = {
        "stable": path,
        "shared_objects": len(shared),
        "task_only_objects": task_only,
        "corrected_task_hashes": corrected_hashes,
        "object_count": len(merged_init),
    }
    return merged, stats


def tro_path(
    data_path: str,
    task_name: str,
    scene_model: str,
    instance_id: int,
    mode: str,
) -> str:
    stem = f"{scene_model}_task_{task_name}_0_{int(instance_id)}_template"
    return os.path.join(
        dataset_root(data_path),
        mode_dir(mode),
        scene_model,
        "json",
        f"{scene_model}_task_{task_name}_instances",
        f"{stem}-tro_state.json",
    )


def _read_csv_catalog(path: str) -> dict[int, str]:
    tasks: dict[int, str] = {}
    names: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"Task ID", "Task"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(
                f"invalid BEHAVIOR 2026 task catalog columns in {path}: "
                f"expected {sorted(required)}, got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            raw_id = str(row.get("Task ID") or "").strip()
            task_name = str(row.get("Task") or "").strip()
            try:
                task_id = int(raw_id)
            except ValueError as exc:
                raise ValueError(
                    f"invalid task id {raw_id!r} at {path}:{line_number}"
                ) from exc
            if not task_name:
                raise ValueError(f"empty task name at {path}:{line_number}")
            if task_id in tasks:
                raise ValueError(f"duplicate task id {task_id} in {path}")
            if task_name in names:
                raise ValueError(f"duplicate task name {task_name!r} in {path}")
            tasks[task_id] = task_name
            names.add(task_name)
    return tasks


def _read_jsonl_catalog(path: str) -> dict[int, str]:
    tasks: dict[int, str] = {}
    names: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = int(row["task_index"])
                task_name = str(row["task_name"]).strip()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid task record at {path}:{line_number}"
                ) from exc
            if not task_name:
                raise ValueError(f"empty task name at {path}:{line_number}")
            if task_id in tasks:
                raise ValueError(f"duplicate task id {task_id} in {path}")
            if task_name in names:
                raise ValueError(f"duplicate task name {task_name!r} in {path}")
            tasks[task_id] = task_name
            names.add(task_name)
    return tasks


def _assert_catalog_matches(
    expected: dict[int, str],
    actual: dict[int, str],
    *,
    source: str,
) -> None:
    expected_ids = set(expected)
    actual_ids = set(actual)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"BEHAVIOR 2026 task catalog IDs differ in {source}: "
            f"missing={missing} extra={extra}"
        )
    for task_id in sorted(expected):
        expected_name = expected[task_id]
        actual_name = actual[task_id]
        if actual_name != expected_name:
            raise ValueError(
                "BEHAVIOR 2026 task catalog mismatch at "
                f"id={task_id}: interface={expected_name!r}, "
                f"{source}={actual_name!r}"
            )


def validate_task_catalog(data_path: str) -> dict[str, object]:
    """Require interface and installed 2026 dataset task IDs to match exactly."""
    from .challenge_tasks import challenge_tasks

    root = dataset_root(data_path)
    metadata_dir = os.path.join(root, "metadata")
    csv_path = os.path.join(metadata_dir, "B100_task_misc.csv")
    jsonl_path = os.path.join(metadata_dir, "task.jsonl")
    for path in (csv_path, jsonl_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"BEHAVIOR 2026 challenge dataset is missing or incomplete: {path}"
            )

    expected = {int(task["id"]): str(task["name"]) for task in challenge_tasks()}
    expected_ids = set(range(100))
    if set(expected) != expected_ids:
        raise ValueError(
            "interface BEHAVIOR 2026 task IDs must be contiguous from 0 through 99"
        )

    csv_tasks = _read_csv_catalog(csv_path)
    jsonl_tasks = _read_jsonl_catalog(jsonl_path)
    _assert_catalog_matches(expected, csv_tasks, source=csv_path)
    _assert_catalog_matches(expected, jsonl_tasks, source=jsonl_path)
    _assert_catalog_matches(csv_tasks, jsonl_tasks, source=jsonl_path)
    return {
        "root": root,
        "count": len(expected),
        "tasks": dict(expected),
        "csv": csv_path,
        "jsonl": jsonl_path,
    }


def validate_task_assets(
    data_path: str,
    task_name: str,
    scene_model: str,
    *,
    mode: str = "public_test",
    instance_ids: Iterable[int] | None = None,
    limit: int = 10,
) -> dict[str, object]:
    """Validate the exact template and TRO files that a task launch may use."""
    from .challenge_tasks import challenge_task_by_name

    mode = normalize_mode(mode)
    catalog = validate_task_catalog(data_path)
    tasks = dict(catalog["tasks"])
    task_ids_by_name = {name: task_id for task_id, name in tasks.items()}
    if task_name not in task_ids_by_name:
        raise ValueError(f"unknown BEHAVIOR 2026 task: {task_name!r}")
    task_meta = challenge_task_by_name(task_name)
    expected_scene = None if task_meta is None else str(task_meta["scene"])
    if expected_scene != scene_model:
        raise ValueError(
            f"scene mismatch for BEHAVIOR 2026 task id={task_ids_by_name[task_name]} "
            f"{task_name}: expected {expected_scene}, got {scene_model}"
        )

    ids = list(eval_instance_ids(mode, limit=limit) if instance_ids is None else instance_ids)
    template_mode = mode_for_instance(ids[0], mode) if len(ids) == 1 else mode
    selected_template = template_path(data_path, task_name, scene_model, template_mode)
    missing = [] if os.path.isfile(selected_template) else [selected_template]

    selected_tros: list[str] = []
    for instance_id in ids:
        instance_mode = mode_for_instance(int(instance_id), mode)
        path = tro_path(data_path, task_name, scene_model, int(instance_id), instance_mode)
        selected_tros.append(path)
        if not os.path.isfile(path):
            missing.append(path)

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:6])
        remainder = len(missing) - min(len(missing), 6)
        suffix = f"\n  - ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            "BEHAVIOR 2026 task assets are incomplete; refusing to start or switch "
            f"task={task_name} scene={scene_model} mode={mode}.\n{preview}{suffix}"
        )

    return {
        "root": catalog["root"],
        "mode": mode,
        "task_id": task_ids_by_name[task_name],
        "task": task_name,
        "scene": scene_model,
        "template": selected_template,
        "instance_ids": ids,
        "tro_paths": selected_tros,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--scene")
    parser.add_argument("--mode", default=os.environ.get("BEHAVIOR_CHALLENGE_MODE", "public_test"))
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.catalog_only:
        result = validate_task_catalog(args.data_path)
        print(
            "challenge catalog ready: "
            f"tasks={result['count']} "
            f"id13={result['tasks'][13]} id17={result['tasks'][17]}"
        )
        return 0
    if not args.task or not args.scene:
        parser.error("--task and --scene are required unless --catalog-only is used")
    instance_ids = None if args.instance_id is None else [args.instance_id]
    result = validate_task_assets(
        args.data_path,
        args.task,
        args.scene,
        mode=args.mode,
        instance_ids=instance_ids,
        limit=args.limit,
    )
    print(
        "challenge data ready: "
        f"task={result['task']} scene={result['scene']} mode={result['mode']} "
        f"instances={result['instance_ids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
