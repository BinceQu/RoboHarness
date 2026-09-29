"""任务表、归档使用的 Challenge 2025 ×2 预算、以及 public instance 抽样。"""

from __future__ import annotations

import random
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path

# 与 eval_utils.TASK_INDICES_TO_NAMES / scripts/build_task_queues.py 一致。
TASK_NAMES: dict[int, str] = {
    0: "turning_on_radio",
    1: "picking_up_trash",
    2: "putting_away_Halloween_decorations",
    3: "cleaning_up_plates_and_food",
    4: "can_meat",
    5: "setting_mousetraps",
    6: "hiding_Easter_eggs",
    7: "picking_up_toys",
    8: "rearranging_kitchen_furniture",
    9: "putting_up_Christmas_decorations_inside",
    10: "set_up_a_coffee_station_in_your_kitchen",
    11: "putting_dishes_away_after_cleaning",
    12: "preparing_lunch_box",
    13: "loading_the_car",
    14: "carrying_in_groceries",
    15: "bringing_in_wood",
    16: "moving_boxes_to_storage",
    17: "bringing_water",
    18: "tidying_bedroom",
    19: "outfit_a_basic_toolbox",
    20: "sorting_vegetables",
    21: "collecting_childrens_toys",
    22: "putting_shoes_on_rack",
    23: "boxing_books_up_for_storage",
    24: "storing_food",
    25: "clearing_food_from_table_into_fridge",
    26: "assembling_gift_baskets",
    27: "sorting_household_items",
    28: "getting_organized_for_work",
    29: "clean_up_your_desk",
    30: "setting_the_fire",
    31: "clean_boxing_gloves",
    32: "wash_a_baseball_cap",
    33: "wash_dog_toys",
    34: "hanging_pictures",
    35: "attach_a_camera_to_a_tripod",
    36: "clean_a_patio",
    37: "clean_a_trumpet",
    38: "spraying_for_bugs",
    39: "spraying_fruit_trees",
    40: "make_microwave_popcorn",
    41: "cook_cabbage",
    42: "chop_an_onion",
    43: "slicing_vegetables",
    44: "chopping_wood",
    45: "cook_hot_dogs",
    46: "cook_bacon",
    47: "freeze_pies",
    48: "canning_food",
    49: "make_pizza",
}

# 归档原始的 Challenge 2025 预算，已经包含 ×2 和取整。
# 不得从 2026 的统计或 ×1.5 预算反推这些值。
TIMEOUT_STEPS_X2: dict[int, int] = {
    0: 4299,
    1: 10535,
    2: 27664,
    3: 27392,
    4: 23694,
    5: 20343,
    6: 15239,
    7: 37781,
    8: 17886,
    9: 27437,
    10: 12532,
    11: 21906,
    12: 16490,
    13: 38452,
    14: 28549,
    15: 27071,
    16: 29192,
    17: 18877,
    18: 22074,
    19: 21275,
    20: 23807,
    21: 38372,
    22: 15384,
    23: 48455,
    24: 39738,
    25: 26136,
    26: 52120,
    27: 31615,
    28: 31342,
    29: 42857,
    30: 18236,
    31: 16470,
    32: 16698,
    33: 22445,
    34: 4780,
    35: 7823,
    36: 24141,
    37: 10584,
    38: 12958,
    39: 16691,
    40: 6475,
    41: 28235,
    42: 12799,
    43: 29689,
    44: 21503,
    45: 18289,
    46: 15359,
    47: 24910,
    48: 45950,
    49: 38373,
}

# Use the 2026 dataset's original mean, exactly as v3.9.3 Evaluator.load_env
# does. Scaling an already-truncated 2025 limit loses fractional steps and
# also ignores changed human statistics (20 of our 50 tasks differ).
_HUMAN_STATS = json.loads(
    (Path(__file__).parent / 'data' / 'challenge_2026_human_stats.json').read_text()
)
_HUMAN_ROWS = {int(row['task_index']): row for row in _HUMAN_STATS['tasks']}
for _task_id, _task_name in TASK_NAMES.items():
    if _HUMAN_ROWS[_task_id]['task_name'] != _task_name:
        raise ValueError(f'2026 human statistics task identity mismatch: {_task_id}')
TIMEOUT_STEPS_X15: dict[int, int] = {
    task_id: int(_HUMAN_ROWS[task_id]['length'] * 1.5)
    for task_id in TASK_NAMES
}
# RoboHarness reproduces the archived 2025 protocol by default. The explicitly
# named 2026 lookup remains available only for auditing other source records.
TIMEOUT_STEPS: dict[int, int] = TIMEOUT_STEPS_X2

# 官方文档：报榜用 public_test slot 0–9。槽位 i 对应真实 ID 301+i。
PUBLIC_REPORT_SLOTS: tuple[int, ...] = tuple(range(10))
PUBLIC_INSTANCE_ID_BASE = 301
DEFAULT_SAMPLE_SEED = 20260911
DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompt"
DEFAULT_CLAUDE_ROOT = Path(__file__).resolve().parents[2] / "harness/claude_code"
DEFAULT_MODEL_HOST = "127.0.0.1"
DEFAULT_MODEL_PORT = 31000
DEFAULT_MODEL_NAME = "Qwen3.8-Flash-Next-FP8"
FORBIDDEN_MODEL_PORTS = frozenset({30000})
FORBIDDEN_HTTP_PORTS = frozenset({15050})
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TASK_TOKEN_RE = re.compile(r"^(?:task[_-]?)?(\d{1,2})$", re.IGNORECASE)
_NAME_BY_LOWER = {name.lower(): idx for idx, name in TASK_NAMES.items()}


@dataclass(frozen=True)
class TaskSpec:
    index: int
    name: str
    timeout_steps: int
    scene: str


def challenge_2026_max_ticks(
    task_id: int | str | None = None,
    task_name: str | None = None,
    port: int | str | None = None,
) -> int | None:
    """Challenge 2026 官方超时：mean_demo_length × 1.5。

    查不到返回 ``None``，禁止用一个猜测的固定步数替代官方任务表。
    """
    if task_id is not None and str(task_id).strip() != "":
        try:
            return int(TIMEOUT_STEPS_X15[int(task_id)])
        except (KeyError, TypeError, ValueError):
            pass
    key = str(task_name or "").replace("-", "_").strip().lower()
    if key:
        index = _NAME_BY_LOWER.get(key)
        if index is not None:
            return int(TIMEOUT_STEPS_X15[index])
    if port is not None and str(port).strip() != "":
        try:
            http_port = int(port)
        except (TypeError, ValueError):
            http_port = -1
        mapped = mapped_task_for_port(http_port)
        if mapped is not None:
            return int(TIMEOUT_STEPS_X15[mapped])
    return None


def challenge_2025_max_ticks(
    task_id: int | str | None = None,
    task_name: str | None = None,
    port: int | str | None = None,
) -> int | None:
    """Archived Challenge 2025 budget (original mean demonstration length ×2)."""
    if task_id is not None and str(task_id).strip() != "":
        try:
            return int(TIMEOUT_STEPS_X2[int(task_id)])
        except (KeyError, TypeError, ValueError):
            pass
    key = str(task_name or "").replace("-", "_").strip().lower()
    if key:
        index = _NAME_BY_LOWER.get(key)
        if index is not None:
            return int(TIMEOUT_STEPS_X2[index])
    if port is not None and str(port).strip() != "":
        try:
            http_port = int(port)
        except (TypeError, ValueError):
            http_port = -1
        mapped = mapped_task_for_port(http_port)
        if mapped is not None:
            return int(TIMEOUT_STEPS_X2[mapped])
    return None


def resolve_task(raw: str) -> TaskSpec:
    """接受 ``0`` / ``task00`` / ``turning_on_radio``。"""

    token = str(raw or "").strip()
    if not token:
        raise ValueError("task 不能为空")
    match = _TASK_TOKEN_RE.fullmatch(token.replace(" ", "_"))
    if match:
        index = int(match.group(1))
    else:
        key = token.replace("-", "_").lower()
        if key not in _NAME_BY_LOWER:
            raise ValueError(f"未知任务: {raw}")
        index = _NAME_BY_LOWER[key]
    if index not in TASK_NAMES:
        raise ValueError(f"任务序号必须是 0–49: {index}")
    name = TASK_NAMES[index]
    return TaskSpec(
        index=index,
        name=name,
        timeout_steps=TIMEOUT_STEPS[index],
        scene=_task_scene(name),
    )


def _task_scene(task_name: str) -> str:
    try:
        from behavior_interface.challenge_tasks import challenge_task_scene
    except Exception:
        return "house_double_floor_lower"
    scene = challenge_task_scene(task_name)
    return str(scene or "house_double_floor_lower")


def instance_id_for_slot(slot: int) -> int:
    if slot not in PUBLIC_REPORT_SLOTS:
        raise ValueError(f"报榜 public slot 必须是 0–9: {slot}")
    return PUBLIC_INSTANCE_ID_BASE + int(slot)


def parse_instance_slots(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    slots: list[int] = []
    for part in str(raw).replace(",", " ").split():
        slot = int(part)
        if slot not in PUBLIC_REPORT_SLOTS:
            raise ValueError(f"instance slot 必须是 0–9: {slot}")
        if slot in slots:
            raise ValueError(f"重复 instance slot: {slot}")
        slots.append(slot)
    return slots


def sample_instance_slots(count: int, *, seed: int = DEFAULT_SAMPLE_SEED) -> list[int]:
    if count < 1 or count > len(PUBLIC_REPORT_SLOTS):
        raise ValueError("num-instances 必须在 1–10")
    chosen = random.Random(int(seed)).sample(list(PUBLIC_REPORT_SLOTS), count)
    return sorted(chosen)


def official_policy_port(http_port: int) -> int:
    port = int(http_port)
    if 15010 <= port <= 15045:
        # Keep challenge policy sockets away from common local Docker services
        # that already occupy 18011/18012 on the evaluation host.
        return 28010 + (port - 15010)
    return 18081 + (port - 15060)


def idle_gate_port(http_port: int) -> int:
    return official_policy_port(http_port) + 1000


# BEHAVIOR 2026 challenge ports: task10–45 use 15010–15045, three tasks per
# GPU; the legacy public ports 15060–15069 remain available for the old set.
OFFICIAL_PORT_TASK: dict[int, int] = {
    **{15010 + index: 10 + index for index in range(36)},
    **{15060 + index: index for index in range(10)},
}
OFFICIAL_PORT_GPU: dict[int, int] = {
    15060: 0,
    15060: 0,
    15061: 0,
    15062: 0,
    15063: 1,
    15064: 1,
    15065: 1,
    15066: 2,
    15067: 2,
    15068: 2,
    15069: 3,
}
OFFICIAL_PORT_GPU.update({
    15010 + index: index // 3
    for index in range(18)
})
OFFICIAL_PORT_GPU.update({
    15028 + index: index // 3
    for index in range(18)
})


def official_port_for_task(task_index: int) -> int | None:
    index = int(task_index)
    if 0 <= index <= 9:
        return 15060 + index
    if 10 <= index <= 45:
        return 15010 + (index - 10)
    return None


def mapped_gpu_for_port(http_port: int) -> int | None:
    return OFFICIAL_PORT_GPU.get(int(http_port))


def mapped_task_for_port(http_port: int) -> int | None:
    if os.environ.get('ROBOHARNESS_HTTP_PORT') == str(int(http_port)):
        task = int(os.environ['ROBOHARNESS_TASK_ID'])
        if task not in TASK_NAMES:
            raise ValueError('Unknown RoboHarness task')
        return task
    return OFFICIAL_PORT_TASK.get(int(http_port))


def validate_port_gpu(http_port: int, gpu: int) -> None:
    if int(http_port) in FORBIDDEN_HTTP_PORTS:
        raise ValueError("禁止操作 15050")
    if not (1024 <= int(http_port) <= 65535):
        raise ValueError(f"非法 HTTP 端口: {http_port}")
    if int(gpu) < 0:
        raise ValueError(f"非法 GPU: {gpu}")
    mapped = mapped_gpu_for_port(http_port)
    if mapped is not None and mapped != int(gpu):
        raise ValueError(
            f"官方口 {http_port} 固定 GPU{mapped}，不能用 GPU{gpu}"
        )


def validate_port_task(http_port: int, task_index: int) -> None:
    mapped = mapped_task_for_port(http_port)
    if mapped is not None and mapped != int(task_index):
        name = TASK_NAMES[mapped]
        raise ValueError(
            f"官方口 {http_port} 固定 task {mapped:02d} {name}，"
            f"不能跑 task {int(task_index):02d}"
        )


def validate_model_endpoint(host: str, port: int, model: str) -> str:
    if int(port) in FORBIDDEN_MODEL_PORTS:
        raise ValueError("禁止 :30000 / Qwen3.8-27B")
    if "27b" in str(model).lower() or "27B" in str(model):
        raise ValueError("禁止 Qwen3.8-27B")
    if host.strip() in {"10.130.140.45"}:
        raise ValueError("不要用 10.130.140.45:31000")
    return f"http://{host.strip()}:{int(port)}"


def make_session_id(task_index: int, http_port: int, slot: int, stamp: str) -> str:
    session_id = f"t{task_index:02d}p{http_port}i{slot}-{stamp}"
    if len(session_id) > 64 or not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"session_id 非法或超长: {session_id}")
    return session_id


def latest_prompt_path(
    task: TaskSpec,
    *,
    prompt_root: Path = DEFAULT_PROMPT_ROOT,
    harness: str = "embodied_claude_code",
    model_slug: str = "qwen38-flash-next-fp8",
) -> Path:
    folder = prompt_root / f"{task.index:02d}_{task.name}"
    pattern = f"{harness}_{model_slug}_v*.txt"
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"找不到 prompt: {folder}/{pattern}")

    def version_key(path: Path) -> tuple[int, int, str]:
        match = re.search(r"_v(\d+)", path.name)
        version = int(match.group(1)) if match else -1
        # 同版本号取文件名更大的，通常是带备注的新稿，例如 v17_now_eef。
        extra = 1 if re.search(r"_v\d+_.+\.txt$", path.name) else 0
        return (version, extra, path.name)

    return max(matches, key=version_key)


def result_json_name(task_name: str, instance_id: int, rollout_id: int = 0) -> str:
    return f"{task_name}_{instance_id}_{rollout_id}.json"
