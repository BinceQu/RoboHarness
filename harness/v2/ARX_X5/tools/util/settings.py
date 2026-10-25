"""All session data remains inside this workspace."""

import json
from pathlib import Path


def settings():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / ".session.json").read_text())
    config.update(workspace=str(root), state_dir=str(root / ".runtime"))
    return config


def initialize_state(root):
    root = Path(root).resolve()
    for name in [
        ".runtime",
        "tool_call_history",
        "perception/proprioception",
        "perception/exteroception",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)
    path = root / ".runtime/state.json"
    if not path.exists():
        path.write_text(
            json.dumps({"counter": 0, "latest_image_id": None, "active_skill": None})
            + "\n"
        )


def write_json(path, value):
    from tools.util.geometry import json_ready

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n"
    )
    temporary.replace(path)
