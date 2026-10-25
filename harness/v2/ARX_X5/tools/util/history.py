"""Immutable inputs, actual joint commands, outputs and click crosshairs."""

import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw
from tools.util.settings import write_json
from tools.util.geometry import relative_uv_to_pixel, json_ready
from tools.util.receipt import without_ok


def clean(value):
    if isinstance(value, str) and value.startswith("data:"):
        return "<image saved separately and attached through native image content>"
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return json_ready(value)


class History:
    def __init__(self, root, tool, arguments):
        self.root = Path(root)
        self.state_file = self.root / ".runtime/state.json"
        state = json.loads(self.state_file.read_text())
        state["counter"] += 1
        self.path = self.root / "tool_call_history" / f"{state['counter']:06d}_{tool}"
        self.path.mkdir(parents=True)
        write_json(self.path / "input.json", {"tool": tool, "arguments": arguments})
        write_json(self.state_file, state)
        self.actions = []

    def action(self, action, sequence, episode):
        entry = {
            "action": action,
            "before_observation_sequence": sequence,
            "episode_id": episode,
        }
        self.actions.append(entry)
        with (self.path / "joint_commands.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")

    def crosshairs(self, arguments, snapshots, session):
        marks = []

        def walk(value, inherited=None):
            if isinstance(value, dict):
                image_id = value.get("image_id", inherited)
                if (
                    image_id
                    and value.get("u") is not None
                    and value.get("v") is not None
                ):
                    snapshot = snapshots.load(session, str(image_id))
                    width, height = snapshot.rgb.shape[1], snapshot.rgb.shape[0]
                    px, py = relative_uv_to_pixel(value["u"], value["v"], width, height)
                    marks.append((snapshot, px, py, value["u"], value["v"]))
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        walk(child, image_id)
            elif isinstance(value, list):
                for child in value:
                    walk(child, inherited)

        walk(arguments)
        groups = {}
        for snapshot, px, py, u, v in marks:
            groups.setdefault(snapshot.image_id, []).append((snapshot, px, py, u, v))
        entries = []
        for image_id, points in groups.items():
            im = Image.fromarray(points[0][0].rgb).convert("RGB")
            draw = ImageDraw.Draw(im)
            for index, (_, x, y, u, v) in enumerate(points):
                for width, color in [(5, "black"), (2, "#00ffff")]:
                    draw.line((x - 12, y, x + 12, y), fill=color, width=width)
                    draw.line((x, y - 12, x, y + 12), fill=color, width=width)
                draw.text(
                    (min(x + 14, im.width - 110), max(0, y - 18)),
                    f"{index + 1}: {u},{v}",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                entries.append(
                    {
                        "image_id": image_id,
                        "u": u,
                        "v": v,
                        "pixel_uv": [x, y],
                        "width": im.width,
                        "height": im.height,
                    }
                )
            im.save(self.path / f"{image_id}_clicks.png")
        write_json(self.path / "clicks.json", entries)

    def image(self, source, caption):
        source = Path(source).resolve()
        if not source.is_relative_to(self.root.resolve()):
            raise ValueError("Images must originate inside the current workspace")
        target = self.path / source.name
        if target != source:
            if target.exists() and target.read_bytes() != source.read_bytes():
                target = self.path / (
                    hashlib.sha256(str(source).encode()).hexdigest()[:8]
                    + "_"
                    + source.name
                )
            shutil.copy2(source, target)
        return {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "mime_type": "image/png" if target.suffix == ".png" else "image/jpeg",
            "caption": caption,
        }

    def finish(self, result):
        result = without_ok(clean(result))
        result["history"] = str(self.path)
        result["evaluator_action_count"] = len(self.actions)
        write_json(self.path / "result.json", result)
        return result
