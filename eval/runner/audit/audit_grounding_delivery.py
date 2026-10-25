"""Audit actual native user-message image bytes without making model calls."""
import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from tools.util.files import atomic_json


def image_blocks(value):
    if isinstance(value, dict):
        if value.get("type") == "image":
            yield value
        for item in value.values():
            yield from image_blocks(item)
    elif isinstance(value, list):
        for item in value:
            yield from image_blocks(item)


def audit(folder):
    report = json.loads((folder / "report.json").read_text())
    rows = []
    for case in report["cases"]:
        images = []
        projects = folder / case["id"] / "private/claude-home/projects"
        for path in projects.rglob("*.jsonl"):
            for number, line in enumerate(path.open(), 1):
                event = json.loads(line)
                # toolUseResult is an additional stored copy, not another
                # image-bearing message. Audit only conversation content.
                if event.get("type") != "user":
                    continue
                for block in image_blocks(event.get("message", {})):
                    source = block.get("source", {})
                    assert source.get("type") == "base64"
                    data = base64.b64decode(source["data"], validate=True)
                    with Image.open(BytesIO(data)) as image:
                        size = list(image.size)
                    images.append({"sha256": sha256(data).hexdigest(), "size": size,
                        "native_transcript": str(path.relative_to(folder)), "line": number})
        expected = case.get("delivered_sha256", case["source_sha256"])
        passed = len(images) == 1 and images[0]["sha256"] == expected and images[0]["size"] == [720, 720]
        rows.append({"id": case["id"], "passed": passed, "images": images})
    result = {"scope": "image content in native Claude conversation user messages",
        "total": len(rows), "passed": sum(row["passed"] for row in rows), "cases": rows}
    atomic_json(folder / "delivery-audit.json", result)
    print(json.dumps({key: result[key] for key in ("scope", "total", "passed")}))
    return all(row["passed"] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    raise SystemExit(0 if audit(parser.parse_args().folder.resolve()) else 1)
