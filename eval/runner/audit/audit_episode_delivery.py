"""Read-only audit of workspace access and images in native conversations."""
import argparse
import base64
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from audit_grounding_delivery import image_blocks
from tools.util.files import atomic_json


def text_blocks(value):
    if isinstance(value, dict):
        if value.get("type") == "text":
            yield value.get("text", "")
        for item in value.values():
            yield from text_blocks(item)
    elif isinstance(value, list):
        for item in value:
            yield from text_blocks(item)


def audit(case):
    calls, reads, delivered, all_images = Counter(), [], {}, []
    read_images = {}
    native = case / "workspace/.runtime/claude-home/projects"
    if not native.exists():
        native = case / "private/claude-home/projects"  # historical evidence
    for path in sorted(native.rglob("*.jsonl")):
        read_targets = {}
        for number, line in enumerate(path.open(), 1):
            if not line.endswith("\n"):
                continue  # A live Claude process may still be writing this row.
            event = json.loads(line)
            content = event.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if event.get("type") == "assistant" and block.get("type") == "tool_use":
                    name, arguments = block["name"], block.get("input", {})
                    calls[name] += 1
                    if name == "Read" and isinstance(arguments.get("file_path"), str):
                        target = Path(arguments["file_path"])
                        if not target.is_absolute():
                            target = Path(event.get("cwd", str(case / "workspace"))) / target
                        read_targets[block["id"]] = str(target.resolve())
                    if name in {"Read", "Skill"} or (name == "Bash" and any(
                        term in arguments.get("command", "") for term in
                        ("cat ", "sed ", "head ", "tail ", "grep ", "rg ", "--schema"))):
                        reads.append({"tool": name, "input": arguments, "line": number,
                                      "transcript": str(path.relative_to(case))})
                if event.get("type") != "user" or block.get("type") != "tool_result":
                    continue
                hashes = [sha256(base64.b64decode(image["source"]["data"], validate=True)).hexdigest()
                          for image in image_blocks(block) if image.get("source", {}).get("type") == "base64"]
                all_images.extend(hashes)
                read_target = read_targets.get(block.get("tool_use_id"))
                if read_target is not None:
                    read_images.setdefault(read_target, set()).update(hashes)
                for text in text_blocks(block):
                    try:
                        receipt = json.loads(text)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(receipt, dict) and isinstance(receipt.get("record"), str):
                        record = Path(receipt["record"])
                        if not record.is_absolute():
                            record = case / "workspace" / record
                        delivered.setdefault(str(record.resolve()), set()).update(hashes)
    rows = []
    for path in sorted((case / "workspace/tool_call_history").glob("*/result.json")):
        receipt = json.loads(path.read_text())
        expected = [image["sha256"] for image in receipt.get("images", [])]
        if not expected or path.parent.name == "000001_capture_head_camera":
            continue  # Initial senses are published before Claude is launched.
        actual = delivered.get(str(path.resolve()), set())
        direct = all(value in actual for value in expected)
        # A failed Bash shell cannot replace its output with image blocks.
        # Its failure hook asks for native Read. Count that recovery only when
        # the explicit Read path AND returned bytes match this exact receipt.
        recovered = all(
            media["sha256"] in actual or media["sha256"] in read_images.get(
                str(Path(media["path"]).resolve()), set())
            for media in receipt["images"])
        rows.append({"record": str(path.relative_to(case)), "images": len(expected),
                     "all_present_in_same_native_tool_result": direct,
                     "all_present_in_native_images": recovered,
                     "native_read_recovery": recovered and not direct})
    result = {"scope": "exact receipt image hashes in the matching native tool_result or an explicit Read of the same image path",
        "native_tool_calls": dict(calls), "workspace_reads": reads,
        "native_image_blocks": len(all_images), "records": rows,
        "total_image_results": len(rows),
        "verified_image_results": sum(row["all_present_in_same_native_tool_result"] for row in rows),
        "native_read_recovery_results": sum(row["native_read_recovery"] for row in rows),
        "verified_including_native_read_results": sum(row["all_present_in_native_images"] for row in rows)}
    atomic_json(case / "native-delivery-audit.json", result)
    return {"instance": case.name, **{key: result[key] for key in
        ("native_tool_calls", "native_image_blocks", "total_image_results", "verified_image_results",
         "native_read_recovery_results", "verified_including_native_read_results")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(case) for case in sorted(args.run.resolve().glob("instance_*"))], indent=2))
