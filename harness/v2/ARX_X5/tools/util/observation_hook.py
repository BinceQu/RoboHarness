"""Attach saved robot observations through native Claude Bash output.

PreToolUse records the observation counter. PostToolUse preserves Bash's
result and appends new receipt paths and exact image bytes as native
content blocks. No model/robot call, resizing or conversation rewrite occurs.
The ordinary Python/CLI JSON interface remains usable by any other agent.
"""

from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

from .settings import settings


def binding(event, env):
    config = settings()
    state_dir = Path(config["state_dir"])
    state = json.loads((state_dir / "state.json").read_text())
    counter = state.get("counter", 0)
    if type(counter) is not int or counter < 0:
        raise ValueError("invalid observation counter")
    identifiers = [event.get("session_id", ""), event.get("tool_use_id", "")]
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value)
        for value in identifiers
    ):
        raise ValueError("invalid native tool/session identifier")
    pending = (
        state_dir / "native-delivery" / identifiers[0] / (identifiers[1] + ".json")
    )
    return Path(config["workspace"]), pending, state, counter


def image_content(media, record):
    path = Path(media["path"]).resolve()
    if not path.is_relative_to(record.parent.resolve()):
        raise ValueError("observation image is outside its immutable record")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != media["sha256"]:
        raise ValueError("observation image hash differs from its receipt")
    mime = media["mime_type"]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError("unsupported observation image type")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": base64.b64encode(raw).decode(),
        },
    }


def original_blocks(response):
    if response.get("structuredContent"):
        return list(response["structuredContent"])
    stdout, stderr = response.get("stdout", ""), response.get("stderr", "")
    blocks = []
    if response.get("isImage"):
        match = re.fullmatch(
            r"data:(image/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)",
            stdout.strip(),
        )
        if match:
            base64.b64decode(match[2], validate=True)
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": match[1],
                        "data": match[2],
                    },
                }
            )
        elif stdout:
            blocks.append({"type": "text", "text": stdout})
    elif stdout:
        blocks.append({"type": "text", "text": stdout})
    if stderr:
        blocks.append({"type": "text", "text": stderr})
    return blocks


def process(event, env=None):
    env = os.environ if env is None else env
    if event.get("tool_name") != "Bash":
        return {}
    bound = binding(event, env)
    if bound is None:
        return {}
    session, pending, state, counter = bound
    kind = event["hook_event_name"]
    if kind == "PreToolUse":
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text(json.dumps({"counter": counter}))
        return {}
    if kind not in {"PostToolUse", "PostToolUseFailure"} or not pending.is_file():
        return {}
    before = json.loads(pending.read_text())["counter"]
    if type(before) is not int or before < 0 or before > counter:
        raise ValueError("observation counter moved backwards")
    receipts = []
    for number in range(before + 1, counter + 1):
        matches = list(
            (session / "tool_call_history").glob(f"{number:06d}_*/result.json")
        )
        if len(matches) > 1:
            raise ValueError("ambiguous history number")
        if not matches:
            continue
        record = matches[0]
        if not record.is_file():
            continue  # A killed command may leave an unfinished immutable record.
        receipt = json.loads(record.read_text())
        if receipt.get("images"):
            receipts.append((record, receipt))
    if not receipts:
        return {}
    if kind == "PostToolUseFailure":
        paths = [
            media["path"] for _, receipt in receipts for media in receipt["images"]
        ]
        return {
            "hookSpecificOutput": {
                "hookEventName": kind,
                "additionalContext": "The Bash command failed after robot observations were saved. The robot may have moved. "
                "Use native Read on these images before another visual decision: "
                + json.dumps(paths)
                + ". Full result: "
                + str(receipts[-1][0]),
            }
        }
    response = dict(event["tool_response"])
    blocks = original_blocks(response)
    try:
        stdout_json = json.loads(response.get("stdout", ""))
    except (ValueError, TypeError):
        stdout_json = None
    seen_hashes = {
        sha256(base64.b64decode(block["source"]["data"], validate=True)).hexdigest()
        for block in blocks
        if block.get("type") == "image"
        and block.get("source", {}).get("type") == "base64"
    }
    for record, receipt in receipts:
        if stdout_json != receipt:
            # Native Bash may have truncated or intentionally summarized a
            # large planner audit. Do not bypass its output limit by injecting
            # that full JSON again. The immutable receipt stays readable.
            brief = {key: receipt[key] for key in
                     ('tool', 'plan_id', 'recommended_arm', 'image_id', 'error')
                     if key in receipt and isinstance(receipt[key], (str, bool, int, float))}
            brief = {key: value[:2048] if isinstance(value, str) else value
                     for key, value in brief.items()}
            brief['full_result_path'] = str(record)
            blocks.append(
                {"type": "text", "text": json.dumps(brief, ensure_ascii=False)}
            )
        for media in receipt["images"]:
            # Always validate the file, including a Bash --image result already
            # present as image content, before asserting its identity.
            picture = image_content(media, record)
            blocks.append(
                {
                    "type": "text",
                    "text": "Saved robot observation: "
                    + media["path"]
                    + "\n"
                    + media.get("caption", ""),
                }
            )
            if media["sha256"] not in seen_hashes:
                blocks.append(picture)
                seen_hashes.add(media["sha256"])
    blocks.append(
        {
            "type": "text",
            "text": (
                "The saved images above are attached as actual image content. Inspect their pixels directly "
                "before the next visual decision. No extra Read is needed for these attached images. "
                "Current latest_head_image_id="
                + json.dumps(state.get("latest_image_id", ""))
                + "; current task_skill_state="
                + json.dumps(
                    {
                        "baseline": "robodojo-baseline",
                        "active_skill": state.get("active_skill"),
                    }
                )
                + "."
            ),
        }
    )
    response["structuredContent"] = blocks
    return {
        "hookSpecificOutput": {"hookEventName": kind, "updatedToolOutput": response}
    }


def main():
    event = json.load(sys.stdin)
    try:
        result = process(event)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        kind = event.get("hook_event_name")
        result = {
            "hookSpecificOutput": {
                "hookEventName": kind,
                "additionalContext": (
                    f"Robot image attachment failed: {exc}. Original Bash output is unchanged. "
                    "Read the latest saved image paths with native Read before another visual decision."
                ),
            }
        }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
