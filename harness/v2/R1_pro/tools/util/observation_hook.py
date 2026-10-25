"""Attach saved robot observations through native Claude Bash output.

PreToolUse records the observation counter. PostToolUse preserves unrelated
Bash output and delivers each robot receipt once, with its exact image bytes as native
content blocks. No model/robot call, resizing or conversation rewrite occurs.
The ordinary Python/CLI JSON interface remains usable by any other agent.
"""
from __future__ import annotations

import base64
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

from .settings import settings
from .files import atomic_json, load_json
from .receipt import camera_observation_ready, model_receipt


def binding(event, env):
    config = settings()
    state_dir = Path(config["state_dir"])
    state = json.loads((state_dir / "state.json").read_text())
    counter = state.get("counter", 0)
    if type(counter) is not int or counter < 0:
        raise ValueError("invalid observation counter")
    identifiers = [event.get("session_id", ""), event.get("tool_use_id", "")]
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value)
           for value in identifiers):
        raise ValueError("invalid native tool/session identifier")
    pending = state_dir / "native-delivery" / identifiers[0] / (identifiers[1] + ".json")
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
    return {"type": "image", "source": {"type": "base64", "media_type": mime,
                                          "data": base64.b64encode(raw).decode()}}


def original_blocks(response):
    if response.get("structuredContent"):
        return list(response["structuredContent"])
    stdout, stderr = response.get("stdout", ""), response.get("stderr", "")
    blocks = []
    if response.get("isImage"):
        match = re.fullmatch(r"data:(image/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)", stdout.strip())
        if match:
            base64.b64decode(match[2], validate=True)
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": match[1], "data": match[2]}})
        elif stdout:
            blocks.append({"type": "text", "text": stdout})
    elif stdout:
        blocks.append({"type": "text", "text": stdout})
    if stderr:
        blocks.append({"type": "text", "text": stderr})
    return blocks


def without_receipt_echoes(blocks, receipts, workspace):
    """Remove only complete JSON values equal to an immutable or public receipt.

    Preserve shell warnings, failures, filtered fragments and arbitrary output.
    The caller emits each verified receipt once beside its image, including when
    stdout contains compiler warnings, multiple results or redirected output.
    """
    known = [value for _, _, receipt in receipts
             for value in (receipt, model_receipt(receipt, workspace))]
    decoder = json.JSONDecoder()
    kept = []
    for block in blocks:
        if block.get("type") != "text":
            kept.append(block)
            continue
        text = block["text"]
        pieces, offset, start = [], 0, 0
        while (offset := text.find("{", offset)) >= 0:
            try:
                value, length = decoder.raw_decode(text[offset:])
            except ValueError:
                offset += 1
                continue
            end = offset + length
            if isinstance(value, dict) and value in known:
                pieces.append(text[start:offset])
                start = end
            offset = end
        if start:
            text = "".join(pieces) + text[start:]
        if text.strip():
            kept.append({**block, "text": text})
    return kept


def repeated_capture_notice(session, counter):
    """Explain unchanged physics from receipts, without acting or judging success."""
    if counter < 3:
        return None
    budgets = []
    for number in range(counter - 2, counter + 1):
        records = list((session / "tool_call_history").glob(f"{number:06d}_*/result.json"))
        if len(records) != 1:
            return None
        receipt = json.loads(records[0].read_text())
        budget = receipt.get("rollout_budget") or {}
        if (not camera_observation_ready(receipt)
                or receipt.get("tool_name") not in {
                    "capture_head_camera", "capture_left_wrist_camera", "capture_right_wrist_camera"}
                or receipt.get("joint_commands_sent") != 0
                or budget.get("available") is not True
                or type(budget.get("used_ticks")) is not int
                or type(budget.get("total_ticks")) is not int
                or type(budget.get("remaining_ticks")) is not int
                or budget["remaining_ticks"] <= 0
                or not budget.get("episode_id") or budget.get("instance_id") is None):
            return None
        budgets.append(tuple(budget.get(key) for key in (
            "episode_id", "instance_id", "used_ticks", "total_ticks", "remaining_ticks")))
    if len(set(budgets)) != 1:
        return None
    return ("Three observation-only captures: no motion; ticks unchanged "
            f"({budgets[-1][2]}/{budgets[-1][3]}, {budgets[-1][4]} left). No keep-alive needed.")


def process(event, env=None):
    env = os.environ if env is None else env
    if event.get("tool_name") not in {"Bash", "TaskOutput"}:
        return {}
    if event.get("hook_event_name") not in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
        return {}
    _, pending, _, _ = binding(event, env)
    pending.parent.mkdir(parents=True, exist_ok=True)
    # Background Bash returns before its receipt exists. Track those holes
    # across tool calls instead of silently advancing past them forever.
    # Separate ledgers keep native sessions from consuming each other's output.
    with (pending.parent / "delivery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return process_locked(event, binding(event, env))


def process_locked(event, bound):
    session, pending, state, counter = bound
    kind = event["hook_event_name"]
    ledger_path = pending.parent / "receipt-delivery.json"
    ledger = load_json(ledger_path)
    if kind == "PreToolUse":
        if ledger is None:
            atomic_json(ledger_path, {"counter": counter, "pending": []})
        atomic_json(pending, {"counter": counter})
        return {}
    if ledger is None:
        # Compatibility with a PreToolUse marker from before a live upgrade.
        if not pending.is_file():
            return {}
        ledger = {"counter": load_json(pending)["counter"], "pending": []}
    before = ledger["counter"]
    if type(before) is not int or before < 0 or before > counter:
        raise ValueError("observation counter moved backwards")
    holes = ledger["pending"]
    if (not isinstance(holes, list)
            or any(type(n) is not int or not 1 <= n <= before for n in holes)):
        raise ValueError("invalid pending observation counters")
    receipts = []
    unfinished = []
    for number in sorted(set(holes) | set(range(before + 1, counter + 1))):
        matches = list((session / "tool_call_history").glob(f"{number:06d}_*/result.json"))
        if len(matches) > 1:
            raise ValueError("ambiguous history number")
        if not matches:
            unfinished.append(number)
            continue
        record = matches[0]
        if not record.is_file():
            unfinished.append(number)
            continue  # A killed command may leave an unfinished immutable record.
        receipt = json.loads(record.read_text())
        if receipt.get("images"):
            receipts.append((number, record, receipt))
    if not receipts:
        atomic_json(ledger_path, {"counter": counter, "pending": unfinished})
        return {}
    if kind == "PostToolUseFailure" or event['tool_name'] == 'TaskOutput':
        # Native TaskOutput discards structuredContent in updatedToolOutput.
        # Failure hooks cannot replace output either. Never mark undelivered
        # pixels as sent: request native Read and retain them for the next Bash.
        atomic_json(ledger_path, {"counter": counter,
                    "pending": sorted(unfinished + [n for n, _, _ in receipts])})
        paths = [media["path"] for _, _, receipt in receipts for media in receipt["images"]]
        timing = (" No keep-alive needed."
                  if receipts[-1][2].get("tool_name", "").startswith("capture_") else "")
        explanation = (
            "Command failed after observations were saved; the robot may have moved. "
            if kind == 'PostToolUseFailure' else
            "Background observations saved; TaskOutput has NOT displayed their pixels. "
        )
        return {"hookSpecificOutput": {"hookEventName": kind, "additionalContext": explanation +
            "Use native Read on these images before another visual decision: " + json.dumps(paths)
            + ". Full result: " + str(receipts[-1][1]) + timing}}
    response = dict(event["tool_response"])
    blocks = without_receipt_echoes(original_blocks(response), receipts, session)
    seen_hashes = {sha256(base64.b64decode(block["source"]["data"], validate=True)).hexdigest()
                   for block in blocks if block.get("type") == "image"
                   and block.get("source", {}).get("type") == "base64"}
    for _, record, receipt in receipts:
        public = model_receipt(receipt, session)
        # Actual pixels follow, so there is no need for CLI-only Read guidance.
        public.pop("observation_instruction", None)
        captions = [media.pop("caption", "") for media in public["images"]]
        blocks.append({"type": "text", "text": json.dumps(public, ensure_ascii=False,
                                                          allow_nan=False, separators=(",", ":"))})
        for media, caption in zip(receipt["images"], captions):
            # Always validate the file, including a Bash --image result already
            # present as image content, before asserting its identity.
            picture = image_content(media, record)
            if media["sha256"] not in seen_hashes:
                blocks.append(picture)
                seen_hashes.add(media["sha256"])
            if caption:
                blocks.append({"type": "text", "text": caption})
    current_skill = {"baseline": "behavior-v2-baseline", "active_skill": state.get("active_skill")}
    if (receipts[-1][2].get("latest_clickable_image_id") != state.get("latest_image_id", "")
            or receipts[-1][2].get("task_skill_state") != current_skill):
        # Late background results must not override a newer camera/skill state.
        blocks.append({"type": "text", "text": "Current image="
            + json.dumps(state.get("latest_image_id", "")) + "; skill="
            + json.dumps(current_skill.get("active_skill")) + ". Older images are historical; click only current head."})
    try:
        notice = repeated_capture_notice(session, counter)
    except (OSError, ValueError, TypeError, KeyError):
        notice = None  # An optional progress notice must never suppress images.
    if notice:
        blocks.append({"type": "text", "text": notice})
    response["structuredContent"] = blocks
    # Commit only after every attachment has passed its immutable hash check.
    atomic_json(ledger_path, {"counter": counter, "pending": unfinished})
    return {"hookSpecificOutput": {"hookEventName": kind, "updatedToolOutput": response}}


def main():
    event = json.load(sys.stdin)
    try:
        result = process(event)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        kind = event.get("hook_event_name")
        result = {"hookSpecificOutput": {"hookEventName": kind, "additionalContext": (
            f"Robot image attachment failed: {exc}. Original Bash output is unchanged. "
            "Read the latest saved image paths with native Read before another visual decision.")}}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
