#!/usr/bin/env python3
"""Build the static project page from the checked-in experiment manifests."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/BinceQu/RoboHarness"
INSTANCES = [301, 304, 306, 308, 310]


def recorded_rollouts():
    manifest = json.loads((ROOT / "website/rollouts.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported recording manifest")
    recordings = manifest["recordings"]
    assets = set()
    for task_id, recording in recordings.items():
        if recording["q"] is not None:
            candidates = [row["q"] for row in recording["candidates"] if row["q"] is not None]
            if not candidates or recording["q"] != max(candidates):
                raise ValueError(f"Selected recording is not the highest scored: {task_id}")
            result = recording["evaluatorResult"]
            if result["instance_id"] != recording["instance"] or result["q_score"]["final"] != recording["q"]:
                raise ValueError(f"Recording/result mismatch: {task_id}")
        elif recording["evaluatorResult"] is not None:
            raise ValueError(f"Unscored recording has an evaluator result: {task_id}")
        for key in ("video", "poster", "trace"):
            relative = Path(recording[key])
            path = (ROOT / "docs" / relative).resolve()
            if not path.is_relative_to(ROOT / "docs/assets/rollouts"):
                raise ValueError(f"Invalid recording asset path: {relative}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != recording[key + "Sha256"]:
                raise ValueError(f"Recording asset hash mismatch: {relative}")
            assets.add(relative)
        trace = json.loads((ROOT / "docs" / recording["trace"]).read_text())
        if trace["task"] != task_id or trace["run"] != recording["run"] or trace["instance"] != recording["instance"]:
            raise ValueError(f"Recording/trace mismatch: {task_id}")
        times = [step["t"] for step in trace["steps"]]
        if not times or times != sorted(set(times)) or times[0] < 0 or times[-1] >= recording["duration"]:
            raise ValueError(f"Invalid trace timeline: {task_id}")
    return recordings, assets


def archive_data():
    tasks = []
    for path in sorted((ROOT / "tasks").glob("task*.json")):
        task = json.loads(path.read_text())
        cases = sorted(task["cases"], key=lambda case: case["instance_id"])
        if [case["instance_id"] for case in cases] != INSTANCES:
            raise ValueError(f"Unexpected instance selection: {path.name}")
        if task["challenge_year"] != 2025 or task["budget_multiplier"] != 2:
            raise ValueError(f"Unexpected evaluation protocol: {path.name}")
        prompts = []
        for case in cases:
            prompt = ROOT / case["prompt"]
            if hashlib.sha256(prompt.read_bytes()).hexdigest() != case["prompt_sha256"]:
                raise ValueError(f"Prompt hash mismatch: {case['prompt']}")
            prompts.append({"instance": case["instance_id"], "path": case["prompt"],
                            "sha256": case["prompt_sha256"]})
        tasks.append({
            "id": task["task"], "name": task["task_name"].replace("_", " "),
            "mean": task["archive_reported_mean_q"], "maxSteps": task["max_steps"],
            "cases": [{"instance": case["instance_id"], "slot": case["slot"],
                       "q": case["archive_reported_q"]} for case in cases],
            "prompts": prompts,
        })
    return tasks


def archive_rows(tasks):
    rows = []
    for task in tasks:
        scores = "".join(f'<td>{case["q"]:.4f}</td>' for case in task["cases"])
        rows.append(
            f'<tr><th scope="row"><a href="{REPO}/blob/main/tasks/{task["id"]}.json">'
            f'{html.escape(task["name"].capitalize())}</a></th>{scores}'
            f'<td class="mean-cell">{task["mean"]:.4f}</td></tr>'
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    output = args.output.resolve()
    # Do not overwrite source directories, including an accidentally selected root.
    if output == ROOT or output == ROOT / "website" or ROOT.is_relative_to(output):
        parser.error("choose a separate generated-output directory")
    source = ROOT / "website"
    tasks = archive_data()
    recordings, media_assets = recorded_rollouts()
    if not set(recordings).issubset(task["id"] for task in tasks):
        raise ValueError("Recording manifest refers to an unknown task")
    for task in tasks:
        if task["id"] in recordings:
            task["rollout"] = {key: recordings[task["id"]][key] for key in (
                "run", "instance", "q", "label", "video", "poster", "trace", "duration",
                "defaultPlaybackRate", "missingTailSeconds")}
    options = "\n".join(
        f'<option value="{task["id"]}"'
        f'{" selected" if task["id"] == "task01" else ""}>'
        f'{html.escape(task["name"].capitalize())}</option>' for task in tasks
    )
    replacements = {
        "{{ARCHIVE_ROWS}}": archive_rows(tasks), "{{TASK_OPTIONS}}": options,
        "{{TASK_DATA}}": json.dumps(tasks, separators=(",", ":")).replace("<", "\\u003c"),
    }
    page = (source / "index.html").read_text()
    for marker, value in replacements.items():
        if marker not in page:
            raise ValueError(f"Missing template marker: {marker}")
        page = page.replace(marker, value)
    if "{{" in page:
        raise ValueError("Unresolved page template marker")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page)
    for asset in ("styles.css", "app.js"):
        shutil.copyfile(source / asset, output / asset)
    for asset in (ROOT / "docs" / "assets").iterdir():
        if asset.suffix in (".png", ".pdf", ".svg"):
            shutil.copyfile(asset, output / "assets" / asset.name)
    for relative in sorted(media_assets):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs" / relative, target)
    shutil.copyfile(source / "rollouts.json", output / "assets/rollouts/manifest.json")
    shutil.copyfile(ROOT / "docs/assets/roboharness-icon.svg", output / "favicon.svg")
    (output / ".nojekyll").touch()
    print(f"Built {output}: {len(tasks)} tasks, {len(recordings)} recorded head-camera rollouts")


if __name__ == "__main__":
    main()
