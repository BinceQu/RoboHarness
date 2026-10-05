#!/usr/bin/env python3
"""Build the static project page from the checked-in experiment manifests."""

import argparse
import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/cbq349/RoboHarness"
INSTANCES = [301, 304, 306, 308, 310]
REPORT = "validation_results/gpu5-20260930-r5/report.json"


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
            f'<span class="task-id">{task["id"]}</span>'
            f'{html.escape(task["name"])}</a></th>{scores}'
            f'<td class="mean-cell">{task["mean"]:.4f}</td></tr>'
        )
    return "\n".join(rows)


def validation_rows():
    report = json.loads((ROOT / REPORT).read_text())
    rows = []
    # Only normal finishes with an official score count in this partial-mean view.
    for run in report["runs"]:
        cases = [case for case in run["cases"]
                 if case.get("finish_reason") in ("model_done", "evaluator_end")
                 and case.get("result_sha256") and case.get("q") is not None]
        mean = f'{sum(case["q"] for case in cases) / len(cases):.4f}' if cases else "—"
        rows.append(f'<tr><th scope="row">{html.escape(run["task"])}</th>'
                    f'<td>{len(cases)}/{run["n_expected"]}</td><td>{mean}</td>'
                    f'<td>{run["archive_mean_q"]:.4f}</td></tr>')
    stamp = datetime.fromisoformat(report["updated_at_utc"]).astimezone(timezone.utc)
    return "\n".join(rows), stamp.strftime("%d %b %Y · %H:%M UTC")


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
    rows, stamp = validation_rows()
    options = "\n".join(
        f'<option value="{task["id"]}"'
        f'{" selected" if task["id"] == "task01" else ""}>'
        f'{task["id"]} · {html.escape(task["name"])}</option>' for task in tasks
    )
    replacements = {
        "{{ARCHIVE_ROWS}}": archive_rows(tasks), "{{TASK_OPTIONS}}": options,
        "{{TASK_DATA}}": json.dumps(tasks, separators=(",", ":")).replace("<", "\\u003c"),
        "{{VALIDATION_ROWS}}": rows, "{{VALIDATION_DATE}}": stamp,
        "{{TASK_COUNT}}": str(len(tasks)),
        "{{CASE_COUNT}}": str(sum(len(task["cases"]) for task in tasks)),
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
    shutil.copyfile(ROOT / "docs/assets/roboharness-icon.svg", output / "favicon.svg")
    (output / ".nojekyll").touch()
    print(f"Built {output}: {len(tasks)} tasks, {len(tasks) * 5} archived cases")


if __name__ == "__main__":
    main()
