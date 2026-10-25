#!/usr/bin/env python3
"""Build the static project site: the portal at /, the paper page at /v1/ and the V2 page at /v2/.

The paper page is generated from the checked-in experiment manifests; the
portal and V2 pages are copied from website/portal and website/v2, with
{{ASSET:path}} markers resolved to locked, externally hosted files.
"""

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import fetch_assets
import method_video


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/BinceQu/RoboHarness"
INSTANCES = [301, 304, 306, 308, 310]


def media_source(relative, expected, media):
    """Return the page URL of a video hosted outside git, checking its locked hash."""
    lock = fetch_assets.lock(ROOT)
    row = fetch_assets.entry(lock, "docs/" + relative.as_posix())
    if row["sha256"] != expected:
        raise ValueError(f"Locked asset differs from the page manifest: {relative}")
    if media == "remote":
        return fetch_assets.url(lock, row)
    if not fetch_assets.verified(ROOT / row["path"], row):
        raise ValueError(f"Missing or changed local video; run scripts/fetch_assets.py --group media: {relative}")
    return f"{relative.as_posix()}?v={expected[:12]}"


# Static pages copied around the generated paper page, and their site folders.
STATIC_PAGES = (("portal", ""), ("v2", "v2"))


def published_path(path):
    """Site location of a locked repository file when videos are embedded locally."""
    if path.startswith("docs/assets/"):
        return "v1/" + path.removeprefix("docs/")
    if path.startswith("website/v2/"):
        return "v2/" + path.removeprefix("website/v2/")
    raise ValueError(f"No site location for asset: {path}")


def static_pages(output, media):
    lock = fetch_assets.lock(ROOT)
    embedded = set()
    for name, folder in STATIC_PAGES:
        source = ROOT / "website" / name
        prefix = "../" * len(Path(folder).parts)

        def asset(match):
            row = fetch_assets.entry(lock, match[1])
            if media == "remote":
                return html.escape(fetch_assets.url(lock, row))
            if not fetch_assets.verified(ROOT / row["path"], row):
                raise ValueError(f"Missing or changed local asset; run scripts/fetch_assets.py --group {row['group']}: {row['path']}")
            embedded.add(row["path"])
            return prefix + published_path(row["path"]) + "?v=" + row["sha256"][:12]

        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix == ".mp4":
                continue
            target = output / folder / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".html":
                page = re.sub(r"\{\{ASSET:([^}]+)\}\}", asset, path.read_text())
                if "{{" in page:
                    raise ValueError(f"Unresolved page template marker: {path}")
                target.write_text(page)
            else:
                shutil.copyfile(path, target)
    for path in sorted(embedded):
        target = output / published_path(path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)


def recorded_rollouts(media):
    manifest = json.loads((ROOT / "website/rollouts.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported recording manifest")
    recordings = manifest["recordings"]
    assets, sources = set(), {}
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
        for key in ("video", "poster"):
            relative = Path(recording[key])
            path = (ROOT / "docs" / relative).resolve()
            if not path.is_relative_to(ROOT / "docs/assets/rollouts"):
                raise ValueError(f"Invalid recording asset path: {relative}")
            if key == "video":
                sources[task_id] = media_source(relative, recording["videoSha256"], media)
                if media == "remote":
                    continue
            elif hashlib.sha256(path.read_bytes()).hexdigest() != recording[key + "Sha256"]:
                raise ValueError(f"Recording asset hash mismatch: {relative}")
            assets.add(relative)
    return recordings, assets, sources


def official_instructions():
    manifest = json.loads((ROOT / "website/task-instructions.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported instruction manifest")
    for task_id, task in manifest["tasks"].items():
        if not task["instruction"] or hashlib.sha256(task["instruction"].encode()).hexdigest() != task["instructionSha256"]:
            raise ValueError(f"Official instruction hash mismatch: {task_id}")
    return manifest["tasks"]


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


def task_cards(tasks):
    cards = []
    for index, task in enumerate(tasks):
        escape = html.escape
        task_id = task["id"]
        title = escape(task["title"])
        rollout = task.get("rollout")
        if rollout:
            media = (
                f'<video class="rollout-video" data-src="{escape(rollout["video"])}" '
                f'poster="{escape(rollout["poster"])}" controls autoplay muted loop playsinline '
                f'preload="none" aria-label="{title}, head-camera rollout, instance {rollout["instance"]}">'
                'Your browser does not support video playback.</video>'
                f'<p class="rollout-error" role="status" hidden>Video could not load. '
                f'<a href="{escape(rollout["video"])}">Open the video ↗</a></p>'
            )
            score = "Partial · No final Q" if rollout["q"] is None else f'Recorded Q {rollout["q"]:.4f}'
            caption = (
                f'<span>Head camera · Instance {rollout["instance"]}</span><span>{score}</span>'
                '<button type="button" class="speed-toggle" aria-label="Playback speed: 4 times. Change playback speed." data-rate="4">4×</button>'
            )
        else:
            media = '<div class="rollout-empty"><svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><rect x="2" y="5" width="14" height="14" rx="3"/><path d="m16 9 6-3v12l-6-3"/></svg><p>Head-camera recording unavailable.</p></div>'
            caption = ''
        columns = []
        for case in task["cases"]:
            columns.append(
                f'<div class="chart-column" style="--q:{case["q"] * 100:.6f}%" aria-hidden="true">'
                '<div class="chart-bar"></div>'
                f'<span class="chart-value">{case["q"]:.4f}</span>'
                f'<span class="chart-label">{case["instance"]}</span></div>'
            )
        score_label = escape(task["name"] + '. Archived Q-scores: ' + '; '.join(
            f'instance {case["instance"]}: {case["q"]:.4f}' for case in task["cases"]))
        previous = tasks[(index - 1) % len(tasks)]["title"]
        following = tasks[(index + 1) % len(tasks)]["title"]
        cards.append(f'''<article class="task-card" id="card-{task_id}" data-task="{task_id}" aria-labelledby="title-{task_id}" hidden>
          <header class="task-card-heading">
            <h3 id="title-{task_id}">{title}</h3>
            <a class="instruction-source" href="{escape(task["instructionUrl"])}" target="_blank" rel="noopener">Official instruction ↗</a>
            <p class="task-instruction">{escape(task["instruction"])}</p>
          </header>
          <div class="task-columns">
            <div class="rollout-panel">{media}</div>
            <div class="score-panel">
              <div class="selected-metric"><span class="task-mean">{task["mean"]:.4f}</span><span>Mean Q-score</span></div>
              <p class="chart-heading">Q-score by instance</p>
              <div class="score-chart" role="img" aria-label="{score_label}">{"".join(columns)}</div>
            </div>
          </div>
          <div class="task-media-caption">{caption}</div>
          <nav class="task-navigation" aria-label="Switch task">
            <button type="button" class="task-arrow" data-step="-1" aria-label="Previous task: {escape(previous)}"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg></button>
            <span class="task-position" aria-label="Task {index + 1} of {len(tasks)}">{index + 1:02d} <span aria-hidden="true">/</span> {len(tasks):02d}</span>
            <button type="button" class="task-arrow" data-step="1" aria-label="Next task: {escape(following)}"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m10 6 6 6-6 6"/></svg></button>
          </nav>
        </article>''')
    return "\n".join(cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    parser.add_argument("--latest-method", action="store_true", help="Use the latest completed method export when its local directory is available")
    parser.add_argument("--media", choices=("remote", "local"), default="remote",
                        help="Link videos to the pinned asset repository, or copy verified local copies (offline package)")
    args = parser.parse_args()
    if args.latest_method and args.media != "local":
        parser.error("--latest-method uses a local export; add --media local")
    output = args.output.resolve()
    # Do not overwrite source directories, including an accidentally selected root.
    if output == ROOT or ROOT.is_relative_to(output) or any(
        output.is_relative_to(ROOT / directory) for directory in ("website", "docs", "scripts", "tasks", "prompt")
    ):
        parser.error("choose a separate generated-output directory")
    source = ROOT / "website"
    prepared_method = method_video.latest(ROOT) if args.latest_method else method_video.snapshot(ROOT, args.media)
    method = prepared_method[0]
    if prepared_method[1] is None:
        method_source = media_source(method_video.VIDEO, method["videoSha256"], "remote")
    else:
        method_source = f'{method_video.VIDEO.as_posix()}?v={method["videoSha256"][:12]}'
    tasks = archive_data()
    instructions = official_instructions()
    recordings, media_assets, video_sources = recorded_rollouts(args.media)
    if not set(recordings).issubset(task["id"] for task in tasks):
        raise ValueError("Recording manifest refers to an unknown task")
    for task in tasks:
        instruction = instructions[task["id"]]
        if instruction["taskName"] != task["name"].replace(" ", "_"):
            raise ValueError(f"Task/instruction mismatch: {task['id']}")
        task.update({"title": instruction["title"], "instruction": instruction["instruction"],
                     "instructionUrl": instruction["officialPage"]})
        if task["id"] in recordings:
            recording = recordings[task["id"]]
            task["rollout"] = {key: recording[key] for key in (
                "run", "instance", "q", "label", "video", "poster", "duration",
                "defaultPlaybackRate", "missingTailSeconds")}
            task["rollout"]["video"] = video_sources[task["id"]]
            task["rollout"]["poster"] += "?v=" + recording["posterSha256"][:12]
    replacements = {
        "{{ARCHIVE_ROWS}}": archive_rows(tasks), "{{TASK_CARDS}}": task_cards(tasks),
        "{{STYLE_VERSION}}": hashlib.sha256((source / "styles.css").read_bytes()).hexdigest()[:12],
        "{{SCRIPT_VERSION}}": hashlib.sha256((source / "app.js").read_bytes()).hexdigest()[:12],
        "{{METHOD_VIDEO}}": html.escape(method_source),
        "{{METHOD_POSTER_VERSION}}": method["posterSha256"][:12],
        "{{METHOD_WIDTH}}": str(method["width"]), "{{METHOD_HEIGHT}}": str(method["height"]),
        "{{TASK_DATA}}": json.dumps(tasks, separators=(",", ":")).replace("<", "\\u003c"),
    }
    page = (source / "index.html").read_text()
    for marker, value in replacements.items():
        if marker not in page:
            raise ValueError(f"Missing template marker: {marker}")
        page = page.replace(marker, value)
    if "{{" in page:
        raise ValueError("Unresolved page template marker")
    paper = output / "v1"
    (paper / "assets").mkdir(parents=True, exist_ok=True)
    # A reused generated directory must not retain old private trace exports.
    for stale in (paper / "assets/rollouts").glob("*-trace.json"):
        stale.unlink()
    (paper / "index.html").write_text(page)
    for asset in ("styles.css", "app.js"):
        shutil.copyfile(source / asset, paper / asset)
    for asset in (ROOT / "docs" / "assets").iterdir():
        if asset.suffix in (".png", ".pdf", ".svg"):
            shutil.copyfile(asset, paper / "assets" / asset.name)
    for relative in sorted(media_assets):
        target = paper / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs" / relative, target)
    shutil.copyfile(source / "rollouts.json", paper / "assets/rollouts/manifest.json")
    method_video.copy_assets(paper, prepared_method)
    shutil.copyfile(ROOT / "docs/assets/roboharness-icon.svg", paper / "favicon.svg")
    (paper / ".nojekyll").touch()
    static_pages(output, args.media)
    (output / ".nojekyll").touch()
    print(f"Built {output}: portal, v2 and paper page (v1) with {len(tasks)} tasks, "
          f"{len(recordings)} recorded head-camera rollouts, method {method['sourceFile']}")


if __name__ == "__main__":
    main()
