#!/usr/bin/env python3
"""Stage the GPT-6 Astra x RoboHarness V2 BEHAVIOR rollouts for the project site.

Reads one private run directory and writes, outside git:
  videos/tNN-head.mp4    head-camera crop at the recorded resolution and frame rate
  previews/tNN-head.mp4  8 s silent head-camera time-lapse for the gallery grid
  posters/tNN-head.jpg   head-camera frame near the end of the rollout
and website/astra/episodes.json with scores, prompts and agent settings.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_SECONDS = 8
PREVIEW_SIZE = 336
# The evaluator puts two wrist views on the left and the square head view on the right.
HEAD_CROP = "crop=ih:ih:iw-ih:0"


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,nb_frames",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    return float(data["format"]["duration"]), stream["width"], stream["height"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ffmpeg(*args):
    target = Path(args[-1])
    temporary = target.with_name(f".{target.stem}.partial{target.suffix}")
    try:
        subprocess.run(["nice", "-n", "19", "ffmpeg", "-nostdin", "-v", "error", "-y",
                        *args[:-1], str(temporary)], check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def stage(source, key, out, duration):
    _, width, height = probe(source)
    if width * 2 != height * 3:
        raise ValueError(f"Expected the evaluator's 3:2 camera composite: {source} ({width}x{height})")
    video, preview, poster = out / "videos" / f"{key}-head.mp4", out / "previews" / f"{key}-head.mp4", out / "posters" / f"{key}-head.jpg"
    if not video.exists():
        ffmpeg("-threads", "2", "-i", str(source), "-map", "0:v:0", "-an", "-vf", HEAD_CROP,
               "-filter_threads", "1", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video))
    if not preview.exists():
        speed = min(1.0, PREVIEW_SECONDS / duration)
        ffmpeg("-threads", "2", "-i", str(video), "-an", "-vf", f"setpts={speed:.6f}*PTS,fps=12,scale={PREVIEW_SIZE}:{PREVIEW_SIZE}",
               "-filter_threads", "1", "-c:v", "libx264", "-threads", "2", "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(preview))
    if not poster.exists():
        ffmpeg("-threads", "2", "-ss", f"{duration * 0.92:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(poster))
    return {kind: {"file": f"{kind}/{path.name}",
                   "bytes": path.stat().st_size, "sha256": sha256(path)}
            for kind, path in (("videos", video), ("previews", preview), ("posters", poster))}


def recorded_card(out, recording=None):
    """Crop the selected browser recording without changing frame order or timing."""
    video = out / "card/portal-recording.mp4"
    poster = out / "card/portal-recording-poster.jpg"
    provenance = out / "card/portal-recording.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    if recording is not None:
        duration, width, _ = probe(recording)
        crop_height = round(width * 9 / 16 / 2) * 2
        ffmpeg("-threads", "2", "-i", str(recording), "-an", "-vf", f"crop=iw:{crop_height}:0:0",
               "-filter_threads", "1", "-c:v", "libx264", "-threads", "3", "-preset", "slow",
               "-crf", "18", "-tune", "fastdecode", "-bf", "0", "-g", "60",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video))
        ffmpeg("-i", str(video), "-frames:v", "1", "-q:v", "2", str(poster))
        provenance.write_text(json.dumps({"source_file": recording.name,
            "source_sha256": sha256(recording), "source_seconds": duration,
            "crop": [0, 0, width, crop_height], "playback_rate": 1,
            "description": "Video area cropped from the selected browser recording; original frame order and timing."}, indent=2) + "\n")
    if not all(path.is_file() for path in (video, poster, provenance)):
        raise ValueError("Pass --card-recording to stage the selected portal recording")
    return {name: {"file": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in (("video", video), ("poster", poster), ("provenance", provenance))}


def bash_calls(stream):
    count = 0
    for line in stream.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "item.completed" and (event.get("item") or {}).get("type") == "command_execution":
            count += 1
    return count


def apply_estimates(rows):
    estimates = json.loads((ROOT / "website/astra/estimates.json").read_text())
    for row in rows:
        row["score_source"] = "official" if row["status"] == "scored" else None
        if row["id"] in estimates and row["status"] != "scored":
            estimate = estimates[row["id"]]
            row.update(status="estimated", q=estimate["q"], official_q=None,
                       score_source="video_estimate", estimate=estimate,
                       note="Video-estimated Q ≈ 0.50 at the last recorded tick (11904). "
                            "Toaster and French press are in the cabinet; the food processor is outside and the doors remain open. "
                            "The run is incomplete and has no official final score.")
    official = [row for row in rows if row["status"] == "scored"]
    scored = [row for row in rows if row["q"] is not None]
    return {"tasks": len(rows), "scored": len(scored), "official_scored": len(official),
            "estimated": sum(row["status"] == "estimated" for row in rows),
            "mean_q": sum(row["q"] for row in scored) / len(scored),
            "official_mean_q": sum(row["q"] for row in official) / len(official),
            "full_success": sum(row["q"] >= 1 for row in scored),
            "partial": sum(0 < row["q"] < 1 for row in scored),
            "zero": sum(row["q"] == 0 for row in scored)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-recording", type=Path, help="Selected browser recording; crop its video area at the original playback timing")
    parser.add_argument("run", type=Path, help="codex-astra-b19-gpu5-20261016/run")
    parser.add_argument("--out", type=Path, required=True, help="media staging directory (not in git)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    for kind in ("videos", "previews", "posters"):
        (args.out / kind).mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.run / "manifest.json").read_text())
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    submission = args.run / "submission"
    rows, work = [], {}
    for task_id, task in sorted(tasks.items()):
        name, key = task["task_name"], f"t{task_id:02d}"
        job = args.run / "jobs" / f"{key}-i301-r0"
        metrics_file = submission / "metrics" / f"{name}_301_0.json"
        row = {"id": key, "task_id": task_id, "task": name, "instance": 301, "rollout": 0,
               "scene": task["scene_model"], "max_ticks": task["max_ticks"]}
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            source = submission / "videos" / f"{name}_301_0.mp4"
            attempts = sorted(p for p in job.glob("attempt*") if (p / "summary.json").exists()
                              and json.loads((p / "summary.json").read_text()).get("phase") == "complete")
            attempt = attempts[-1] / "instance_301"
            row.update(status="scored", q=metrics["q_score"]["final"], success=metrics["success"],
                       steps=metrics["steps"], sim_seconds=round(metrics["time"]["simulator_time"], 1))
        else:
            source = next((submission / "videos_incomplete").glob(f"{name}_301_0.*.mp4"))
            attempt = job / source.name.split(".")[1].split("_")[0] / "instance_301"
            ticks = int(source.stem.rsplit("tick", 1)[1])
            row.update(status="unscored", q=None, success=None, steps=ticks, sim_seconds=round(ticks / 30, 1),
                       note=f"This attempt ended at tick {ticks} before the evaluator scored it; no official score.")
        prompt = (attempt / "prompt.txt").read_text()
        streams = sorted(attempt.glob("native/*/stream.jsonl"))
        row.update(prompt=prompt, bash_calls=sum(bash_calls(s) for s in streams) if streams else None)
        duration, width, height = probe(source)
        row.update(video_seconds=round(duration, 1), width=height, height=height, video_view="head")
        rows.append(row)
        work[key] = (source, duration)

    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = {key: pool.submit(stage, src, key, args.out, d) for key, (src, d) in work.items()}
        for row in rows:
            row["media"] = futures[row["id"]].result()
            print(row["id"], row["q"], "staged", flush=True)

    agent, render = manifest["agent"], manifest["render"]
    settings = {
        "Model": "GPT-6 Astra (gpt-6-astra)",
        "Reasoning effort": agent["reasoning_effort"],
        "Reasoning summary": agent["reasoning_summary"],
        "Coding agent": "Codex CLI 0.161.0, codex exec (native, unmodified)",
        "Agent tools": "Codex native shell + view_image; no MCP, web search off, multi-agent off",
        "Approval / sandbox": "never / danger-full-access (needed for the workspace GPU planner)",
        "Harness": "RoboHarness V2 workspace (harness/v2/R1_pro): readable Python tools over the 27-value joint/controller API",
        "Skills": "behavior-v2-baseline, pick-up-object, place-object-in-container, open-doors-and-drawers",
        "Prompt": "baseline_v3 task instructions, one per task",
        "Robot": "Galaxea R1 Pro (dual 7-DoF arms, torso, holonomic base)",
        "Video view": "Head camera only; recorded 448x448 pixels, original frame rate",
        "Benchmark": "BEHAVIOR Challenge 2026, official evaluator v3.9.3",
        "Episode": "public instance 301, rollout 0, seed 0; one attempt per task",
        "Tick budget": "official per-task max ticks (shown per episode)",
        "Rendering": f"low-memory profile, texture memory fraction {render['texture_memory_fraction']}",
    }
    summary = apply_estimates(rows)
    card = recorded_card(args.out, args.card_recording)
    page = {"version": 1, "title": "GPT-6 Astra on RoboHarness V2", "benchmark": "BEHAVIOR Challenge 2026 · tasks 0–99",
            "summary": summary, "settings": settings, "card": card, "episodes": rows}
    target = ROOT / "website" / "astra" / "episodes.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(page, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
