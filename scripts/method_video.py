#!/usr/bin/env python3
"""Prepare the latest completed method film and publish its isolated snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path("assets/method/roboharness-method.mp4")
POSTER = Path("assets/method/roboharness-method-poster.jpg")
MANIFEST = Path("assets/method/manifest.json")
PUBLISH_PATHS = ["website/method-video.json", "docs/" + str(VIDEO), "docs/" + str(POSTER)]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def configuration(root=ROOT):
    config = json.loads((root / "website/method-video.json").read_text())
    if config["version"] != 1:
        raise ValueError("Unsupported method-video manifest")
    return config


def source_candidates(config):
    directory = Path(config["sourceDirectory"])
    candidates = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = re.fullmatch(r"roboharness_promo_v(\d+)\.mp4", path.name)
            if match and path.is_file() and int(match[1]) >= config["minimumVersion"]:
                candidates.append((int(match[1]), path))
    return sorted(candidates, reverse=True)


def snapshot(root=ROOT):
    metadata = configuration(root)["published"]
    if not metadata:
        raise ValueError("No published method video is available")
    video, poster = root / "docs" / VIDEO, root / "docs" / POSTER
    if digest(video) != metadata["videoSha256"] or digest(poster) != metadata["posterSha256"]:
        raise ValueError("Published method video or poster hash mismatch")
    return metadata, video, poster


def latest(root=ROOT):
    config = configuration(root)
    candidates = source_candidates(config)
    if not candidates:
        return snapshot(root)
    version, video = candidates[0]
    initial = video.stat()
    if initial.st_size == 0 or time.time() - initial.st_mtime < 5:
        raise ValueError(f"Method export is still being written: {video.name}")
    video_hash = digest(video)
    cache = root / ".local/method-video-cache" / video_hash
    cache.mkdir(parents=True, exist_ok=True)
    poster = cache / "poster.jpg"
    cached_metadata = cache / "metadata.json"
    if cached_metadata.exists() and poster.exists():
        metadata = json.loads(cached_metadata.read_text())
        if digest(poster) != metadata["posterSha256"]:
            raise ValueError("Cached method poster hash mismatch")
    else:
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,pix_fmt", "-of", "json", str(video)
        ], text=True))
        stream = next(row for row in probe["streams"] if row["codec_type"] == "video")
        duration = float(probe["format"]["duration"])
        if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p" or duration <= 0:
            raise ValueError("Method export must be a completed H.264/yuv420p MP4")
        validation = video.with_suffix(".validation.json")
        report = json.loads(validation.read_text()) if validation.exists() else {}
        if not (report.get("full_decode_passed") is True and report.get("sha256") == video_hash):
            subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                            "-map", "0:v:0", "-f", "null", "-"], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(min(2, duration / 2)),
                        "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(poster)], check=True)
        metadata = {"version": 1, "sourceFile": video.name, "sourceVersion": version,
                    "video": VIDEO.as_posix(), "videoSha256": video_hash, "bytes": initial.st_size,
                    "poster": POSTER.as_posix(), "posterSha256": digest(poster),
                    "duration": duration, "width": stream["width"], "height": stream["height"]}
        cached_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    final = video.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError("Method export changed while it was being validated")
    metadata = dict(metadata, sourceFile=video.name, sourceVersion=version)
    return metadata, video, poster


def copy_assets(output, prepared):
    metadata, video, poster = prepared
    for source, relative, expected in [(video, VIDEO, metadata["videoSha256"]),
                                       (poster, POSTER, metadata["posterSha256"])]:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError(f"Method asset changed during copy: {relative}")
    (output / MANIFEST).write_text(json.dumps(metadata, indent=2) + "\n")


def write_snapshot(root, prepared):
    metadata, video, poster = prepared
    config = configuration(root)
    for source, relative, expected in [(video, VIDEO, metadata["videoSha256"]),
                                       (poster, POSTER, metadata["posterSha256"])]:
        target = root / "docs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            if digest(temporary) != expected:
                raise ValueError("Method export changed while preparing its published snapshot")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    config["published"] = metadata
    (root / "website/method-video.json").write_text(json.dumps(config, indent=2) + "\n")


def git(root, *arguments, capture=False):
    result = subprocess.run(["git", "-C", str(root), *arguments], check=True,
                            stdout=subprocess.PIPE if capture else None, text=True,
                            env=git_environment(), timeout=120)
    return result.stdout.strip() if capture else None


def git_environment():
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2"
    return environment


def publish_package(output, root=ROOT):
    state_dir = root / ".local/method-publisher"
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "publish.lock").open("w") as lock:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        return _publish_package(output, root)


def _publish_package(output, root=ROOT):
    """Publish only method assets in an owned checkout; leave the working repo intact."""
    metadata = json.loads((output / MANIFEST).read_text())
    selected = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    state_dir = root / ".local/method-publisher"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    if state_path.exists():
        if json.loads(state_path.read_text())["selection"] == selected:
            return
    elif configuration(root)["published"] == metadata:
        state_path.write_text(json.dumps({"selection": selected,
                                         "commit": git(root, "rev-parse", "HEAD", capture=True)}) + "\n")
        return
    checkout = state_dir / "repository"
    marker = checkout / ".git/roboharness-method-publisher"
    if not checkout.exists():
        remote = git(root, "remote", "get-url", "origin", capture=True)
        with tempfile.TemporaryDirectory(prefix="clone-", dir=state_dir) as temporary:
            staging = Path(temporary) / "repository"
            subprocess.run(["git", "clone", "--single-branch", "--branch", "main", "--no-tags", remote, str(staging)],
                           check=True, env=git_environment(), timeout=120)
            (staging / ".git/roboharness-method-publisher").write_text(str(root.resolve()))
            os.replace(staging, checkout)
    if not marker.is_file() or marker.read_text() != str(root.resolve()):
        raise ValueError("Refusing to modify a checkout not owned by the method publisher")
    git(checkout, "fetch", "origin", "main")
    # This disposable checkout contains only the publisher's own generated edits.
    git(checkout, "reset", "--hard", "origin/main")
    for key in ("user.name", "user.email"):
        git(checkout, "config", key, git(root, "config", "--get", key, capture=True))
    if metadata["bytes"] >= 100 * 1024 * 1024:
        raise ValueError("Method export exceeds GitHub's single-file limit; offline sync remains available")
    write_snapshot(checkout, (metadata, output / VIDEO, output / POSTER))
    git(checkout, "add", "--", *PUBLISH_PATHS)
    dirty = git(checkout, "diff", "--cached", "--name-only", capture=True)
    if dirty:
        git(checkout, "commit", "-m", f"Update method video to {metadata['sourceFile']}")
        git(checkout, "push", "origin", "HEAD:main")
    commit = git(checkout, "rev-parse", "HEAD", capture=True)
    state_path.write_text(json.dumps({"selection": selected, "commit": commit}) + "\n")
    print(f"Published method video: {metadata['sourceFile']} ({commit[:7]})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepared = latest()
    write_snapshot(ROOT, prepared)
    print(json.dumps(prepared[0]), flush=True)


if __name__ == "__main__":
    main()
