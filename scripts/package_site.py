#!/usr/bin/env python3
"""Package the complete project page for offline viewing and Windows sync."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def source_fingerprint():
    paths = {Path(__file__), ROOT / "scripts/build_site.py", ROOT / "LICENSE", ROOT / "THIRD_PARTY_NOTICES.md"}
    for directory in ("website", "tasks", "prompt", "docs/assets", "scripts/offline"):
        paths.update(path for path in (ROOT / directory).rglob("*") if path.is_file())
    stats = [(str(path.relative_to(ROOT)), path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(paths)]
    return hashlib.sha256(json.dumps(stats, separators=(",", ":")).encode()).hexdigest()


def package(output, archive=None):
    fingerprint = source_fingerprint()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "offline-manifest.json"
    if not manifest_path.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a nonempty folder without an offline manifest: {output}")
    with tempfile.TemporaryDirectory(prefix=".roboharness-package-", dir=output.parent) as temporary:
        staging = Path(temporary) / "site"
        subprocess.run([sys.executable, str(ROOT / "scripts/build_site.py"), "--output", str(staging)], check=True)
        for path in (ROOT / "scripts/offline").iterdir():
            if path.is_file():
                target = staging / path.name
                text = path.read_text(encoding="utf-8")
                if path.suffix in (".ps1", ".cmd"):
                    target.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
                else:
                    target.write_text(text, encoding="utf-8")
        licenses = staging / "licenses"
        licenses.mkdir()
        shutil.copyfile(ROOT / "LICENSE", licenses / "LICENSE.txt")
        shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", licenses / "THIRD_PARTY_NOTICES.txt")
        # Keep the legal documents available when the laptop is offline.
        page = staging / "index.html"
        text = page.read_text().replace(
            "https://github.com/BinceQu/RoboHarness/blob/main/LICENSE", "licenses/LICENSE.txt"
        ).replace(
            "https://github.com/BinceQu/RoboHarness/blob/main/THIRD_PARTY_NOTICES.md", "licenses/THIRD_PARTY_NOTICES.txt"
        )
        page.write_text(text)
        files = {path.relative_to(staging).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {
            "version": 1,
            "source": "https://github.com/BinceQu/RoboHarness",
            "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "builtAt": datetime.now(timezone.utc).isoformat(),
            "sourceFingerprint": fingerprint,
            "contentHash": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "files": files,
        }
        for relative, entry in files.items():
            target = output / relative
            if target.is_file() and target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, target)
        # Publish the complete manifest last. Laptop downloads verify every file.
        temporary_manifest = output / ".offline-manifest.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary_manifest, manifest_path)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".roboharness-", suffix=".zip", dir=archive.parent, delete=False) as temporary:
            temporary_zip = Path(temporary.name)
        try:
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_STORED) as bundle:
                for relative in sorted([*files, "offline-manifest.json"]):
                    bundle.write(output / relative, f"roboharness_homepage/{relative}")
            os.replace(temporary_zip, archive)
        finally:
            temporary_zip.unlink(missing_ok=True)
    print(json.dumps({"folder": str(output), "files": len(files), "bytes": sum(row["bytes"] for row in files.values()),
                      "videos": sum(name.endswith(".mp4") for name in files), "archive": str(archive) if archive else None}), flush=True)
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "roboharness_homepage")
    parser.add_argument("--zip", type=Path, dest="archive")
    parser.add_argument("--watch", action="store_true", help="Rebuild after the website source changes")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    archive = args.archive.expanduser().resolve() if args.archive else None
    if output == ROOT or ROOT.is_relative_to(output) or any(output.is_relative_to(ROOT / directory) for directory in (
        "website", "docs", "scripts", "tasks", "prompt"
    )):
        parser.error("choose a separate generated-output directory")
    if archive and (archive == output or archive.is_relative_to(output)):
        parser.error("place the ZIP outside the offline folder")
    if args.interval < 1:
        parser.error("the watch interval must be at least one second")
    previous = None
    if args.watch and (output / "offline-manifest.json").exists():
        previous = json.loads((output / "offline-manifest.json").read_text()).get("sourceFingerprint")
    while True:
        try:
            current = source_fingerprint()
            if not args.watch or current != previous or (archive and not archive.exists()):
                previous = package(output, archive)
        except Exception as error:
            if not args.watch:
                raise
            print(f"Offline package update failed; will retry: {error}", file=sys.stderr, flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
