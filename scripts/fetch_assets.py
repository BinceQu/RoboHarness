#!/usr/bin/env python3
"""Download externally hosted assets listed in assets.lock.json and verify their hashes.

Videos, reference results and validation reports are hosted on Hugging Face so
that cloning the repository does not download them. Each file is restored to
its original repository path and checked against the locked SHA-256.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("assets.lock.json")
ATTEMPTS = 4


def lock(root=ROOT):
    data = json.loads((root / LOCK).read_text())
    if data["version"] != 1:
        raise ValueError("Unsupported asset lock")
    return data


def entry(data, path):
    for row in data["files"]:
        if row["path"] == path:
            return row
    raise ValueError(f"Asset is not in {LOCK}: {path}")


def url(data, row):
    return f'https://huggingface.co/datasets/{data["repository"]}/resolve/{row["revision"]}/{row["remote"]}'


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verified(path, row):
    return path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]


def fetch(data, row, root=ROOT, force=False):
    target = root / row["path"]
    if not force and verified(target, row):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(ATTEMPTS):
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".fetch-", delete=False) as temporary:
            partial = Path(temporary.name)
            try:
                with urllib.request.urlopen(url(data, row), timeout=120) as response:
                    shutil.copyfileobj(response, temporary, 1024 * 1024)
                break
            except OSError as error:
                partial.unlink(missing_ok=True)
                if attempt + 1 == ATTEMPTS:
                    raise
                print(f'retrying {row["path"]}: {error}', file=sys.stderr)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        time.sleep(2 ** attempt)
    if not verified(partial, row):
        partial.unlink()
        raise ValueError(f'Downloaded asset hash mismatch: {row["path"]}')
    os.replace(partial, target)
    return True


def fetch_groups(groups, root=ROOT, force=False):
    data = lock(root)
    rows = [row for row in data["files"] if row["group"] in groups]
    downloaded = sum(fetch(data, row, root, force) for row in rows)
    return len(rows), downloaded


def upload(root, source, path, group, remote, message):
    """Upload one file to the asset repository and pin it in the lock."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ValueError("Publishing assets requires huggingface_hub and an HF_TOKEN with write access") from error
    data = lock(root)
    commit = HfApi().upload_file(path_or_fileobj=str(source), path_in_repo=remote,
                                 repo_id=data["repository"], repo_type="dataset", commit_message=message)
    row = {"path": path, "group": group, "remote": remote, "revision": commit.oid,
           "bytes": source.stat().st_size, "sha256": digest(source)}
    data["files"] = [old for old in data["files"] if old["path"] != path] + [row]
    data["files"].sort(key=lambda old: old["path"])
    temporary = root / (str(LOCK) + ".tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(temporary, root / LOCK)
    return row


def main():
    data = lock()
    groups = sorted({row["group"] for row in data["files"]})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=groups + ["all"],
                        help="asset group to fetch (repeatable; default: reference)")
    parser.add_argument("--force", action="store_true", help="download even if a verified copy exists")
    parser.add_argument("--list", action="store_true", help="print the selected files and their URLs")
    args = parser.parse_args()
    selected = set(groups) if "all" in (args.group or []) else set(args.group or ["reference"])
    if args.list:
        for row in data["files"]:
            if row["group"] in selected:
                print(f'{row["group"]:<10} {row["bytes"]:>11} {row["path"]}\n{"":<23}{url(data, row)}')
        return
    total, downloaded = fetch_groups(selected, force=args.force)
    print(f"{total} assets verified ({downloaded} downloaded) for: {', '.join(sorted(selected))}", file=sys.stderr)


if __name__ == "__main__":
    main()
