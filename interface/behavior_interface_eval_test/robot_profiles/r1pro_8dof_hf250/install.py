#!/usr/bin/env python3
"""Expose this self-contained RobotDefinition through an immutable data root.

Kit keeps inotify watchers on the directories of every loaded asset
(``omnigibson-robot-assets/models/r1pro/usd/materials`` and friends).  When a
launcher replaces ``models/r1pro`` inside a data root that live evaluators are
using, every one of them wakes its watcher callbacks on tasking fibers, and the
contended ``carb::thread::mutex`` trips Kit 107.3's thread-ownership fatal
check (``unlock() called by non-owning thread``).  On 2026-09-14 that killed
nine official evaluators in the same second.

So this module never mutates a shared data root.  It builds a sibling
``<root>__r1pro-<digest>`` overlay once: every entry of the root is symlinked
except ``omnigibson-robot-assets/models/r1pro``, which is a private copy of
the profile tree.  The overlay is named by the content digest and never
rewritten, so two profile variants (or a profile and stock) coexist without
touching each other's watched directories.
"""

from __future__ import annotations

import argparse
import errno
import filecmp
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


PROFILE_DIR = Path(__file__).resolve().parent
MODEL = "r1pro"
PROFILE_MODEL = "r1pro_8dof_hf250"
SOURCE = PROFILE_DIR / "assets" / "models" / MODEL
ROBOT_ASSETS = "omnigibson-robot-assets"
BACKUP_DIRNAME = ".behavior_interface_eval_test_backups"
VARIANT_MARKER = ".behavior_interface_eval_test_variant.json"
PROFILE_LABEL = MODEL
STOCK_LABEL = f"{MODEL}-stock"


def _same_tree(left: Path, right: Path) -> bool:
    """Return whether two asset trees have the same entries and file bytes."""
    if not left.is_dir() or not right.is_dir():
        return False

    def entries(root: Path) -> dict[str, tuple[str, str | None]]:
        result: dict[str, tuple[str, str | None]] = {}
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[relative] = ("symlink", os.readlink(path))
            elif path.is_dir():
                result[relative] = ("directory", None)
            elif path.is_file():
                result[relative] = ("file", None)
            else:
                result[relative] = ("other", None)
        return result

    left_entries = entries(left)
    if left_entries != entries(right):
        return False
    for relative, (kind, _) in left_entries.items():
        if kind == "file" and not filecmp.cmp(
            left / relative,
            right / relative,
            shallow=False,
        ):
            return False
    return True


def tree_digest(root: Path) -> str:
    """Content digest of an asset tree (paths, kinds, bytes, link targets)."""
    if not root.is_dir():
        raise FileNotFoundError(f"asset tree is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            digest.update(b"L" + relative + b"\0" + os.readlink(path).encode("utf-8") + b"\0")
        elif path.is_dir():
            digest.update(b"D" + relative + b"\0")
        elif path.is_file():
            digest.update(b"F" + relative + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()[:16]


@contextmanager
def _installation_lock(data_root: Path):
    """Serialize overlay checks and builds for one data root."""
    root_key = hashlib.sha256(str(data_root).encode("utf-8")).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / (
        f"behavior_interface_eval_test_r1pro_install_{root_key}.lock"
    )
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def variant_root_for(data_root: Path, label: str, digest: str) -> Path:
    return data_root.parent / f"{data_root.name}__{label}-{digest}"


def _models_dir(root: Path) -> Path:
    return root / ROBOT_ASSETS / "models"


def _link_children(source: Path, target: Path, *, skip: frozenset[str]) -> None:
    for child in sorted(source.iterdir(), key=lambda item: item.name):
        if child.name in skip:
            continue
        (target / child.name).symlink_to(child, target_is_directory=child.is_dir())


def _expected_links(data_root: Path) -> list[tuple[Path, Path]]:
    """(link inside overlay relative to root, absolute target) pairs."""
    pairs: list[tuple[Path, Path]] = []
    for child in data_root.iterdir():
        if child.name != ROBOT_ASSETS:
            pairs.append((Path(child.name), child))
    assets = data_root / ROBOT_ASSETS
    for child in assets.iterdir():
        if child.name != "models":
            pairs.append((Path(ROBOT_ASSETS) / child.name, child))
    for child in _models_dir(data_root).iterdir():
        if child.name not in {MODEL, PROFILE_MODEL}:
            pairs.append((Path(ROBOT_ASSETS) / "models" / child.name, child))
    return pairs


def _overlay_is_complete(data_root: Path, variant_root: Path, model_tree: Path) -> bool:
    if not variant_root.is_dir() or variant_root.is_symlink():
        return False
    if not _same_tree(model_tree, _models_dir(variant_root) / MODEL):
        return False
    for relative, target in _expected_links(data_root):
        link = variant_root / relative
        if not link.is_symlink() or Path(os.readlink(link)) != target:
            return False
    return True


def _live_users(root: Path) -> list[int]:
    """PIDs of same-user processes whose OMNIGIBSON_DATA_PATH is ``root``."""
    wanted = os.path.realpath(root)
    prefix = b"OMNIGIBSON_DATA_PATH="
    users: list[int] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry.name}/environ", "rb") as stream:
                environ = stream.read()
        except OSError:
            continue
        for item in environ.split(b"\0"):
            if item.startswith(prefix):
                value = item[len(prefix):].decode("utf-8", "replace")
                if value and os.path.realpath(value) == wanted:
                    users.append(int(entry.name))
                break
    return users


def _build_overlay(
    data_root: Path,
    variant_root: Path,
    model_tree: Path,
    *,
    label: str,
    digest: str,
) -> None:
    """Assemble the overlay in a staging directory, then publish it atomically."""
    staging = Path(
        tempfile.mkdtemp(prefix=f".{variant_root.name}.", dir=variant_root.parent)
    )
    try:
        _link_children(data_root, staging, skip=frozenset({ROBOT_ASSETS}))
        assets = staging / ROBOT_ASSETS
        assets.mkdir()
        _link_children(data_root / ROBOT_ASSETS, assets, skip=frozenset({"models"}))
        models = assets / "models"
        models.mkdir()
        _link_children(
            _models_dir(data_root),
            models,
            skip=frozenset({MODEL, PROFILE_MODEL}),
        )
        shutil.copytree(model_tree, models / MODEL, symlinks=True)
        (staging / VARIANT_MARKER).write_text(
            json.dumps(
                {
                    "data_root": str(data_root),
                    "label": label,
                    "digest": digest,
                    "source": str(model_tree),
                    "profile": PROFILE_MODEL,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        # mkdtemp creates 0700; match the shared root so other launchers can read it.
        os.chmod(staging, 0o755)
        try:
            os.rename(staging, variant_root)
        except OSError as exc:
            # Another launcher published the same digest first; keep theirs.
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not variant_root.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _add_missing_links(data_root: Path, variant_root: Path) -> None:
    """Link entries the data root gained after the overlay was published.

    Adding symlinks at the overlay's top levels does not touch any directory
    Kit watches, so this is safe even while evaluators use the overlay.
    """
    for relative, target in _expected_links(data_root):
        link = variant_root / relative
        if link.is_symlink() and Path(os.readlink(link)) == target:
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(
                f"overlay entry {link} does not point at {target}; "
                "remove the stale overlay while no evaluator uses it"
            )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=target.is_dir())


def resolve_overlay(data_root: Path, model_tree: Path, *, label: str) -> Path:
    """Return an immutable data root exposing ``model_tree`` as ``models/r1pro``."""
    data_root = data_root.expanduser().resolve()
    models_dir = _models_dir(data_root)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"official robot assets directory is missing: {models_dir}")
    if not model_tree.is_dir():
        raise FileNotFoundError(f"robot asset tree is missing: {model_tree}")
    digest = tree_digest(model_tree)
    variant_root = variant_root_for(data_root, label, digest)
    with _installation_lock(data_root):
        if _overlay_is_complete(data_root, variant_root, model_tree):
            return variant_root
        if variant_root.exists():
            if _same_tree(model_tree, _models_dir(variant_root) / MODEL):
                _add_missing_links(data_root, variant_root)
                return variant_root
            users = _live_users(variant_root)
            if users:
                raise RuntimeError(
                    f"overlay {variant_root} no longer matches {model_tree} but is in "
                    f"use by pid(s) {sorted(users)}; refusing to rewrite assets under "
                    "live evaluators"
                )
            shutil.rmtree(variant_root)
        _build_overlay(data_root, variant_root, model_tree, label=label, digest=digest)
        if not _overlay_is_complete(data_root, variant_root, model_tree):
            raise RuntimeError(f"overlay verification failed: {variant_root}")
    return variant_root


def _stock_tree(data_root: Path) -> Path | None:
    """Where the untouched stock r1pro lives for this data root, if anywhere."""
    backup = data_root / ROBOT_ASSETS / BACKUP_DIRNAME / MODEL
    if backup.is_dir():
        return backup
    return None


def install(data_root: Path) -> Path:
    """Return ``models/r1pro`` of the immutable overlay carrying this profile."""
    variant_root = resolve_overlay(data_root, SOURCE, label=PROFILE_LABEL)
    return _models_dir(variant_root) / MODEL


def resolve_stock_data_root(data_root: Path) -> Path:
    """Data root whose ``models/r1pro`` is the stock tree, without mutating anything.

    Roots that an older launcher rewrote in place keep the stock copy under
    ``.behavior_interface_eval_test_backups``; those get a ``-stock`` overlay.
    Roots that were never rewritten still hold stock ``models/r1pro`` and are
    returned unchanged.
    """
    data_root = data_root.expanduser().resolve()
    stock = _stock_tree(data_root)
    if stock is None:
        if not (_models_dir(data_root) / MODEL).is_dir():
            raise FileNotFoundError(f"stock r1pro is missing: {_models_dir(data_root) / MODEL}")
        return data_root
    return resolve_overlay(data_root, stock, label=STOCK_LABEL)


def restore(data_root: Path) -> Path:
    """Return ``models/r1pro`` of the stock data root (see resolve_stock_data_root)."""
    return _models_dir(resolve_stock_data_root(data_root)) / MODEL


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the immutable OMNIGIBSON_DATA_PATH to launch with. The given "
            "data root is never modified."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "data",
    )
    parser.add_argument("--restore-stock", action="store_true")
    args = parser.parse_args()
    if args.restore_stock:
        print(resolve_stock_data_root(args.data_root))
    else:
        print(resolve_overlay(args.data_root, SOURCE, label=PROFILE_LABEL))


if __name__ == "__main__":
    main()
