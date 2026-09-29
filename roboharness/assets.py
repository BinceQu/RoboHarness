"""Expand the losslessly compressed custom robot asset with an integrity check."""
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def prepare_robot_asset():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'configs/robot_asset.json').read_text())
    target = root / manifest['path']
    archive = root / manifest['archive']
    with archive.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == manifest['sha256']:
            return target
        temp = target.with_suffix('.tmp')
        with gzip.open(archive, 'rb') as src, temp.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != manifest['sha256']:
            temp.unlink()
            raise ValueError('Robot asset checksum mismatch')
        temp.replace(target)
        return target
