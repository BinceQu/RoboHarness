"""Policy-owned immutable RGB-D snapshots and signed motion plans."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import struct
import threading
from typing import Any
import zlib

import numpy as np

from tools.util.geometry import json_ready


_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{secrets.token_hex(4)}.tmp")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _validate(value: str, expression: re.Pattern[str], label: str) -> str:
    normalized = str(value or "").strip()
    if not expression.fullmatch(normalized):
        raise ValueError(f"invalid {label}")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _encode_png_stdlib(rgb: np.ndarray) -> bytes:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("PNG image must be HxWx3")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image[:, :, :3])
    height, width = image.shape[:2]
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _encode_png(rgb: np.ndarray) -> bytes:
    try:
        import cv2
    except ImportError:
        return _encode_png_stdlib(rgb)
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image[:, :, ::-1])
    return bytes(encoded) if ok else _encode_png_stdlib(image)


def _depth_png(depth: np.ndarray) -> bytes | None:
    values = np.asarray(depth, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [2.0, 98.0])
    scale = max(1e-6, float(high - low))
    visual = np.clip((values - low) / scale, 0.0, 1.0)
    visual = (255.0 * (1.0 - visual)).astype(np.uint8)
    return _encode_png(np.repeat(visual[:, :, None], 3, axis=2))


def file_data_url(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file():
        return None
    mime = "image/png" if target.suffix == ".png" else "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(target.read_bytes()).decode(
        "ascii"
    )


@dataclass
class Snapshot:
    session_id: str
    image_id: str
    role: str
    rgb: np.ndarray
    depth: np.ndarray | None
    intrinsic_matrix: np.ndarray | None
    extrinsic_matrix: np.ndarray | None
    robot: dict[str, Any]
    sequence: int
    episode_id: str
    paths: dict[str, str]
    integrity_digest: str = ""

    def camera_metadata(self) -> dict[str, Any]:
        return {
            "intrinsic_matrix": None
            if self.intrinsic_matrix is None
            else self.intrinsic_matrix.tolist(),
            "extrinsic_matrix": None
            if self.extrinsic_matrix is None
            else self.extrinsic_matrix.tolist(),
            "image_width": int(self.rgb.shape[1]),
            "image_height": int(self.rgb.shape[0]),
            "frame": "robodojo_env",
            "source": "RoboDojo evaluator observation",
            "extrinsic_convention": "usd_camera_xform_to_world",
            "depth_coordinate_convention": "optical_x_right_y_down_z_forward",
            "optical_to_usd_axes": [1.0, -1.0, -1.0],
        }


class SnapshotStore:
    def __init__(
        self,
        root: str | Path | None = None,
        secret: bytes | None = None,
    ) -> None:
        default_root = os.environ.get(
            "ROBODOJO_INTERFACE_RUNS",
            str(Path(__file__).resolve().parents[2] / ".runtime/artifacts"),
        )
        self.root = Path(root or default_root)
        env_secret = os.environ.get("ROBODOJO_INTERFACE_SNAPSHOT_KEY")
        self._secret = secret or (
            env_secret.encode("utf-8") if env_secret else secrets.token_bytes(32)
        )
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}
        self._latest: dict[str, tuple[str, str]] = {}

    def _directory(self, session_id: str) -> Path:
        session = _validate(session_id, _SESSION_RE, "session_id")
        return self.root / session / "images"

    def save(
        self,
        *,
        session_id: str,
        role: str,
        camera: dict[str, Any],
        robot: dict[str, Any],
    ) -> Snapshot:
        directory = self._directory(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            count = self._counts.get(session_id, 0) + 1
            while (directory / f"img_{count:06d}.json").exists():
                count += 1
            self._counts[session_id] = count
        image_id = f"img_{count:06d}"
        rgb = np.asarray(camera["rgb"]).copy()
        depth_value = camera.get("depth")
        depth = (
            None
            if depth_value is None
            else np.asarray(depth_value, dtype=np.float32).copy()
        )
        intrinsic_value = camera.get("intrinsic_matrix")
        extrinsic_value = camera.get("extrinsic_matrix")
        intrinsic = (
            None
            if intrinsic_value is None
            else np.asarray(intrinsic_value, dtype=np.float64).reshape(3, 3)
        )
        extrinsic = (
            None
            if extrinsic_value is None
            else np.asarray(extrinsic_value, dtype=np.float64).reshape(4, 4)
        )

        prefix = directory / image_id
        rgb_npy = prefix.with_suffix(".rgb.npy")
        np.save(rgb_npy, rgb)
        paths: dict[str, str] = {"rgb_array": str(rgb_npy)}
        encoded_rgb = _encode_png(rgb)
        rgb_png = prefix.with_suffix(".png")
        rgb_png.write_bytes(encoded_rgb)
        paths["rgb"] = str(rgb_png)
        if depth is not None:
            depth_path = prefix.with_suffix(".depth.npy")
            np.save(depth_path, depth)
            paths["depth"] = str(depth_path)
            depth_visual = _depth_png(depth)
            if depth_visual is not None:
                depth_png = prefix.with_suffix(".depth.png")
                depth_png.write_bytes(depth_visual)
                paths["depth_visual"] = str(depth_png)

        snapshot = Snapshot(
            session_id=str(session_id),
            image_id=image_id,
            role=str(role),
            rgb=rgb,
            depth=depth,
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            robot=json_ready(robot),
            sequence=int(camera["sequence"]),
            episode_id=str(camera["episode_id"]),
            paths=paths,
        )
        metadata = {
            "session_id": snapshot.session_id,
            "image_id": image_id,
            "role": snapshot.role,
            "sequence": snapshot.sequence,
            "episode_id": snapshot.episode_id,
            "camera": snapshot.camera_metadata(),
            "robot": snapshot.robot,
            "paths": paths,
        }
        artifact_digests = {
            name: _sha256_file(Path(path)) for name, path in paths.items()
        }
        signed_body = {
            "metadata": metadata,
            "artifact_sha256": artifact_digests,
        }
        digest = hmac.new(
            self._secret,
            _canonical_json(signed_body),
            hashlib.sha256,
        ).hexdigest()
        snapshot.integrity_digest = digest
        _atomic_json(
            prefix.with_suffix(".json"),
            {
                **metadata,
                "integrity": {
                    "algorithm": "hmac-sha256+sha256-artifacts",
                    "digest": digest,
                    "artifact_sha256": artifact_digests,
                },
            },
        )
        with self._lock:
            self._latest[str(role)] = (snapshot.session_id, snapshot.image_id)
        return snapshot

    def load(self, session_id: str, image_id: str) -> Snapshot:
        directory = self._directory(session_id)
        image = _validate(image_id, _IDENTIFIER_RE, "image_id")
        prefix = directory / image
        metadata_path = prefix.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"frozen capture {session_id}/{image_id} does not exist"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        integrity = dict(metadata.pop("integrity", {}))
        artifact_digests = dict(integrity.get("artifact_sha256", {}))
        expected_digest = hmac.new(
            self._secret,
            _canonical_json(
                {
                    "metadata": metadata,
                    "artifact_sha256": artifact_digests,
                }
            ),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(str(integrity.get("digest", "")), expected_digest):
            raise ValueError("capture metadata integrity validation failed")
        if (
            str(metadata.get("session_id")) != str(session_id)
            or str(metadata.get("image_id")) != image
        ):
            raise ValueError("capture identity does not match its storage path")
        paths = dict(metadata["paths"])
        if set(paths) != set(artifact_digests):
            raise ValueError("capture artifact manifest is incomplete")
        resolved_directory = directory.resolve()
        verified_paths: dict[str, str] = {}
        for name, stored_path in paths.items():
            artifact = Path(stored_path)
            resolved = artifact.resolve()
            if resolved.parent != resolved_directory:
                raise ValueError("capture artifact path escapes its session directory")
            if not resolved.name.startswith(f"{image}."):
                raise ValueError("capture artifact name does not match image_id")
            if not resolved.is_file():
                raise FileNotFoundError(f"capture artifact is missing: {name}")
            if not hmac.compare_digest(
                _sha256_file(resolved), str(artifact_digests[name])
            ):
                raise ValueError(
                    f"capture artifact integrity validation failed: {name}"
                )
            verified_paths[name] = str(resolved)
        paths = verified_paths
        rgb = np.load(paths["rgb_array"], allow_pickle=False)
        depth = (
            np.load(paths["depth"], allow_pickle=False) if "depth" in paths else None
        )
        camera = metadata["camera"]
        intrinsic = camera.get("intrinsic_matrix")
        extrinsic = camera.get("extrinsic_matrix")
        return Snapshot(
            session_id=str(metadata["session_id"]),
            image_id=str(metadata["image_id"]),
            role=str(metadata["role"]),
            rgb=rgb,
            depth=depth,
            intrinsic_matrix=None
            if intrinsic is None
            else np.asarray(intrinsic, dtype=np.float64),
            extrinsic_matrix=None
            if extrinsic is None
            else np.asarray(extrinsic, dtype=np.float64),
            robot=dict(metadata["robot"]),
            sequence=int(metadata["sequence"]),
            episode_id=str(metadata["episode_id"]),
            paths=paths,
            integrity_digest=str(integrity["digest"]),
        )

    def latest(
        self,
        *,
        role: str = "head",
        session_id: str | None = None,
    ) -> Snapshot:
        with self._lock:
            latest = self._latest.get(str(role))
        if latest is not None and (
            session_id in (None, "") or latest[0] == str(session_id)
        ):
            return self.load(*latest)

        candidates: list[Path] = []
        if session_id not in (None, ""):
            directory = self._directory(str(session_id))
            candidates = list(directory.glob("img_*.json"))
        elif self.root.is_dir():
            candidates = list(self.root.glob("*/images/img_*.json"))
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for metadata_path in candidates:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("role")) == str(role):
                return self.load(
                    str(metadata["session_id"]),
                    str(metadata["image_id"]),
                )
        raise FileNotFoundError(f"no frozen {role} capture is available")


class PlanStore:
    def __init__(
        self, root: str | Path | None = None, secret: bytes | None = None
    ) -> None:
        default_root = os.environ.get(
            "ROBODOJO_INTERFACE_RUNS",
            str(Path(__file__).resolve().parents[2] / ".runtime/artifacts"),
        )
        self.root = Path(root or default_root)
        env_secret = os.environ.get("ROBODOJO_INTERFACE_PLAN_KEY")
        self._secret = secret or (
            env_secret.encode("utf-8") if env_secret else secrets.token_bytes(32)
        )
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}
        self._sessions_by_plan: dict[str, str] = {}

    def _directory(self, session_id: str) -> Path:
        session = _validate(session_id, _SESSION_RE, "session_id")
        return self.root / session / "plans"

    def save(self, session_id: str, record: dict[str, Any]) -> dict[str, Any]:
        directory = self._directory(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            count = self._counts.get(session_id, 0) + 1
            while (directory / f"plan_{count:06d}.json").exists():
                count += 1
            self._counts[session_id] = count
        plan_id = f"plan_{count:06d}"
        body = {**json_ready(record), "plan_id": plan_id, "session_id": session_id}
        digest = hmac.new(
            self._secret, _canonical_json(body), hashlib.sha256
        ).hexdigest()
        signed = {**body, "integrity": {"algorithm": "hmac-sha256", "digest": digest}}
        path = directory / f"{plan_id}.json"
        _atomic_json(path, signed)
        with self._lock:
            self._sessions_by_plan[plan_id] = str(session_id)
        return {**signed, "plan_record_path": str(path)}

    def load(self, session_id: str, plan_id: str) -> dict[str, Any]:
        directory = self._directory(session_id)
        plan = _validate(plan_id, _IDENTIFIER_RE, "plan_id")
        path = directory / f"{plan}.json"
        if not path.is_file():
            raise FileNotFoundError(f"plan {session_id}/{plan_id} does not exist")
        signed = json.loads(path.read_text(encoding="utf-8"))
        integrity = dict(signed.pop("integrity", {}))
        expected = hmac.new(
            self._secret, _canonical_json(signed), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(str(integrity.get("digest", "")), expected):
            raise ValueError("plan integrity validation failed")
        if (
            str(signed.get("session_id")) != str(session_id)
            or str(signed.get("plan_id")) != plan
        ):
            raise ValueError("plan identity does not match its storage path")
        return {**signed, "integrity": integrity, "plan_record_path": str(path)}

    def load_any(self, plan_id: str, session_id: str | None = None) -> dict[str, Any]:
        plan = _validate(plan_id, _IDENTIFIER_RE, "plan_id")
        if session_id not in (None, ""):
            return self.load(str(session_id), plan)
        with self._lock:
            known_session = self._sessions_by_plan.get(plan)
        if known_session is not None:
            return self.load(known_session, plan)
        candidates = list(self.root.glob(f"*/plans/{plan}.json"))
        if len(candidates) != 1:
            if not candidates:
                raise FileNotFoundError(f"plan {plan_id} does not exist")
            raise ValueError("plan_id is ambiguous without session_id")
        return self.load(candidates[0].parents[1].name, plan)
