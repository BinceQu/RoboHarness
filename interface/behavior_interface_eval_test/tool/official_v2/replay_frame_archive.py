"""Bounded hot cache for lossless replay payloads backed by local temp files.

Only compressed, policy-owned observation bytes enter this module. Chunk
references keep in-flight readers valid across history eviction / episode
reset; dropping the last reference closes the anonymous temporary file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from typing import BinaryIO


REPLAY_ARCHIVE_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass
class _ArchiveChunk:
    stream: BinaryIO
    size: int = 0

    def __del__(self) -> None:
        self.stream.close()


@dataclass
class ReplayFramePayload:
    chunk: _ArchiveChunk
    offset: int
    gray_size: int
    depth_size: int
    digest: bytes
    cached: tuple[bytes, bytes] | None

    def read(self) -> tuple[bytes, bytes]:
        # Keep a local reference: the compression worker may evict this cache
        # row concurrently. Its immutable bytes remain valid for this reader.
        cached = self.cached
        if cached is not None:
            return cached
        size = self.gray_size + self.depth_size
        payload = os.pread(self.chunk.stream.fileno(), size, self.offset)
        if len(payload) != size or hashlib.sha256(payload).digest() != self.digest:
            raise OSError("lossless replay archive payload failed integrity check")
        return payload[:self.gray_size], payload[self.gray_size:]


class ReplayFrameArchive:
    """Archive in compression workers; retain the existing RAM cache budget.

    The owner bounds total retained bytes / frames and releases history refs.
    This class retains at most one writable chunk plus the bounded hot cache.
    Disk usage beyond retained payloads is limited to the hot cache, chunk
    edge slack and the owner's bounded pending queue. No per-frame files, fsync or
    network storage are used on the evaluator observation callback.
    """

    def __init__(
        self, memory_max_bytes: int, *,
        chunk_bytes: int = REPLAY_ARCHIVE_CHUNK_BYTES,
        cache_max_frames: int = 2048,
    ):
        if int(memory_max_bytes) < 0 or int(chunk_bytes) < 1 or int(cache_max_frames) < 1:
            raise ValueError("replay cache bytes must be nonnegative and chunk / frame bounds positive")
        self.memory_max_bytes = int(memory_max_bytes)
        self._cache_max_frames = int(cache_max_frames)
        self._chunk_bytes = int(chunk_bytes)
        self._lock = threading.Lock()
        self._chunk: _ArchiveChunk | None = None
        self._cache: deque[ReplayFramePayload] = deque()
        self._cache_bytes = 0
        self._published_cache_bytes = 0

    def store(self, gray: bytes, depth: bytes) -> ReplayFramePayload:
        size = len(gray) + len(depth)
        digest = hashlib.sha256(gray)
        digest.update(depth)
        with self._lock:
            if self._chunk is None or self._chunk.size + size > self._chunk_bytes:
                # Explicit local scratch avoids TMPDIR pointing at the NFS
                # submission directory. TemporaryFile is private and removed
                # automatically, including after a process crash.
                self._chunk = _ArchiveChunk(tempfile.TemporaryFile(
                    prefix="official-track-replay-", dir="/tmp", buffering=0,
                ))
            chunk = self._chunk
            offset = chunk.size
            try:
                if chunk.stream.write(gray) != len(gray) or chunk.stream.write(depth) != len(depth):
                    raise OSError("incomplete lossless replay archive write")
            except OSError:
                # Never reuse offsets after a partial write / ENOSPC. Earlier
                # payloads in this chunk remain readable via their own refs.
                self._chunk = None
                raise
            chunk.size += size
            payload = ReplayFramePayload(
                chunk, offset, len(gray), len(depth), digest.digest(), (gray, depth),
            )
            self._cache.append(payload)
            self._cache_bytes += size
            while self._cache_bytes > self.memory_max_bytes or len(self._cache) > self._cache_max_frames:
                expired = self._cache.popleft()
                self._cache_bytes -= expired.gray_size + expired.depth_size
                expired.cached = None
            self._published_cache_bytes = self._cache_bytes
            return payload

    def status(self) -> dict[str, int | str]:
        # Health / ingest must never wait for a compression worker's disk I/O.
        return {
            "replay_archive_backend": "local_anonymous_tempfile_with_bounded_hot_cache",
            "replay_memory_cache_bytes": self._published_cache_bytes,
            "replay_memory_cache_max_bytes": self.memory_max_bytes,
            "replay_memory_cache_max_frames": self._cache_max_frames,
            "replay_archive_chunk_bytes": self._chunk_bytes,
        }
