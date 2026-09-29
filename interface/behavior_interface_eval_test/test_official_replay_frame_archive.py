"""Lossless cold replay, bounded RAM and reader lifetime regressions."""

import gc
import os
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from behavior_interface_eval_test.tool.official_v2.replay_frame_archive import ReplayFrameArchive


class ReplayFrameArchiveTest(unittest.TestCase):
    def test_hot_read_keeps_original_bytes_and_avoids_disk(self):
        archive = ReplayFrameArchive(1024)
        gray, depth = b"gray" * 11, b"depth" * 17
        payload = archive.store(gray, depth)
        with mock.patch("os.pread", side_effect=AssertionError("hot replay read disk")):
            restored = payload.read()
        self.assertIs(restored[0], gray)
        self.assertIs(restored[1], depth)

    def test_cache_eviction_retains_exact_cold_bytes(self):
        archive = ReplayFrameArchive(8, chunk_bytes=16)
        payloads = [archive.store(bytes([i]) * 3, bytes([255 - i]) * 5) for i in range(40)]
        self.assertIsNone(payloads[0].cached)
        self.assertEqual(archive.status()["replay_memory_cache_bytes"], 8)
        for i, payload in enumerate(payloads):
            self.assertEqual(payload.read(), (bytes([i]) * 3, bytes([255 - i]) * 5))

    def test_cold_corruption_and_truncation_fail_closed(self):
        for truncate in (False, True):
            with self.subTest(truncate=truncate):
                archive = ReplayFrameArchive(0)
                payload = archive.store(b"rgb", b"depth")
                if truncate:
                    os.ftruncate(payload.chunk.stream.fileno(), 1)
                else:
                    os.pwrite(payload.chunk.stream.fileno(), b"x", payload.offset)
                with self.assertRaisesRegex(OSError, "integrity check"):
                    payload.read()

    def test_chunk_lifetime_follows_pinned_replay_readers(self):
        archive = ReplayFrameArchive(0, chunk_bytes=4)
        first = archive.store(b"ab", b"cd")
        chunk = weakref.ref(first.chunk)
        stream = first.chunk.stream
        second = archive.store(b"ef", b"gh")
        del archive
        gc.collect()
        self.assertEqual(first.read(), (b"ab", b"cd"))
        self.assertEqual(second.read(), (b"ef", b"gh"))
        del first
        gc.collect()
        self.assertIsNone(chunk())
        self.assertTrue(stream.closed)

    def test_concurrent_archiving_preserves_each_frame(self):
        archive = ReplayFrameArchive(64, chunk_bytes=128)
        def store(i):
            return archive.store(bytes([i]) * 29, bytes([255-i]) * 41)
        with ThreadPoolExecutor(max_workers=4) as pool:
            payloads = list(pool.map(store, range(80)))
        for i, payload in enumerate(payloads):
            self.assertEqual(payload.read(), (bytes([i]) * 29, bytes([255-i]) * 41))
        self.assertLessEqual(archive.status()["replay_memory_cache_bytes"], 64)

    def test_health_does_not_wait_for_archive_writer(self):
        archive = ReplayFrameArchive(1024)
        with archive._lock:
            self.assertEqual(archive.status()["replay_memory_cache_bytes"], 0)

    def test_tiny_payloads_cannot_accumulate_unbounded_cache_metadata(self):
        archive = ReplayFrameArchive(1024, cache_max_frames=2)
        payloads = [archive.store(b"g", b"d") for _ in range(10)]
        self.assertEqual(archive.status()["replay_memory_cache_bytes"], 4)
        self.assertEqual(sum(payload.cached is not None for payload in payloads), 2)
        self.assertEqual(payloads[0].read(), (b"g", b"d"))

    def test_write_failure_is_reported(self):
        archive = ReplayFrameArchive(1024)
        with mock.patch("tempfile.TemporaryFile", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                archive.store(b"gray", b"depth")

    def test_partial_write_does_not_poison_next_capture_or_older_payloads(self):
        archive = ReplayFrameArchive(0)
        previous = archive.store(b"old gray", b"old depth")
        with mock.patch.object(previous.chunk.stream, "write", return_value=0):
            with self.assertRaisesRegex(OSError, "incomplete"):
                archive.store(b"new gray", b"new depth")
        following = archive.store(b"next gray", b"next depth")
        self.assertIsNot(previous.chunk, following.chunk)
        self.assertEqual(previous.read(), (b"old gray", b"old depth"))
        self.assertEqual(following.read(), (b"next gray", b"next depth"))


if __name__ == "__main__":
    unittest.main()
