"""Observation delivery must work without any host image-converter binary."""
from io import BytesIO
import subprocess

import numpy as np
from PIL import Image
import pytest

from tools.util.media import MediaExtractor


@pytest.fixture
def extractor(monkeypatch):
    def reject_process(*args, **kwargs):
        pytest.fail("Image delivery must use the workspace libraries, not a host executable")
    monkeypatch.setattr(subprocess, "Popen", reject_process)
    monkeypatch.setenv("PATH", "")
    return MediaExtractor(None, max_bytes=16 * 1024 * 1024, max_images=1)


def test_native_head_compression_preserves_pixels_and_ignores_exif(extractor, tmp_path):
    y, x = np.indices((720, 720))
    pixels = np.stack((30 + x // 3, 20 + y // 3, np.full_like(x, 100)), axis=-1)
    pixels += np.random.default_rng(42).integers(-8, 9, size=pixels.shape)
    pixels = pixels.astype(np.uint8)
    markers = {(100, 110): [220, 30, 15], (610, 130): [20, 210, 40],
               (130, 590): [30, 60, 220], (620, 630): [230, 200, 20]}
    for (u, v), color in markers.items():
        pixels[v - 15:v + 16, u - 15:u + 16] = color
    source = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6  # Deliberately misleading: RGB-D pixels must not rotate.
    path = tmp_path / "head.png"
    source.save(path, exif=exif)
    assert path.stat().st_size > extractor.max_model_bytes
    selection = extractor.extract({"rgb_path": str(path), "image_id": "head-current", "feed": "head"},
                                  tool_name="capture_head_camera")
    assert not selection.warnings and len(selection.selected) == 1
    media = selection.selected[0]
    assert (media.width, media.height, media.source_width, media.source_height) == (720, 720, 720, 720)
    assert media.source_image_id == "head-current" and media.source_feed == "head"
    assert media.mime_type == "image/jpeg" and len(media.data) <= extractor.max_model_bytes
    with Image.open(BytesIO(media.data)) as decoded:
        assert decoded.size == (720, 720) and not decoded.getexif()
        for point, expected in markers.items():
            assert np.max(np.abs(np.array(decoded.getpixel(point)) - expected)) <= 4
        difference = np.asarray(decoded).astype(float) - pixels.astype(float)
        assert np.sqrt(np.mean(difference ** 2)) < 8


def test_small_jpeg_passes_through_byte_for_byte(extractor, tmp_path):
    path = tmp_path / "head.jpg"
    Image.new("RGB", (720, 720), (12, 34, 56)).save(path, quality=95)
    selection = extractor.extract({"rgb_path": str(path)})
    assert not selection.warnings
    assert selection.selected[0].data == path.read_bytes()


def test_large_inspection_image_keeps_aspect_and_source_dimensions(extractor, tmp_path):
    path = tmp_path / "wrist.png"
    Image.new("RGB", (1280, 720), (23, 45, 67)).save(path)
    selection = extractor.extract({"rgb_path": str(path), "feed": "left_wrist"})
    assert not selection.warnings
    media = selection.selected[0]
    assert (media.width, media.height) == (720, 405)
    assert (media.source_width, media.source_height) == (1280, 720)
