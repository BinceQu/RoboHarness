"""Long session paths must work without relocating private compiler files."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from tools.util.temporary import nvrtc_compile_tmpdir


def test_long_workspace_path_and_compiler_child(tmp_path):
    root = tmp_path / ("session-" + "x" * 120) / ".runtime" / "compile"
    cwd = Path.cwd()
    with patch.dict(os.environ, {"ROBOT_COMPILE_TMP": str(root)}):
        with nvrtc_compile_tmpdir() as alias:
            assert len(alias) < 64
            assert Path(alias).samefile(root)
            with tempfile.NamedTemporaryFile() as stream:
                assert Path(stream.name).resolve().parent == root
            child = subprocess.run([sys.executable, "-c",
                "import os, pathlib; p=pathlib.Path(os.environ['TMPDIR'])/'child'; "
                "p.write_text('private'); print(p.resolve())"],
                capture_output=True, text=True, check=True)
            assert Path(child.stdout.strip()) == root / "child"
            assert Path.cwd() == cwd
        assert not Path(alias).exists()


def test_nested_exception_restores_environment_and_tempfile(tmp_path):
    before = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    original = tempfile.tempdir
    with patch.dict(os.environ, {"ROBOT_COMPILE_TMP": str(tmp_path)}):
        with pytest.raises(ValueError, match="compiler failure"):
            with nvrtc_compile_tmpdir() as outer:
                with nvrtc_compile_tmpdir() as inner:
                    assert Path(inner).samefile(outer)
                assert os.environ['TMPDIR'] == outer
                assert tempfile.tempdir == outer
                raise ValueError("compiler failure")
    assert {key: os.environ.get(key) for key in before} == before
    assert tempfile.tempdir == original
