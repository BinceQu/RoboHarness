"""Short private temporary path for local numerical compilation."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import threading


_COMPILE_TEMP_LOCK = threading.RLock()


@contextmanager
def nvrtc_compile_tmpdir():
    """Keep files in the workspace while giving NVRTC a short Linux path.

    NVRTC rejects long TMPDIR strings even when the directory is writable.
    An open directory descriptor provides a short alias without moving files,
    sharing a /tmp cache, or changing the process's working directory. Include
    our PID so compiler children can also use the alias while this context lives.
    """
    with _COMPILE_TEMP_LOCK:
        root = os.environ.get("ROBOT_COMPILE_TMP")
        if not root:
            raise RuntimeError("ROBOT_COMPILE_TMP must name this session's compilation directory")
        Path(root).mkdir(parents=True, exist_ok=True)
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        saved = {name: os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP")}
        old_tempdir = tempfile.tempdir
        try:
            short_path = f"/proc/{os.getpid()}/fd/{descriptor}"
            if not Path(short_path).samefile(root):
                raise RuntimeError("Compilation directory descriptor does not reference this workspace")
            for name in saved:
                os.environ[name] = short_path
            tempfile.tempdir = short_path
            yield short_path
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            tempfile.tempdir = old_tempdir
            os.close(descriptor)
