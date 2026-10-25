"""Relocate the complete environment and prove no shared implementation files."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


TEMPLATE = Path(__file__).resolve().parents[2]/'ARX_X5'


def test_copied_session_is_independent_and_runs_without_source_paths():
    spec = importlib.util.spec_from_file_location('x5_session', TEMPLATE/'session.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='x5-session-') as temporary:
        destination = Path(temporary)/'long path with spaces'/'ARX_X5'
        root = module.create(destination, 'independent-unit', 'smoke')
        environment = subprocess.run([str(root/'.venv/bin/python'), '-m', 'tools.util.environment', '--check'],
            cwd=root, capture_output=True, text=True, check=True)
        assert json.loads(environment.stdout)['ok'] is True
        assert list((root/'tool_call_history').iterdir()) == [root/'tool_call_history/README.md']
        assert not (root/'perception/latest.json').exists()
        assert json.loads((root/'.runtime/state.json').read_text())['counter'] == 0
        for path in ['tools/capture_head_camera.py', '.venv/bin/pytest',
                     '.venv/lib/python3.10/site-packages/numpy/__init__.py']:
            assert (root/path).stat().st_ino != (TEMPLATE/path).stat().st_ino
        source = TEMPLATE/'tools/capture_head_camera.py'
        original = source.read_bytes()
        (root/'tools/capture_head_camera.py').write_text('# local experiment only\n')
        assert source.read_bytes() == original
