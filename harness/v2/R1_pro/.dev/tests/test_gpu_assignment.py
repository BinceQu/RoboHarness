"""The operator's per-session GPU assignment applies to every numerical tool."""
import json
import os
from pathlib import Path

import pytest
from tools.util.settings import configure


@pytest.mark.parametrize('gpu', [0, 2, 3, 4, 5])
def test_explicit_session_gpu_replaces_inherited_gpu(tmp_path, monkeypatch, gpu):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    config = {'workspace': str(workspace), 'gpu': gpu, 'state_dir': str(workspace / '.runtime'),
        'mesh_root': str(workspace / 'robot_config'), 'numerical_python': str(workspace / '.venv/bin/python'),
        'compile_tmp': str(workspace / '.runtime/compile')}
    (workspace / '.session.json').write_text(json.dumps(config))
    monkeypatch.setenv('ROBOT_WORKSPACE', str(workspace))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'wrong')
    # configure changes process environment; restore every key after the test.
    before = dict(os.environ)
    try:
        configure()
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert os.environ['BEHAVIOR_INTERFACE_PHYSICAL_GPU'] == str(gpu)
        assert os.environ['BEHAVIOR_EVAL_TEST_PHYSICAL_GPU'] == str(gpu)
        assert os.environ['OFFICIAL_V2_LITE_OCCUPANCY_DEVICE'] == 'cuda:0'
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.mark.parametrize('gpu', [None, -1, True, '0'])
def test_missing_or_invalid_assignment_is_rejected(tmp_path, monkeypatch, gpu):
    (tmp_path / '.session.json').write_text(json.dumps({'workspace': str(tmp_path), 'gpu': gpu,
        'state_dir': str(tmp_path / '.runtime'), 'compile_tmp': str(tmp_path / '.runtime/compile')}))
    monkeypatch.setenv('ROBOT_WORKSPACE', str(tmp_path))
    before = dict(os.environ)
    try:
        with pytest.raises(RuntimeError, match='explicitly assign'):
            configure()
    finally:
        os.environ.clear()
        os.environ.update(before)
