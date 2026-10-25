from types import SimpleNamespace
from pathlib import Path
import json

import pytest

from health import ConnectionHealth
import fleet
from lifecycle import read_json, write_json


def snapshot(age, tool=None):
    return {'evaluator': {'heartbeat_age_s': age, 'observation_age_s': 90000},
            'scheduler': {'current_tool': tool}}


def test_heartbeat_distinguishes_frozen_physics_and_disconnection():
    health = ConnectionHealth()
    assert health.check(snapshot(.1), active=True) is None
    assert 'heartbeat' in health.check(snapshot(181), active=True)
    assert health.check(snapshot(181, 'plan_grasp'), active=True) is None
    assert 'heartbeat' in health.check(snapshot(901, 'plan_grasp'), active=True)
    assert health.check(snapshot(90000), active=False) is None


def test_socket_outage_has_grace_and_recovers():
    health = ConnectionHealth()
    assert health.check(None, active=True, now=0) is None
    assert health.check(None, active=True, now=119) is None
    assert 'socket' in health.check(None, active=True, now=120)
    assert health.check(snapshot(.1), active=True, now=121) is None
    assert health.check(None, active=True, now=130) is None


def test_fleet_refuses_mismatched_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'ROOT', tmp_path)
    write_json(tmp_path/'runs/run/status.json', {'gpu': 5, 'port': 123, 'task': 'task'})
    with pytest.raises(RuntimeError, match='identity'):
        fleet.inspect_slot({'gpu': 1, 'port': 123, 'task': 'task'}, 'run')


def test_fleet_never_signals_an_unverified_or_outside_run(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='outside'):
        fleet.stop_attempt('../../outside', 'test')
    write_json(tmp_path/'runs/run/status.json', {'supervisor': {'pid': 22, 'start_ticks': 1}})
    monkeypatch.setattr(fleet, 'alive', lambda _: True)
    monkeypatch.setattr(Path, 'read_bytes', lambda _: b'python\0unrelated.py\0')
    monkeypatch.setattr(fleet.os, 'kill', lambda *_: pytest.fail('must not signal'))
    with pytest.raises(RuntimeError, match='command'):
        fleet.stop_attempt('run', 'test')


def test_completed_attempt_is_preserved_before_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'ROOT', tmp_path)
    official = {'score': .8, 'phase': 'complete'}
    write_json(tmp_path/'runs/run/status.json', official)
    write_json(tmp_path/'runs/run/official_result.json', {'score': 80})
    monkeypatch.setattr(fleet, 'alive', lambda _: False)
    fleet.stop_attempt('run', 'complete')
    assert read_json(tmp_path/'runs/run/status.json') == official
    assert read_json(tmp_path/'runs/run/official_result.json') == {'score': 80}
    assert read_json(tmp_path/'runs/run/fleet_retired.json')['score'] == .8


def test_native_launch_cannot_inherit_disabled_compaction(tmp_path, monkeypatch):
    import run
    write_json(tmp_path/'prompt.txt', 'test')
    monkeypatch.setenv('DISABLE_COMPACT', '1')
    monkeypatch.setenv('DISABLE_AUTO_COMPACT', '1')
    captured = {}
    def popen(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pid=42)
    monkeypatch.setattr(run.subprocess, 'Popen', popen)
    monkeypatch.setattr(run, 'identity', lambda _: {'pid': 42, 'start_ticks': 1})
    args = SimpleNamespace(model='NVFP4', base_url='http://local:31000')
    run.start_agent(args, tmp_path, tmp_path)
    env = captured['env']
    assert 'DISABLE_COMPACT' not in env and 'DISABLE_AUTO_COMPACT' not in env
    assert env['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] == '180000'
    assert env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] == '262144'
