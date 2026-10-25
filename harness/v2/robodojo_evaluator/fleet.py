"""Keep the explicitly configured RoboDojo slots running; retain every attempt.

Run under a user systemd service, outside the launching terminal's lifecycle.
Only receipts below this evaluator's runs/ directory may be stopped. Replacing
a slot creates a new workspace; a failed attempt is never given a zero score.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from health import ConnectionHealth, workspace_status
from lifecycle import identity, now, read_json, write_json
from run import ExistingProcess, stop_child

ROOT = Path(__file__).resolve().parent
TERMINAL = {'complete', 'failed', 'stopped'}


def alive(receipt):
    current = identity(receipt.get('pid', -1))
    return bool(current and current['start_ticks'] == receipt.get('start_ticks'))


def inspect_slot(slot, run_id):
    run = ROOT/'runs'/run_id
    state = read_json(run/'status.json')
    owned = read_json(run/'fleet_run.json')
    if state and any(state.get(key) != slot[key] for key in ('task', 'gpu', 'port')):
        raise RuntimeError(f'{run_id}: slot identity does not match; refusing replacement')
    receipt = state.get('supervisor') or owned.get('supervisor') or {}
    report = {'run_id': run_id, 'phase': state.get('phase', 'preparing'),
              'supervisor_alive': alive(receipt), 'state': state, 'owned': owned}
    if state.get('workspace') and state.get('phase') not in TERMINAL:
        try:
            report['connection'] = workspace_status(state['workspace'])
        except (OSError, ValueError, RuntimeError) as exc:
            report['connection_error'] = str(exc)
    return report


def replacement_reason(report, generation):
    state, owned = report['state'], report['owned']
    if not report['supervisor_alive']:
        return 'supervisor exited' if state.get('phase') not in TERMINAL else state['phase']
    if owned.get('generation') != generation:
        return 'replace previous generation with verified fixes'
    if state.get('phase') in TERMINAL:
        return state['phase']
    # Startup copying/loading has its own deadline; an unresponsive live
    # supervisor must not make a slot disappear forever.
    stamp = state.get('updated_at') or owned.get('launched_at')
    if stamp:
        age = time.time()-datetime.fromisoformat(stamp).timestamp()
        limit = 2700 if state.get('phase') in (None, 'starting', 'loading') else 180
        if age > limit:
            return f'supervisor status stopped updating ({age:.0f}s)'
    for key in ('evaluator', 'policy'):
        if state.get(key) and not alive(state[key]):
            return f'{key} exited'
    return None


def stop_attempt(run_id, reason):
    run = (ROOT/'runs'/run_id).resolve()
    if run.parent != (ROOT/'runs').resolve():
        raise ValueError('Run is outside RoboDojo runs/')
    state = read_json(run/'status.json')
    owned = read_json(run/'fleet_run.json')
    write_json(run/'stop_requested.json', {'at': now(), 'reason': reason, 'by': 'robodojo-fleet'})
    receipt = state.get('supervisor') or owned.get('supervisor') or {}
    if alive(receipt):
        command = Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().split(b'\0')
        if str(ROOT/'run.py').encode() not in command or run_id.encode() not in command:
            raise RuntimeError('Supervisor command does not match run receipt')
        os.kill(receipt['pid'], signal.SIGTERM)
        deadline = time.monotonic()+60
        while alive(receipt) and time.monotonic() < deadline:
            time.sleep(.2)
        if alive(receipt):
            os.kill(receipt['pid'], signal.SIGKILL)
    # A killed supervisor may leave its explicitly owned children. Never use
    # a GPU-wide, user-wide or command-pattern kill.
    state = read_json(run/'status.json') or state
    state = {**read_json(run/'children.json'), **state}
    for key in ('agent', 'evaluator', 'policy'):
        if state.get(key):
            stop_child(ExistingProcess(state[key]), grace_seconds=5)
    write_json(run/'fleet_retired.json', {'at': now(), 'reason': reason,
        'final_phase': state.get('phase'), 'score': state.get('score')})


def launch_slot(slot, config):
    deadline = time.monotonic()+30
    while True:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('127.0.0.1', slot['port']))
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Policy port {slot['port']} did not become free")
            time.sleep(.5)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_id = f"rd12-t{slot['number']}-s0-l0-{stamp}-{time.time_ns()%1000000}"
    run = ROOT/'runs'/run_id
    run.mkdir(parents=True)
    command = [sys.executable, str(ROOT/'run.py'), '--gpu', str(slot['gpu']),
        '--port', str(slot['port']), '--task', slot['task'], '--seed', '0', '--layout', '0',
        '--id', run_id, '--prompt', slot['prompt'], '--model', config['model'],
        '--base-url', config['base_url'], '--session-timeout', '259200']
    if slot.get('without_evaluator_planner'):
        command.append('--without-evaluator-planner')
    with (run/'supervisor.log').open('x') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'supervisor': identity(process.pid), 'launched_at': now(),
               'generation': config['generation'], 'command': command}
    write_json(run/'fleet_run.json', receipt)
    return run_id, process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    config_path = args.config.resolve(strict=True)
    directory = config_path.parent
    config = read_json(config_path)
    slots = config['slots']
    if len({s['number'] for s in slots}) != len(slots) or len({s['port'] for s in slots}) != len(slots):
        raise ValueError('Duplicate fleet slot/port')
    lock = (directory/'fleet.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs_path = directory/'runs.json'
    active = read_json(runs_path)
    for slot in slots:
        active.setdefault(slot['number'], slot['initial_run'])
    write_json(runs_path, active)
    stopping = False
    def interrupt(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    checks = {s['number']: ConnectionHealth() for s in slots}
    retry_after = {}
    children = []
    pending = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        while not stopping or pending:
            # Reap detached supervisors that this fleet started.
            children = [p for p in children if p.poll() is None]
            for gpu, (number, future) in list(pending.items()):
                if not future.done():
                    continue
                try:
                    run_id, child = future.result()
                    active[number] = run_id
                    children.append(child)
                    retry_after[number] = time.monotonic()+30
                    checks[number] = ConnectionHealth()
                    write_json(runs_path, active)
                    print(json.dumps({'at': now(), 'slot': number, 'launched': run_id}), flush=True)
                except Exception as exc:
                    print(json.dumps({'at': now(), 'slot': number, 'recovery_error': str(exc)}), flush=True)
                    retry_after[number] = time.monotonic()+30
                del pending[gpu]
            reports = {}
            for slot in slots:
                number = slot['number']
                try:
                    report = inspect_slot(slot, active[number])
                    fault = replacement_reason(report, config['generation'])
                    if not fault:
                        fault = checks[number].check(report.get('connection'),
                            active=report['phase'] in ('running', 'waiting_for_model', 'holding_for_official_score'))
                    report['recovery_reason'] = fault
                    reports[number] = report
                except Exception as exc:
                    reports[number] = {'error': str(exc)}
            # At most one replacement/initialization per GPU; fill already
            # vacant slots before replacing old residents on that GPU.
            for slot in sorted(slots, key=lambda s: reports[s['number']].get('supervisor_alive', True)):
                if stopping:
                    break
                number, gpu = slot['number'], slot['gpu']
                report = reports[number]
                if not report.get('recovery_reason') or gpu in pending:
                    continue
                if time.monotonic() < retry_after.get(number, 0):
                    continue
                loading = any(other['gpu'] == gpu and other['number'] != number and
                    reports[other['number']].get('owned', {}).get('generation') == config['generation'] and
                    reports[other['number']].get('supervisor_alive') and
                    reports[other['number']].get('phase') in ('starting', 'preparing', 'loading')
                    for other in slots)
                if loading:
                    continue
                def replace(slot=slot, old=active[number], reason=report['recovery_reason']):
                    stop_attempt(old, reason)
                    return launch_slot(slot, config)
                pending[gpu] = (number, pool.submit(replace))
            write_json(directory/'fleet-status.json', {'at': now(), 'process': identity(os.getpid()),
                'generation': config['generation'], 'slots': reports,
                'recovering': {str(gpu): number for gpu, (number, _) in pending.items()}})
            for _ in range(10):
                if stopping:
                    break
                time.sleep(.5)
    # Deliberately retain live supervisors on service restart. The next fleet
    # instance reattaches by PID + start_ticks and never duplicates a slot.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
