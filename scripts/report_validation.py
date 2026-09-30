#!/usr/bin/env python3
"""Compare completed official rollouts with their archived references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboharness.runner import atomic_json, official_result, read_json
from roboharness.native_context import contract as native_contract, inspect_listing


def local_process_state(identity: dict) -> dict:
    pid, birth = identity.get('pid'), identity.get('start')
    result = {'pid': pid, 'expected_start': birth}
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not birth:
        return {**result, 'state': 'unknown_identity'}
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        actual_birth, state = fields[19], fields[0]
    except FileNotFoundError:
        return {**result, 'state': 'missing'}
    except (OSError, IndexError):
        return {**result, 'state': 'unreadable'}
    if actual_birth != str(birth):
        return {**result, 'state': 'pid_reused', 'observed_start': actual_birth}
    return {**result, 'state': 'exited' if state in ('Z', 'X') else 'alive'}


def local_liveness(path: Path) -> dict:
    processes_path = path / 'processes.json'
    processes = read_json(processes_path) if processes_path.is_file() else {}
    session_path = path / 'active_session.json'
    session = read_json(session_path) if session_path.is_file() else {}
    identities = {role: identity for role, identity in processes.items()
                  if role in ('guardian', 'interface', 'gate', 'evaluator')}
    if session.get('owner_pid'):
        identities['controller'] = {'pid': session['owner_pid'], 'start': session.get('owner_start')}
    # An agent may exit normally while its official scoring JSON is being saved.
    states = {role: local_process_state(identity) for role, identity in identities.items()}
    dead = [role for role, record in states.items() if record['state'] in ('missing', 'pid_reused', 'exited')]
    required = {'controller', 'interface', 'gate', 'evaluator'}
    verified = required.issubset(states) and all(record['state'] == 'alive' for record in states.values())
    return {'state': 'dead' if dead else 'alive' if verified else 'unverified',
            'failed_roles': dead, 'processes': states}


def native_contexts(path: Path, task: str, expected: dict, finished: set[int]) -> list[dict]:
    rows = []
    for iid in expected:
        folder = path / f'instance_{iid}' / 'claude-home'
        transcripts = list(folder.glob('projects/*/*.jsonl'))
        if len(transcripts) > 1:
            result = {'state': 'ambiguous', 'reason': 'Multiple native transcripts for one case'}
        elif not transcripts:
            result = {'state': 'missing' if iid in finished else 'pending' if folder.exists() else 'not_started'}
        else:
            try:
                result = inspect_listing(transcripts[0], native_contract(task, iid))
                result['transcript'] = str(transcripts[0].relative_to(path))
            except (OSError, ValueError) as error:
                result = {'state': 'unreadable', 'reason': str(error)}
            if iid in finished and result['state'] == 'pending':
                result['state'] = 'missing'
        rows.append({'instance_id': iid, **result})
    return rows


def collect(path: Path, *, check_live: bool = False) -> dict:
    plan = read_json(path / 'plan.json')
    summary = read_json(path / 'summary.json')
    task = plan['task_config']
    expected = {case['instance_id']: case for case in plan['cases']}
    completed = []
    seen = set()
    for case in summary.get('cases', []):
        iid = case['instance_id']
        if iid not in expected or iid in seen:
            raise ValueError(f'Unexpected or duplicate instance in {path}: {iid}')
        seen.add(iid)
        result_path = path / case['result']
        if not result_path.resolve().is_relative_to((path / 'output').resolve()):
            raise ValueError(f'Result is outside this run output: {result_path}')
        result = official_result(result_path, task, iid)
        if result is None or hashlib.sha256(result_path.read_bytes()).hexdigest() != case['result_sha256']:
            raise ValueError(f'Official result is missing or has changed: {result_path}')
        q = result['q_score']['final']
        reference = expected[iid]
        archive_q = reference.get('archive_reported_q', reference['reference_q'])
        completed.append({'instance_id': iid, 'q': q, 'archive_q': archive_q,
                          'raw_reference_q': reference['reference_q'], 'delta_archive_q': q - archive_q,
                          'steps': result['steps'], 'success': result['success'],
                          'finish_reason': case.get('finish_reason'),
                          'result_sha256': case['result_sha256'],
                          'source_result': case['result']})
    status = summary['status']
    if status == 'starting' and any(path.glob('instance_*/case.json')):
        status = 'running'
    liveness = None
    error = summary.get('error')
    if check_live and status in ('starting', 'running'):
        liveness = local_liveness(path)
        if liveness['state'] == 'dead':
            # Normal completion writes its terminal summary before process cleanup.
            # Re-read it to avoid reporting that cleanup as an interruption.
            if read_json(path / 'summary.json').get('status') in ('complete', 'failed'):
                return collect(path, check_live=check_live)
            status = 'interrupted'
            error = 'Recorded local processes are no longer alive: ' + ', '.join(liveness['failed_roles'])
    if status == 'complete' and len(completed) != len(expected):
        raise ValueError(f'Run claims completion without all official scores: {path}')
    archive_mean = sum(c.get('archive_reported_q', c['reference_q']) for c in expected.values()) / len(expected)
    mean = sum(c['q'] for c in completed) / len(completed) if completed else None
    completed_archive_mean = sum(c['archive_q'] for c in completed) / len(completed) if completed else None
    superseded_path = path / 'superseded_context.json'
    superseded = read_json(superseded_path) if superseded_path.is_file() else None
    caveats = [{'instance_id': case['instance_id'], 'reason': reason}
               for case in expected.values() for reason in case.get('reproduction_caveats', [])]
    caveats.extend(
        {'instance_id': case['instance_id'],
         'reason': 'The wall-clock safety timeout forced episode submission; '
                   'this is not termination under the archived step budget or model completion.'}
        for case in completed if case.get('finish_reason') == 'wall_timeout'
    )
    contexts = (native_contexts(path, task['task'], expected, seen)
                if plan['harness'] == 'claude_code' else [])
    caveats.extend(
        {'instance_id': context['instance_id'],
         'reason': 'Initial native Skill listing is ' + context['state'] +
                   '; a complete recorded listing match is required, including descriptions.'}
        for context in contexts if context['state'] not in ('match', 'pending', 'not_started')
    )
    complete = (status == 'complete' and superseded is None and not caveats
                and not plan.get('diagnostic_only', False))
    matches_cases = (all(abs(c['delta_archive_q']) < 1e-6 for c in completed)
                     if complete else None)
    return {'task': task['task'], 'task_name': task['task_name'], 'run_directory': path.name,
            'status': status, 'model': plan['model'], 'harness': plan['harness'],
            'n_finished': len(completed), 'n_expected': len(expected), 'cases': completed,
            'mean_q': mean, 'archive_mean_q': archive_mean,
            'completed_archive_mean_q': completed_archive_mean,
            'delta_archive_mean_q': mean - archive_mean if complete else None,
            'matches_archive_mean': abs(mean - archive_mean) < 1e-6 if complete else None,
            'matches_archive_cases': matches_cases,
            'reproduction_verified': (matches_cases and abs(mean - archive_mean) < 1e-6)
                                     if complete else False,
            'diagnostic_only': plan.get('diagnostic_only', False),
            'error': error, 'superseded': superseded,
            'reported_status': summary['status'], 'local_liveness': liveness,
            'reproduction_caveats': caveats, 'native_skill_contexts': contexts}


def write_report(output: Path, rows: list[dict], run_paths: list[Path]):
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    atomic_json(output / 'report.json', {'updated_at_utc': timestamp, 'runs': rows})
    lines = ['# GPU validation results', '', f'Updated: {timestamp}', '',
             'Only fresh official evaluator JSON contributes to new scores. An incomplete',
             'run has no final mean comparison and is not a successful reproduction claim.', '',
             'Comparisons cover only the selected instances. Run all five archived instances',
             'to compare a complete task mean.', '',
             '| Task | Status | Local processes | Completed | New Q (completed cases) | Archived Q (same completed cases) | Final difference | Every case matches |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row, run in zip(rows, run_paths):
        mean = '—' if row['mean_q'] is None else f'{row["mean_q"]:.6f}'
        archive = ('—' if row['completed_archive_mean_q'] is None
                   else f'{row["completed_archive_mean_q"]:.6f}')
        delta = '—' if row['delta_archive_mean_q'] is None else f'{row["delta_archive_mean_q"]:+.6f}'
        matches = '—' if row['matches_archive_cases'] is None else ('yes' if row['matches_archive_cases'] else 'no')
        local = row['local_liveness']['state'] if row['local_liveness'] else 'not checked'
        lines.append(f'| {row["task"]} | {row["status"]} | {local} | {row["n_finished"]}/{row["n_expected"]} | '
                     f'{mean} | {archive} | {delta} | {matches} |')
        for case in row['cases']:
            source = run / case['source_result']
            target = output / row['task'] / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
    for row in rows:
        if row['error']:
            lines += ['', f'**{row["task"]}: {row["status"]}.** {row["error"]}']
        for caveat in row['reproduction_caveats']:
            lines += ['', f'**{row["task"]}/{caveat["instance_id"]}: reproduction limitation.** '
                      f'{caveat["reason"]}']
        if row['superseded']:
            note = row['superseded']
            lines += ['', f'**{row["task"]}: superseded diagnostic attempt.** {note["reason"]}',
                      f'Replacement run: `{note["replacement_run"]}`.']
    lines += ['', 'Case scores, hashes and failure details are in [report.json](report.json).', '']
    temporary = output / 'README.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(output / 'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'validation_results/latest')
    parser.add_argument('--watch', action='store_true', help='Update until every selected run completes or fails')
    parser.add_argument('--check-live', action='store_true',
                        help='Verify recorded process identities on this Linux host; do not use for copied runs')
    parser.add_argument('--require-match', action='store_true',
                        help='Exit 2 unless all selected cases have verified matching archived scores')
    args = parser.parse_args()
    previous = None
    while True:
        rows = [collect(path.resolve(), check_live=args.check_live) for path in args.runs]
        if len({r['task'] for r in rows}) != len(rows):
            raise ValueError('Choose one run per task; separate attempts must be reported separately.')
        signature = json.dumps(rows, sort_keys=True)
        if signature != previous:
            write_report(args.output.resolve(), rows, [p.resolve() for p in args.runs])
            print(' | '.join(f'{r["task"]}: {r["status"]}, {r["n_finished"]}/{r["n_expected"]}' for r in rows), flush=True)
            previous = signature
        if not args.watch or all(r['status'] in ('complete', 'failed', 'interrupted') for r in rows):
            if any(r['status'] in ('failed', 'interrupted') for r in rows):
                return 1
            if args.require_match and not all(r['reproduction_verified'] for r in rows):
                return 2
            return 0
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
