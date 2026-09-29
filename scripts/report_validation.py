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


def collect(path: Path) -> dict:
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
                          'result_sha256': case['result_sha256'],
                          'source_result': case['result']})
    status = summary['status']
    if status == 'starting' and any(path.glob('instance_*/case.json')):
        status = 'running'
    if status == 'complete' and len(completed) != len(expected):
        raise ValueError(f'Run claims completion without all official scores: {path}')
    archive_mean = sum(c.get('archive_reported_q', c['reference_q']) for c in expected.values()) / len(expected)
    mean = sum(c['q'] for c in completed) / len(completed) if completed else None
    complete = status == 'complete'
    return {'task': task['task'], 'task_name': task['task_name'], 'run_directory': path.name,
            'status': status, 'model': plan['model'], 'harness': plan['harness'],
            'n_finished': len(completed), 'n_expected': len(expected), 'cases': completed,
            'mean_q': mean, 'archive_mean_q': archive_mean,
            'delta_archive_mean_q': mean - archive_mean if complete else None,
            'matches_archive_mean': abs(mean - archive_mean) < 1e-6 if complete else None,
            'error': summary.get('error')}


def write_report(output: Path, rows: list[dict], run_paths: list[Path]):
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    atomic_json(output / 'report.json', {'updated_at_utc': timestamp, 'runs': rows})
    lines = ['# GPU validation results', '', f'Updated: {timestamp}', '',
             'Only fresh official evaluator JSON contributes to new scores. An incomplete',
             'run has no final mean comparison and is not a successful reproduction claim.', '',
             'Comparisons cover only the selected instances. Run all five archived instances',
             'to compare a complete task mean.', '',
             '| Task | Status | Completed | New Q (completed cases) | Archived selected-case Q | Final difference |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row, run in zip(rows, run_paths):
        mean = '—' if row['mean_q'] is None else f'{row["mean_q"]:.6f}'
        delta = '—' if row['delta_archive_mean_q'] is None else f'{row["delta_archive_mean_q"]:+.6f}'
        lines.append(f'| {row["task"]} | {row["status"]} | {row["n_finished"]}/{row["n_expected"]} | '
                     f'{mean} | {row["archive_mean_q"]:.6f} | {delta} |')
        for case in row['cases']:
            source = run / case['source_result']
            target = output / row['task'] / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
    lines += ['', 'Case scores, hashes and failure details are in [report.json](report.json).', '']
    temporary = output / 'README.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(output / 'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'validation_results/latest')
    parser.add_argument('--watch', action='store_true', help='Update until every selected run completes or fails')
    args = parser.parse_args()
    previous = None
    while True:
        rows = [collect(path.resolve()) for path in args.runs]
        if len({r['task'] for r in rows}) != len(rows):
            raise ValueError('Choose one run per task; separate attempts must be reported separately.')
        signature = json.dumps(rows, sort_keys=True)
        if signature != previous:
            write_report(args.output.resolve(), rows, [p.resolve() for p in args.runs])
            print(' | '.join(f'{r["task"]}: {r["status"]}, {r["n_finished"]}/{r["n_expected"]}' for r in rows), flush=True)
            previous = signature
        if not args.watch or all(r['status'] in ('complete', 'failed') for r in rows):
            return 1 if any(r['status'] == 'failed' for r in rows) else 0
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
