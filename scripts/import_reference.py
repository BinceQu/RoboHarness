#!/usr/bin/env python3
"""Curate the selected archived cases; never infer prompts from version names alone."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

INSTANCES = (301, 304, 306, 308, 310)
TASKS = {
    0: ('turning_on_radio', 4299),
    1: ('picking_up_trash', 10535),
    2: ('putting_away_Halloween_decorations', 27664),
    3: ('cleaning_up_plates_and_food', 27392),
    5: ('setting_mousetraps', 20343),
    6: ('hiding_Easter_eggs', 15239),
    7: ('picking_up_toys', 37781),
    8: ('rearranging_kitchen_furniture', 17886),
    9: ('putting_up_Christmas_decorations_inside', 27437),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def first_user_prompt(path: Path) -> str:
    if path.name == 'session.json':
        return load(path)['user_prompt']
    if path.suffix == '.txt':
        return path.read_text()
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('type') != 'user':
                continue
            content = item.get('message', {}).get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [x['text'] for x in content if x.get('type') == 'text']
                if texts:
                    return '\n'.join(texts)
    raise ValueError(f'No user prompt in {path}')


def one(paths) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one source, found {paths}')
    return paths[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--behavior-source', type=Path, required=True)
    ap.add_argument('--prompt-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--radio-reference', choices=['5test', 'manifest'], default='5test')
    args = ap.parse_args()
    source, output = args.behavior_source.resolve(), args.output.resolve()
    root = source / 'test_results'
    context_index = load(Path(__file__).resolve().parents[1] /
                         'harness/claude_code/tests/fixtures/archive_context.json')
    contexts = {(s['task'], s['instance_id']): s for s in context_index['sources']}
    summary = []
    for index, (name, budget) in TASKS.items():
        key = f'task{index:02d}'
        archive = root / key
        prompt_candidates = list(args.prompt_source.glob(f'{index:02d}_*/*.txt'))
        prompt_candidates += list(archive.glob('prompts/*.txt')) + list(archive.glob('03_prompts/*.txt'))
        cases = []
        for iid in INSTANCES:
            notes = []
            slot = iid - 301
            stem = f'{name}_{iid}_0.json'
            if index == 0:
                run = 't00-15060-09152118' if iid <= 304 else 't00-15060-09171200'
                sub = 'json.quarantine-5test-09172314'
                clocks = {301: '09161944', 304: '09162209', 306: '09171213', 308: '09171540', 310: '09171640'}
                if args.radio_reference == 'manifest' and iid <= 304:
                    sub = 'json.quarantine-v21-09161615'
                    clocks = {**clocks, 301: '09131706', 304: '09131732'}
                    notes.append('The top-level manifest selects an older quarantined result; its session association is not conclusive.')
                score = archive / 'eval_runs' / run / 'p15060/output' / sub / stem
                prompt = archive / 'agent_sessions' / f't00p15060i{slot}-{clocks[iid]}' / 'session.json'
                notes.append('Archive manifest reports mean 0.4; the coherent 5test result set reports 0.6. See docs/provenance.md.')
            elif index == 1:
                score = archive / 'scoreboard' / f'{iid}_v4.json'
                prompt = one((archive / 'monitor' / str(iid)).glob('2_*/session.json'))
            elif index == 2:
                score = archive / f'instance_{iid}/evaluator_result.json'
                prompt = archive / f'instance_{iid}/session.json'
                if not prompt.exists():
                    prompt = archive / f'instance_{iid}/prompt.txt'
                if iid == 310:
                    notes.append('The top-level SUMMARY is stale; the selected case is the completed v18 rerun (q=1), not the v17 backup (q=0).')
            elif index in (3, 6):
                score = archive / '01_official_json' / stem
                suffix = f'_{iid}' + ('_v9' if index == 6 and iid >= 308 else '') + '.jsonl'
                prompt = one((archive / '06_claude_transcripts').glob('*' + suffix))
                if index == 6 and iid == 301:
                    notes.append('The actual transcript prompt differs from the archived v8 filename snapshot; the transcript text is preserved verbatim.')
            elif index == 5:
                score = archive / 'output_json' / stem
                clock = {301: '09152148', 304: '09160125', 306: '09161131', 308: '09171210', 310: '09181018'}[iid]
                prompt = archive / 'session_cards' / f'i{iid}_t05p15065i{slot}-{clock}' / 'session.json'
                if iid == 308:
                    notes.append('Session prompt matches v8; summary.json incorrectly calls this case v7.')
            elif index == 7:
                session = next(s for s in load(archive / 'manifest.json')['sessions'] if s['instance_id'] == iid)
                score = archive / session['official_result']
                prompt = archive / session['agent_monitor'] / 'session.json'
            elif index == 8:
                score = archive / f'instance_{iid}' / stem
                session = load(archive / f'instance_{iid}/session_end_{iid}.json')['session_id']
                prompt = source / 'work/agent_monitor/t08_rearranging_kitchen_furniture' / session / 'session.json'
                notes.append('The archived session_end JSON identifies the monitor session; the matching Claude transcript preserves the exact prompt bytes.')
            else:
                score = archive / '_meta/official_score_json' / stem
                prompt = one((archive / f'instance_{iid}').glob('*/session.json'))
                if iid <= 306:
                    notes.append('Actual session prompt matches v3; INDEX.md calls this batch v2. Session text takes precedence.')
            raw_score = load(score)
            # The scoreboard is an original evaluator JSON, not a derived mean.
            assert raw_score['task'] == name and int(raw_score['instance_id']) == iid, score
            archived_prompt_record = prompt
            context = contexts[(key, iid)]
            prompt = source / context['path']
            if sha(prompt.read_bytes()) != context['sha256']:
                raise ValueError(f'Archived transcript checksum mismatch: {prompt}')
            text = first_user_prompt(prompt)
            if text.strip() != first_user_prompt(archived_prompt_record).strip():
                raise ValueError(f'Transcript prompt disagrees with the selected archive: {prompt}')
            digest = sha(text.encode())
            assert digest == context['prompt_sha256'], prompt
            matches = sorted({p.name for p in prompt_candidates if p.read_text().strip() == text.strip()})
            versions = [m for m in matches if re.search(r'_v\d+\.txt$', m) and '_wins_' not in m]
            label = re.search(r'_v(\d+)\.txt$', versions[0])[0][1:-4] if versions else 'recovered'
            dest_prompt = output / 'prompt' / key / f'{label}_{digest[:12]}.txt'
            dest_prompt.parent.mkdir(parents=True, exist_ok=True)
            dest_prompt.write_text(text)
            dest_score = output / 'reference_results' / key / stem
            dest_score.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(score, dest_score)
            if raw_score['steps'] > budget + 1:
                notes.append(f'Historical evaluator reported {raw_score["steps"]} steps, exceeding the planned {budget}-step budget. New runs keep the declared budget; this overrun is not reproduced by silently increasing it.')
            case = {
                'instance_id': iid, 'slot': slot,
                'prompt': dest_prompt.relative_to(output).as_posix(),
                'prompt_sha256': digest,
                'prompt_source': prompt.relative_to(source).as_posix(),
                'prompt_source_sha256': context['sha256'],
                'archive_prompt_record': archived_prompt_record.relative_to(source).as_posix(),
                'archive_prompt_record_sha256': sha(archived_prompt_record.read_bytes()),
                'claude_mcp_name': context['mcp_server_name'],
                'matching_prompt_filenames': versions,
                'reference_result': dest_score.relative_to(output).as_posix(),
                'reference_sha256': sha(score.read_bytes()),
                'reference_source': score.relative_to(source).as_posix(),
                'reference_q': raw_score['q_score']['final'],
                'reference_steps': raw_score['steps'],
                'notes': notes,
            }
            if index == 6 and iid == 301:
                case['reproduction_caveats'] = [
                    'The archived Claude session starts after earlier navigation and arm '
                    'operations in an already running scene. Their request parameters and '
                    'a complete scene snapshot were not found in the supplied archive. '
                    'A fresh reset does not reproduce that starting state; see '
                    'reference_results/initial_states.json and docs/provenance.md.'
                ]
            if context.get('source_session_id'):
                case['source_session_id'] = context['source_session_id']
            elif archived_prompt_record.name == 'session.json':
                case['source_session_id'] = load(archived_prompt_record).get('session_id')
            elif index == 8:
                case['source_session_id'] = session
            cases.append(case)
        config = {
            'schema_version': 1, 'task': key, 'task_index': index, 'task_name': name,
            'scene': 'house_single_floor' if index in (7, 9) else 'house_double_floor_lower',
            'protocol': 'archived-v391-x2', 'evaluator_commit': '26f2c7ef7b9cf96bd0414f81e1e751e493762779',
            'challenge_year': 2025, 'budget_multiplier': 2,
            'max_steps': budget, 'sample_seed': 20260911, 'evaluator_seed': 0,
            'model': 'Qwen3.8-Flash-Next-FP8', 'harness': 'claude_code',
            'robot_profile': 'r1pro_8dof_hf250', 'annotator': 'cpu',
            'idle_gate': True, 'spatial_map': False, 'cases': cases,
            'reference_mean_q': sum(c['reference_q'] for c in cases) / len(cases),
        }
        # Keep the directory author's reported result separate from the raw
        # evaluator JSON. Neither source is rewritten to make them agree.
        if index in (0, 5):
            report = archive / ('manifest.json' if index == 0 else 'summary.json')
            body = load(report)
            reported = body['final_q_by_instance'] if index == 0 else {
                iid: row['q_score_final'] for iid, row in body['instances'].items()}
            config['archive_reported_mean_q'] = body['mean_q']
            config['archive_report_source'] = report.relative_to(source).as_posix()
            config['archive_report_sha256'] = sha(report.read_bytes())
            for case in cases:
                case['archive_reported_q'] = reported[str(case['instance_id'])]
        else:
            config['archive_reported_mean_q'] = config['reference_mean_q']
        (output / 'tasks').mkdir(exist_ok=True)
        (output / 'tasks' / (key + '.json')).write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        summary.append({k: config[k] for k in ['task', 'task_name', 'archive_reported_mean_q', 'reference_mean_q']})
        print(key, f'mean={config["reference_mean_q"]:.6f}', [(c['instance_id'], Path(c['prompt']).name) for c in cases])
    (output / 'reference_results/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    used = {c['prompt'] for path in (output / 'tasks').glob('*.json') for c in load(path)['cases']}
    for path in (output / 'prompt').glob('*/*.txt'):
        if path.relative_to(output).as_posix() not in used:
            path.unlink()


if __name__ == '__main__':
    main()
