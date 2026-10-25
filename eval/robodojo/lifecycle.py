"""Operator-side receipts for the existing official RoboDojo episode loop.

No task physics, reward or observation implementation is replaced here.
Only evaluator transport can call benchmark_control; the agent's Unix API
continues to expose status and robot tools, never scoring/episode control.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        temporary = Path(f.name)
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'ppid': int(fields[1]), 'pgid': int(fields[2]), 'sid': int(fields[3])}
    except (OSError, ValueError, IndexError):
        return None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_plan(args, run):
    source = args.robodojo/'task/RoboDojo/tasks'/f'{args.task}.py'
    budgets = {node.value.value for node in ast.walk(ast.parse(source.read_text()))
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
               and any(isinstance(t, ast.Attribute) and t.attr == 'step_lim'
                       for t in node.targets)}
    if len(budgets) != 1:
        raise ValueError(f'Cannot resolve unchanged official action budget: {source}')
    budget = budgets.pop()
    prompt = args.prompt.resolve(strict=True)
    prompt_text = prompt.read_text()
    stated = re.search(r'Budget:\s*(\d+) action steps', prompt_text)
    if not stated or int(stated[1]) != budget:
        raise ValueError('Prompt and official evaluator action budgets disagree')
    layout = args.robodojo/f'Assets/Eval_Layout/RoboDojo/arx_x5/{args.seed}/{args.task}_{args.layout}.json'
    if not layout.is_file():
        raise ValueError(f'Missing official layout: {layout}')
    key = f's{args.seed}_{args.task}'
    result = (args.robodojo/'eval_result/RoboDojo'/args.task/'RoboDojoInterface/arx_x5'/
              f'{args.seed}_ckpt_name=workspace-v2,action_type=joint'/run.name/'_result.json')
    plan = {'run_id': run.name, 'run_dir': str(run), 'task': args.task,
            'gpu': args.gpu, 'port': args.port, 'seed': args.seed,
            'max_steps': budget, 'model': args.model, 'model_url': args.base_url,
            'prompt': str(prompt), 'prompt_sha256': digest(prompt),
            'task_source_sha256': digest(source), 'session_timeout_s': args.session_timeout,
            'created_at': now(), 'chunks': [{'key': key, 'task': args.task,
            'variant': 'random' if args.task.endswith('_random') else 'standard',
            'seed': args.seed, 'episodes': 1,
            'layout_ids': [args.layout], 'result_path': str(result),
            'layouts': [{'layout_id': args.layout, 'path': str(layout), 'sha256': digest(layout)}]}]}
    write_json(run/'plan.json', plan)
    (run/'prompt.txt').write_text(prompt_text)
    return plan


class BenchmarkControl:
    def benchmark_control(self, obs):
        """Existing evaluator lifecycle protocol; never expose reward to agent."""
        self.adapter.touch()
        command = obs.get('command')
        if command == 'identity':
            return {'episode_id': self.adapter.status()['episode_id'],
                    'instruction': self.adapter.instruction()}
        if command == 'hold':
            self.scheduler.reset()
            return self.world.make_joint_action()
        if command == 'close_episode':
            self.scheduler.reset()
            return {'ok': True}
        raise ValueError(f'Unsupported benchmark control: {command}')
