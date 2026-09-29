"""Launch the evaluator, observation interface, idle gate and selected agent."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / 'interface'
COMMIT = '26f2c7ef7b9cf96bd0414f81e1e751e493762779'
PROFILE = INTERFACE / 'behavior_interface_eval_test/robot_profiles/r1pro_8dof_hf250'
# Final integer limits from the archived Challenge 2025 plans. Do not derive
# these from the different 2026 human statistics or its 1.5 multiplier.
ARCHIVED_MAX_STEPS = {
    'task00': 4299, 'task01': 10535, 'task02': 27664, 'task03': 27392,
    'task05': 20343, 'task06': 15239, 'task07': 37781, 'task08': 17886,
    'task09': 27437,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_archive_protocol(task: dict) -> None:
    expected = ARCHIVED_MAX_STEPS.get(task.get('task'))
    if (expected is None or type(task.get('max_steps')) is not int
            or task.get('max_steps') != expected
            or task.get('challenge_year') != 2025 or task.get('budget_multiplier') != 2
            or task.get('protocol') != 'archived-v391-x2'
            or task.get('evaluator_commit') != COMMIT):
        raise ValueError(
            f'{task.get("task")}: archived reproduction requires Challenge 2025 ×2, '
            f'BEHAVIOR v3.9.1, and exactly {expected} max_steps.'
        )


def load_task(token: str) -> dict:
    match = re.fullmatch(r'(?:task)?(\d{1,2})', token)
    key = f'task{int(match[1]):02d}' if match else token
    candidates = list((ROOT / 'tasks').glob('*.json'))
    for path in candidates:
        task = read_json(path)
        if key in (task['task'], task['task_name']):
            validate_archive_protocol(task)
            for case in task['cases']:
                if sha((ROOT / case['prompt']).read_bytes()) != case['prompt_sha256']:
                    raise ValueError(f'Prompt checksum mismatch: {case["prompt"]}')
                if sha((ROOT / case['reference_result']).read_bytes()) != case['reference_sha256']:
                    raise ValueError(f'Reference checksum mismatch: {case["reference_result"]}')
            return task
    raise ValueError(f'No archived task {token!r}; use --list (task04 is not in the supplied archive).')


def select_cases(task: dict, instances: str | None) -> list[dict]:
    if instances is None:
        return task['cases']
    ids = [int(x) for x in re.split(r'[,\s]+', instances.strip()) if x]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Instance IDs must be nonempty and unique.')
    mapping = {c['instance_id']: c for c in task['cases']}
    if set(ids) - mapping.keys():
        raise ValueError('Use actual instance IDs from this archive: 301,304,306,308,310.')
    # Preserve the caller's order, and use the same order in the evaluator.
    return [mapping[i] for i in ids]


def read_config(path: Path | None) -> dict:
    defaults = {
        'interface_python': str(ROOT / '.venv-interface/bin/python'),
        'evaluator_python': str(ROOT / '.venv-evaluator/bin/python'),
        'agent_python': str(ROOT / 'harness/claude_code/.venv/bin/python'),
        'data_path': str(ROOT / 'data'),
        'model_url': os.environ.get('ROBOHARNESS_MODEL_URL', 'http://127.0.0.1:31000'),
        'model': 'Qwen3.8-Flash-Next-FP8',
        'claude_bin': shutil.which('claude') or 'claude',
        'codex_bin': shutil.which('codex') or 'codex',
        'startup_timeout_s': 2400,
        'session_timeout_s': 86400,
        'cache_dir': str(Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'roboharness'),
        'evaluator_dependencies': '',
        'interface_dependencies': '',
        'unmask_evaluator_cuda': False,
    }
    selected = path or ROOT / 'configs/local.json'
    if selected.exists():
        defaults.update(read_json(selected))
    elif path:
        raise FileNotFoundError(selected)
    for key in ('interface_python', 'evaluator_python', 'agent_python', 'data_path', 'cache_dir',
                'interface_dependencies', 'evaluator_dependencies'):
        if defaults[key]:
            value = Path(defaults[key]).expanduser()
            defaults[key] = str(value if value.is_absolute() else ROOT / value)
    return defaults


def official_result(path: Path, task: dict, iid: int) -> dict | None:
    try:
        result = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if (result.get('task') != task['task_name'] or result.get('instance_id') != iid
            or result.get('rollout_id') != 0):
        raise ValueError(f'Evaluator result identity mismatch: {path}')
    q = result.get('q_score', {}).get('final')
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError(f'Invalid official Q score: {path}')
    if not isinstance(result.get('steps'), int) or result['steps'] < 0:
        raise ValueError(f'Invalid evaluator step count: {path}')
    return result


def request_json(port: int, endpoint: str, payload=None, timeout=8):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}{endpoint}',
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def handoff_ready(iid: int, health: dict, status: dict, request: dict) -> bool:
    pending = (request.get('op') in ('finish', 'reset') and request.get('request_id')
               and request['request_id'] != status.get('applied_request_id'))
    return bool(not pending and health.get('evaluator_connected')
                and health.get('episode_initialization', {}).get('ready')
                and status.get('current_instance_id') == iid
                and status.get('state') not in ('resetting', 'error', 'queued', 'finish_queued')
                and health.get('action_source') in ('hold', 'idle_hold', None, ''))


def process_start(pid: int) -> str | None:
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def owned_pids(token: str) -> list[int]:
    marker = f'ROBOHARNESS_RUN_TOKEN={token}'.encode()
    pids = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if marker in (path / 'environ').read_bytes().split(b'\0'):
                pids.append(int(path.name))
        except OSError:
            pass
    return pids


def cleanup(token: str, grace: float = 10):
    identities = {pid: process_start(pid) for pid in owned_pids(token)}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, birth in identities.items():
            if birth and process_start(pid) == birth:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, sig)
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and any(process_start(p) == b for p, b in identities.items() if b):
                time.sleep(0.2)


class Run:
    def __init__(self, plan: dict, config: dict):
        self.plan, self.config = plan, config
        self.path = Path(plan['run_dir'])
        self.task = plan['task_config']
        self.port = plan['port']
        self.token = plan['run_id']
        self.children = {}
        self.env = {}
        self.results = []
        self.created = False

    def log(self, message):
        line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message
        print(line, flush=True)
        with (self.path / 'run.log').open('a') as stream:
            stream.write(line + '\n')

    def spawn(self, role, command, env, *, stdin=None, stdout=None, stderr=None):
        out = Path(stdout) if stdout else self.path / 'logs' / f'{role}.log'
        out.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            output = stack.enter_context(out.open('wb'))
            error = stack.enter_context(Path(stderr).open('wb')) if stderr else subprocess.STDOUT
            input_stream = stack.enter_context(Path(stdin).open('rb')) if stdin else subprocess.DEVNULL
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=input_stream,
                                    stdout=output, stderr=error, start_new_session=True)
        self.children[role] = proc
        atomic_json(self.path / 'processes.json', {k: {'pid': v.pid, 'start': process_start(v.pid)} for k, v in self.children.items()})
        self.log(f'{role} pid={proc.pid}')
        return proc

    def assert_alive(self, *roles):
        for role in roles:
            proc = self.children[role]
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f'{role} exited ({code}); see {self.path / "logs" / (role + ".log")}')

    def prepare(self):
        self.path.mkdir(parents=True, exist_ok=False)
        self.created = True
        (self.path / 'logs').mkdir()
        (self.path / 'operator').mkdir()
        atomic_json(self.path / 'plan.json', self.plan)
        atomic_json(self.path / 'runtime_config.json', {
            key: self.config[key] for key in ('interface_python', 'evaluator_python', 'agent_python',
                'data_path', 'unmask_evaluator_cuda', 'interface_dependencies', 'evaluator_dependencies')})
        # No inherited source checkout, plugin cache, simulator root or stale session.
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('BEHAVIOR_', 'OMNIGIBSON_', 'EMBODIED_', 'CLAUDE_', 'QWEN_', 'ROBOHARNESS_'))
            and k not in {'PYTHONPATH', 'PYTHONHOME', 'CUDA_VISIBLE_DEVICES'}}
        gpu = str(self.plan['gpu'])
        cache = Path(self.config['cache_dir']) / f'gpu{gpu}'
        temp = cache / 'r' / self.token.rsplit('-', 1)[-1]
        for sub in ('tmp', 'appdata', 'warp', 'cuda', 'inductor', 'runtime'):
            (temp / sub).mkdir(parents=True, exist_ok=True)
        env.update({
            'PYTHONPATH': os.pathsep.join([str(ROOT), str(INTERFACE)]),
            'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'ROBOHARNESS_RUN_TOKEN': self.token,
            'ROBOHARNESS_ROOT': str(ROOT), 'ROBOHARNESS_HTTP_PORT': str(self.port),
            'ROBOHARNESS_TASK_ID': str(self.task['task_index']), 'ROBOHARNESS_GPU': gpu,
            'ROBOHARNESS_MAX_STEPS': str(self.task['max_steps']),
            'ROBOHARNESS_PROTOCOL': self.task['protocol'],
            'CUDA_VISIBLE_DEVICES': gpu, 'BEHAVIOR_EVAL_TEST_PHYSICAL_GPU': gpu,
            'BEHAVIOR_INTERFACE_PHYSICAL_GPU': gpu, 'OMNIGIBSON_GPU_ID': '0',
            'BEHAVIOR_EVAL_TEST_PORT': str(self.port), 'PORT': str(self.port),
            'BEHAVIOR_ROBOT_CONFIG': str(INTERFACE / 'configs/r1pro_8dof_high_force.yaml'),
            'BEHAVIOR_AGENT_MAX_TICKS': str(self.task['max_steps']),
            'BEHAVIOR_EVAL_TEST_POLICY_PORT': str(self.port + 1000),
            'BEHAVIOR_EVAL_OPERATOR_DIR': str(self.path / 'operator'),
            'BEHAVIOR_EVAL_SESSION_GUARD': str(self.path / 'active_session.json'),
            'BEHAVIOR_AGENT_RUNS': str(self.path / 'recordings'),
            'BEHAVIOR_AGENT_MONITOR_ROOT': str(self.path / 'monitor'),
            'BEHAVIOR_INTERFACE_WORK_DIR': str(self.path / 'work'),
            'BEHAVIOR_INTERFACE_RUNTIME_TMP_ROOT': str(temp / 'tmp'),
            'BEHAVIOR_INTERFACE_AUTO_STORAGE_CLEANUP': '0',
            'BEHAVIOR_INTERFACE_START_MIN_FREE_GIB': '0',
            'BEHAVIOR_INTERFACE_TEXCACHE_MAX_GIB': '8',
            'BEHAVIOR_EVAL_TEST_ROBOT_PROFILE': self.task['robot_profile'],
            'BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE': self.task['annotator'],
            'BEHAVIOR_EVAL_TEST_KIT_GPU_MAP': 'auto',
            'BEHAVIOR_EVAL_TEST_TASKING_THREADS': '8',
            'BEHAVIOR_EVAL_TEST_OBSERVATION_MAX_AGE_S': '86400',
            'BEHAVIOR_EVAL_TEST_ENABLE_PRIVILEGED_GRASP_ORACLE': '0',
            'BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL': '0',
            'BEHAVIOR_SPATIAL_MAP': '0', 'INTERFACE_TOOL_VERSION': 'v2',
            'OFFICIAL_V2_LITE_OCCUPANCY_DEVICE': 'cuda:0',
            'OMNIGIBSON_HEADLESS': '1', 'OMNI_KIT_ACCEPT_EULA': 'YES',
            'TMPDIR': str(temp / 'tmp'), 'TMP': str(temp / 'tmp'), 'TEMP': str(temp / 'tmp'),
            'XDG_RUNTIME_DIR': str(temp / 'runtime'),
            'WARP_CACHE_PATH': str(cache / 'warp'), 'CUDA_CACHE_PATH': str(cache / 'cuda'),
            'TORCHINDUCTOR_CACHE_DIR': str(cache / 'inductor'),
            'TORCHINDUCTOR_COMPILE_THREADS': '2', 'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
            'NO_PROXY': 'localhost,127.0.0.1,::1', 'no_proxy': 'localhost,127.0.0.1,::1',
        })
        env.pop('DISPLAY', None)
        # The native cache remains private to this release, and only numerical
        # dependencies may be taken from a separately configured installation.
        py = self.config['interface_python']
        data = subprocess.check_output([py, str(PROFILE / 'install.py'), '--data-root', self.config['data_path']], env=env, text=True).strip().splitlines()[-1]
        env['OMNIGIBSON_DATA_PATH'] = data
        self.env = env
        atomic_json(self.path / 'active_session.json', {})
        (self.path / 'runtime.txt').write_text('BEHAVIOR=' + COMMIT + '\ndata=' + data + '\n')
        self.spawn('guardian', [sys.executable, '-m', 'roboharness.guardian', str(os.getpid()), process_start(os.getpid()), self.token],
                   {k: v for k, v in env.items() if k != 'ROBOHARNESS_RUN_TOKEN'})

    def start_stack(self):
        self.log(f'Starting {self.task["task"]} on GPU {self.plan["gpu"]}, HTTP {self.port}')
        py = self.config['interface_python']
        env = dict(self.env)
        if self.config['interface_dependencies']:
            env['PYTHONPATH'] += os.pathsep + self.config['interface_dependencies']
        self.spawn('interface', [py, '-m', 'behavior_interface_eval_test.official_policy_interface',
            '--task', self.task['task_name'], '--scene', self.task['scene'], '--robot-dof', '8',
            '--tool-version', 'official_v2', '--ui-tool-version', 'v2', '--policy-host', '127.0.0.1',
            '--policy-port', str(self.port + 1000), '--http-host', '127.0.0.1', '--http-port', str(self.port)], env)
        deadline = time.monotonic() + 180
        while True:
            self.assert_alive('interface')
            try:
                health = request_json(self.port, '/__official__/healthz')
                if health.get('session_isolation', {}).get('enabled'):
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Interface did not become healthy in 180 seconds.')
            time.sleep(2)
        self.spawn('gate', [py, '-m', 'official_idle_step_gate.proxy', '--listen-host', '127.0.0.1',
            '--listen-port', str(self.port + 2000), '--backend-policy-uri', f'ws://127.0.0.1:{self.port + 1000}',
            '--backend-http-url', f'http://127.0.0.1:{self.port}', '--fail-closed', '--max-idle-wait-s', '0'], env)
        deadline = time.monotonic() + 60
        while True:
            self.assert_alive('gate')
            try:
                status = request_json(self.port + 2000, '/status')
                if status.get('fail_closed') is True and status.get('max_idle_wait_s') == 0:
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Idle gate did not become ready with the required configuration.')
            time.sleep(1)
        evaluator_env = dict(self.env)
        evaluator_env['OMNIGIBSON_APPDATA_PATH'] = str(Path(self.env['TMPDIR']).parent / 'appdata')
        if self.config['unmask_evaluator_cuda']:
            evaluator_env.pop('CUDA_VISIBLE_DEVICES', None)
            evaluator_env['BEHAVIOR_EVAL_TEST_UNMASK_CUDA'] = '1'
            evaluator_env['OMNIGIBSON_GPU_ID'] = str(self.plan['gpu'])
        evaluator_env['PYTHONPATH'] = os.pathsep.join(filter(None, [
            str(ROOT / 'BEHAVIOR/OmniGibson'), str(ROOT / 'BEHAVIOR/bddl3'), str(ROOT / 'BEHAVIOR/joylo'),
            self.config['evaluator_dependencies'], self.env['PYTHONPATH'],
        ]))
        self.spawn('evaluator', [self.config['evaluator_python'], '-m', 'behavior_interface_eval_test.official_evaluator_entrypoint',
            '--task-name', self.task['task_name'], '--robot-config', str(PROFILE / 'evaluator_robot.yaml'),
            '--host', '127.0.0.1', '--port', str(self.port + 2000), '--mode', 'public_test',
            '--instance-indices', *[str(c['slot']) for c in self.plan['cases']], '--num-rollouts', '1',
            '--max-steps', str(self.task['max_steps']),
            '--env-wrapper', 'behavior_interface_eval_test.official_rgbd_wrapper.OfficialRGBDFullResWrapper',
            '--output-dir', str(self.path / 'output'),
            *(['--write-video'] if self.plan['write_video'] else [])], evaluator_env)

    def wait_ready(self, case):
        iid = case['instance_id']
        deadline = time.monotonic() + self.config['startup_timeout_s']
        last_log, stable = 0, 0
        while time.monotonic() < deadline:
            self.assert_alive('interface', 'gate', 'evaluator')
            health = {}
            try:
                health = request_json(self.port, '/__official__/healthz')
                status_path = self.path / 'operator' / f'behavior_eval_operator_status_p{self.port}.json'
                status = read_json(status_path) if status_path.exists() else {}
                request_path = self.path / 'operator' / f'behavior_eval_operator_request_p{self.port}.json'
                request = read_json(request_path) if request_path.exists() else {}
                ready = handoff_ready(iid, health, status, request)
                stable = stable + 1 if ready else 0
                if stable >= 2:
                    self.log(f'instance {iid} ready'); return
            except (OSError, ValueError):
                stable = 0
            if time.monotonic() - last_log > 60:
                self.log(f'Waiting for instance {iid}: connected={health.get("evaluator_connected")} initialization={health.get("episode_initialization", {}).get("state")}')
                last_log = time.monotonic()
            time.sleep(2)
        raise TimeoutError(f'Evaluator initialization timed out for {iid}.')

    def start_agent(self, case):
        iid, slot = case['instance_id'], case['slot']
        sid = f't{self.task["task_index"]:02d}p{self.port}i{slot}-{uuid.uuid4().hex[:16]}'
        case_dir = self.path / f'instance_{iid}'
        case_dir.mkdir()
        raw = (ROOT / case['prompt']).read_text()
        # Preserve the archived bytes; only the connection-port hint is rendered.
        rendered = re.sub(r'\b1506\d\b', str(self.port), raw)
        (case_dir / 'prompt.source.txt').write_text(raw)
        (case_dir / 'prompt.txt').write_text(rendered)
        atomic_json(case_dir / 'case.json', {**case, 'session_id': sid, 'rendered_prompt_sha256': sha(rendered.encode())})
        atomic_json(self.path / 'active_session.json', {
            'port': self.port, 'session_id': sid, 'owner_pid': os.getpid(),
            'owner_start': process_start(os.getpid()), 'run_id': self.token,
        })
        response = request_json(self.port, '/api/agent_monitor/session_begin', {'session_id': sid}, timeout=60)
        if response.get('ok') is not True or response.get('session_id') != sid:
            raise RuntimeError(f'Could not begin session: {response}')
        env = dict(self.env)
        env.update({
            'BEHAVIOR_SESSION_ID': sid, 'BEHAVIOR_PORT': str(self.port),
            'BEHAVIOR_BASE_URL': f'http://127.0.0.1:{self.port}', 'BEHAVIOR_ALLOW_REMOTE': '0',
            'BEHAVIOR_EVAL_OWNER_PORT': str(self.port), 'BEHAVIOR_EVAL_OWNER_RUN': self.token,
            'EMBODIED_ANTHROPIC_BASE_URL': self.config['model_url'], 'QWEN_MODEL': self.config['model'],
            'EMBODIED_CLAUDE_PYTHON': self.config['agent_python'],
            'EMBODIED_CODEX_PYTHON': self.config['agent_python'],
            'CLAUDE_BIN': self.config['claude_bin'], 'CODEX_BIN': self.config['codex_bin'],
            'CLAUDE_CONFIG_DIR': str(case_dir / 'claude-home'),
            'CLAUDE_PLUGIN_DATA': str(case_dir / 'plugin-data'),
            'BEHAVIOR_RECORD_ROOT': str(case_dir / 'trajectory'),
        })
        harness = ROOT / 'harness' / self.plan['harness']
        if self.plan['harness'] == 'claude_code':
            command = [str(harness / 'scripts/run'), '--port', str(self.port), '--qwen-model', self.config['model'],
                       '--', 'exec', '--output-format', 'json', '-']
        else:
            env['CODEX_HOME'] = str(case_dir / 'codex-home')
            env['ROBOHARNESS_CODEX_MODEL'] = self.config['model']
            env['ROBOHARNESS_CODEX_BASE_URL'] = self.config['model_url']
            command = [str(harness / 'scripts/run'), '--port', str(self.port), '--', 'exec', '--json', '-']
        proc = self.spawn('agent', command, env, stdin=case_dir / 'prompt.txt',
                          stdout=case_dir / 'agent.json', stderr=case_dir / 'agent.stderr.log')
        self.write_summary('running')
        self.log(f'instance {iid} session {sid} prompt={case["prompt"]}')
        return proc, case_dir

    def finish_episode(self, reason):
        sys.path.insert(0, str(INTERFACE))
        from behavior_interface_eval_test.operator_scene_control import write_finish_request
        old = os.environ.get('BEHAVIOR_EVAL_OPERATOR_DIR')
        os.environ['BEHAVIOR_EVAL_OPERATOR_DIR'] = str(self.path / 'operator')
        try:
            write_finish_request(self.port, reason=reason)
        finally:
            if old is None:
                os.environ.pop('BEHAVIOR_EVAL_OPERATOR_DIR', None)
            else:
                os.environ['BEHAVIOR_EVAL_OPERATOR_DIR'] = old

    def wait_score(self, case, agent, case_dir):
        iid = case['instance_id']
        path = self.path / 'output/json' / f'{self.task["task_name"]}_{iid}_0.json'
        deadline = time.monotonic() + self.config['session_timeout_s']
        finished_at, reason, last_log = None, None, 0
        while True:
            result = official_result(path, self.task, iid)
            if result is not None:
                break
            self.assert_alive('interface', 'gate', 'evaluator')
            code = agent.poll()
            if finished_at is None and (code is not None or time.monotonic() >= deadline):
                if code not in (None, 0):
                    raise RuntimeError(f'Agent failed ({code}); see {case_dir / "agent.stderr.log"}')
                if self.plan['harness'] == 'claude_code' and code == 0:
                    body = read_json(case_dir / 'agent.json')
                    if body.get('is_error') or body.get('type') != 'result':
                        raise RuntimeError('Agent returned an error or no final result; not a valid evaluation.')
                reason = 'model_done' if code == 0 else 'wall_timeout'
                self.finish_episode(reason)
                finished_at = time.monotonic()
                self.log(f'instance {iid}: {reason}; waiting for evaluator JSON')
            if finished_at is not None and time.monotonic() - finished_at > 600:
                raise TimeoutError('Evaluator did not write a result after finish request.')
            if time.monotonic() - last_log > 60:
                try:
                    monitor = request_json(self.port, '/api/agent_monitor')
                    self.log(f'instance {iid}: agent={"running" if code is None else code} ticks={monitor.get("session_ticks")}')
                except (OSError, ValueError):
                    self.log(f'instance {iid}: waiting for official score')
                last_log = time.monotonic()
            time.sleep(2)
        atomic_json(self.path / 'active_session.json', {})
        if agent.poll() is None:
            os.killpg(agent.pid, signal.SIGTERM)
            try:
                agent.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(agent.pid, signal.SIGKILL)
                agent.wait(timeout=5)
        row = {'instance_id': iid, 'q': result['q_score']['final'], 'reference_q': case['reference_q'],
               'delta_q': result['q_score']['final'] - case['reference_q'], 'steps': result['steps'],
               'success': result['success'], 'result': str(path.relative_to(self.path)),
               'result_sha256': sha(path.read_bytes()), 'finish_reason': reason or 'evaluator_end'}
        row['archive_reported_q'] = case.get('archive_reported_q', case['reference_q'])
        row['delta_archive_q'] = row['q'] - row['archive_reported_q']
        atomic_json(case_dir / 'comparison.json', row)
        self.results.append(row)
        self.write_summary('running')
        self.log(f'instance {iid}: Q={row["q"]:.6f}, reference={row["reference_q"]:.6f}, delta={row["delta_q"]:+.6f}')

    def write_summary(self, status, error=None):
        result = {'status': status, 'run_id': self.token, 'task': self.task['task'],
                  'harness': self.plan['harness'], 'model': self.config['model'], 'protocol': self.task['protocol'],
                  'n_expected': len(self.plan['cases']), 'n_finished': len(self.results), 'cases': self.results,
                  'mean_q': sum(x['q'] for x in self.results) / len(self.results) if self.results else None,
                  'reference_mean_q': sum(x['reference_q'] for x in self.plan['cases']) / len(self.plan['cases'])}
        result['archive_reported_mean_q'] = sum(
            x.get('archive_reported_q', x['reference_q']) for x in self.plan['cases']) / len(self.plan['cases'])
        result['delta_archive_mean_q'] = (result['mean_q'] - result['archive_reported_mean_q']
                                         if len(self.results) == len(self.plan['cases']) else None)
        if error:
            result['error'] = str(error)
        atomic_json(self.path / 'summary.json', result)

    def execute(self):
        try:
            self.prepare()
            self.write_summary('starting')
            self.start_stack()
            for case in self.plan['cases']:
                self.wait_ready(case)
                agent, case_dir = self.start_agent(case)
                self.wait_score(case, agent, case_dir)
            self.write_summary('complete')
        except BaseException as exc:
            if self.created:
                self.write_summary('failed', exc)
            raise
        finally:
            cleanup(self.token)
            for proc in self.children.values():
                with contextlib.suppress(subprocess.TimeoutExpired):
                    if proc is self.children.get('guardian'):
                        proc.terminate()
                    proc.wait(timeout=3)


def preflight(config, harness):
    for key in ('interface_python', 'evaluator_python', 'agent_python'):
        if not Path(config[key]).is_file():
            raise ValueError(f'{key} does not exist: {config[key]}; run scripts/setup.sh or configure configs/local.json.')
    code = subprocess.check_output(['git', '-C', str(ROOT / 'BEHAVIOR'), 'rev-parse', 'HEAD'], text=True).strip()
    if code != COMMIT:
        raise ValueError(f'BEHAVIOR must be the archived v3.9.1 commit {COMMIT}; found {code}')
    for entry in ('behavior-1k-assets', 'omnigibson-robot-assets', '2026-challenge-task-instances', 'omnigibson.key'):
        if not (Path(config['data_path']) / entry).exists():
            raise ValueError(f'Missing data: {entry}; see docs/setup.md. The dataset and key are external to this repository.')
    binary = config['claude_bin' if harness == 'claude_code' else 'codex_bin']
    if not shutil.which(binary):
        raise ValueError(f'Agent CLI is not executable: {binary}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='task00', help='task00, 0, or task name')
    ap.add_argument('--instances', help='Actual IDs, e.g. 301,304 (default: all five archived cases)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--port', type=int, help='HTTP port (default: 16060 + task index); reserves port+1000 and port+2000')
    ap.add_argument('--harness', choices=['claude_code', 'codex'], default='claude_code')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--model-url')
    ap.add_argument('--model')
    ap.add_argument('--run-dir', type=Path)
    ap.add_argument('--write-video', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args(argv)
    try:
        if args.list:
            for path in sorted((ROOT / 'tasks').glob('*.json')):
                task = load_task(path.stem)
                print(f'{task["task"]}  {task["task_name"]:42s} archive Q={task["archive_reported_mean_q"]:.6f}  JSON Q={task["reference_mean_q"]:.6f}')
            return 0
        task = load_task(args.task)
        cases = select_cases(task, args.instances)
        config = read_config(args.config)
        for key in ('model_url', 'model'):
            if getattr(args, key):
                config[key] = getattr(args, key)
        port = args.port or 16060 + task['task_index']
        if not 1024 <= port <= 63535 or args.gpu < 0:
            raise ValueError('Use GPU >= 0 and an unprivileged port <= 63535.')
        run_id = task['task'] + '-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        path = args.run_dir.resolve() if args.run_dir else ROOT / 'runs' / run_id
        if path.exists():
            raise ValueError(f'Run directory must be new: {path}')
        plan = {'run_id': run_id, 'run_dir': str(path), 'port': port, 'gpu': args.gpu,
                'harness': args.harness, 'model': config['model'], 'model_url': config['model_url'],
                'write_video': args.write_video, 'task_config': task, 'cases': cases}
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False)); return 0
        preflight(config, args.harness)
        from .assets import prepare_robot_asset
        prepare_robot_asset()
        lock_root = Path(config['cache_dir']) / 'locks'
        lock_root.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            for selected in sorted((port, port + 1000, port + 2000)):
                handle = stack.enter_context((lock_root / f'port-{selected}.lock').open('a'))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(f'Another RoboHarness run owns port {selected}.')
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', selected))
            def stop(signum, frame):
                raise KeyboardInterrupt(f'Received signal {signum}')
            signal.signal(signal.SIGTERM, stop)
            Run(plan, config).execute()
        print(f'Results: {path / "summary.json"}')
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'RoboHarness: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
