"""One supervised native Claude session on one official RoboDojo layout.

Use --detach to launch. Status, commands, full transcripts and official score
receipts remain in the run directory. An API failure is never scored as zero.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, build_opener

from lifecycle import digest, identity, now, read_json, write_json
from start import arguments, launch
from health import ConnectionHealth, workspace_status


class ExistingProcess:
    """A verified process retained during an operator supervisor handoff.

    Exit status is unavailable after reparenting. Native result receipts and
    official score files remain mandatory in serve(), including for exit 0.
    """
    def __init__(self, receipt):
        self.receipt = receipt
        self.pid = receipt['pid']
        self.returncode = None

    def poll(self):
        current = identity(self.pid)
        if not current or current['start_ticks'] != self.receipt['start_ticks']:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout):
        deadline = time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return self.returncode


def adopt_running(run, args):
    """Replace only a supervisor; preserve the agent, scene and native context."""
    state = read_json(run/'status.json')
    previous = state['supervisor']
    live = identity(previous['pid'])
    if not live or live['start_ticks'] != previous['start_ticks']:
        raise RuntimeError('Supervisor identity changed; refusing handoff')
    command = Path(f"/proc/{previous['pid']}/cmdline").read_bytes().split(b'\0')
    if str(Path(__file__).resolve()).encode() not in command or args.id.encode() not in command:
        raise RuntimeError('Supervisor command does not match this run')
    old_owner = read_json(run/'owner.json')
    os.kill(previous['pid'], signal.SIGSTOP)
    try:
        state = read_json(run/'status.json')
        if state['phase'] != 'running':
            raise RuntimeError('Handoff is only supported while the original agent is running')
        for key in ('task', 'gpu', 'port', 'seed', 'layout', 'model'):
            if state[key] != getattr(args, key):
                raise RuntimeError(f'Handoff changes experiment parameter {key}')
        children = [ExistingProcess(state[k]) for k in ('policy', 'evaluator', 'agent')]
        if any(child.poll() is not None for child in children):
            raise RuntimeError('A child has exited; refusing a live handoff')
        ack = read_json(run/'episode_ack.json')
        event = read_json(run/'episode.json')
        if event['state'] != 'ready' or ack['state'] != 'running' or ack['event_id'] != event['event_id']:
            raise RuntimeError('Episode is closing; refusing a live handoff')
        current = identity(os.getpid())
        write_json(run/'owner.json', current)
        write_json(run/'supervisor_handoff.json', {'at': now(), 'previous': previous,
                   'current': current, 'children': [p.receipt for p in children]})
    except BaseException:
        write_json(run/'owner.json', old_owner)
        os.kill(previous['pid'], signal.SIGCONT)
        raise
    # Children have their own process groups; kill this verified PID only.
    os.kill(previous['pid'], signal.SIGKILL)
    state.update(supervisor=current, supervisor_handoff_at=now())
    return state, children, ack


def model_health(base_url, model):
    client = build_opener(ProxyHandler({}))
    with client.open(base_url.rstrip('/')+'/v1/models', timeout=15) as response:
        data = json.load(response)
        route = response.headers.get('x-behavior-model-route', '')
    if model not in {item['id'] for item in data.get('data', [])}:
        raise RuntimeError(f'Requested model is not available: {model}')
    if model == 'Qwen3.8-Flash-Next-NVFP4' and not route.startswith('A-'):
        raise RuntimeError(f'Expected A route, received {route!r}')
    return {'model': model, 'route': route, 'checked_at': now()}


def stop_child(process, grace_seconds=20):
    if process is None:
        return
    # A shell may exit before Isaac Sim finishes shutting down. Track all
    # members of this explicitly owned process group, including orphan workers.
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            member = identity(int(path.name))
            if member and member['pgid'] == process.pid and member['sid'] == process.pid:
                members.append(member)
    if not members:
        process.poll()
        return
    original = getattr(process, 'receipt', None)
    leader = next((p for p in members if p['pid'] == process.pid), None)
    if leader and ((original and original['start_ticks'] != leader['start_ticks'])
                   or process.poll() is not None):
        raise RuntimeError('Child PID was reused; refusing to signal another process group')
    def surviving():
        return [p for p in members if (live := identity(p['pid']))
                and live['start_ticks'] == p['start_ticks']]
    for sig, seconds in [(signal.SIGTERM, grace_seconds), (signal.SIGKILL, 10)]:
        if not surviving():
            break
        os.killpg(process.pid, sig)
        deadline = time.monotonic()+seconds
        while surviving() and time.monotonic() < deadline:
            process.poll()
            time.sleep(.1)
    if surviving():
        raise RuntimeError('Owned child group did not exit after SIGKILL')
    process.poll()


def native_resume_id(workspace):
    paths = list((workspace/'.runtime/claude-home/projects').glob('*/*.jsonl'))
    if len(paths) != 1:
        raise RuntimeError('Cannot identify exactly one native session for safe resume')
    pending = set()
    for line in paths[0].open():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        content = event.get('message', {}).get('content', [])
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get('type') == 'tool_use':
                pending.add(block['id'])
            elif block.get('type') == 'tool_result':
                pending.discard(block.get('tool_use_id'))
    if pending:
        raise RuntimeError('Unresolved tool calls: automatic resume could replay an action')
    return paths[0].stem


class Stream:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.result = None
        self.model = None
        self.reply_at = None
        self.tools = 0
        self.thinking = 0

    def poll(self):
        if not self.path.is_file():
            return
        with self.path.open() as stream:
            stream.seek(self.offset)
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    self.offset = offset
                    break
                self.offset = stream.tell()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'result':
                    self.result = event
                if event.get('type') == 'assistant':
                    message = event.get('message', {})
                    self.model = message.get('model', self.model)
                    self.reply_at = now()
                    for block in message.get('content', []):
                        self.tools += block.get('type') == 'tool_use'
                        self.thinking += block.get('type') == 'thinking'


def start_agent(args, run, workspace, attempt=0, native_id=None):
    environment = dict(os.environ)
    for key in list(environment):
        if (key.startswith(('ANTHROPIC_', 'CLAUDE_', 'BEHAVIOR_', 'ROBODOJO_', 'QWEN_'))
                or key in {'DISABLE_COMPACT', 'DISABLE_AUTO_COMPACT', 'CLAUDECODE'}):
            environment.pop(key)
    environment.update(EMBODIED_ANTHROPIC_BASE_URL=args.base_url,
                       QWEN_MODEL=args.model, CLAUDE_CODE_AUTO_COMPACT_WINDOW='180000',
                       CLAUDE_CODE_MAX_CONTEXT_TOKENS='262144')
    command = [str(workspace/'.venv/bin/python'), '-m', 'tools.util.launch',
               '--model', args.model, '--base-url', args.base_url, '--',
               '-p', '--output-format', 'stream-json', '--verbose']
    if native_id:
        command.extend(['--resume', native_id])
    suffix = '' if attempt == 0 else f'.resume-{attempt}'
    output_path = run/f'agent{suffix}.stream.jsonl'
    with (run/'prompt.txt').open() as prompt, output_path.open('w') as output, \
            (run/f'agent{suffix}.stderr.log').open('w') as error:
        agent = subprocess.Popen(command, cwd=workspace, env=environment,
                                 stdin=subprocess.DEVNULL if native_id else prompt,
                                 stdout=output, stderr=error, start_new_session=True)
    write_json(run/f'agent{suffix}.json', {'process': identity(agent.pid),
               'command': command, 'started_at': now(), 'native_id': native_id})
    return agent, Stream(output_path)


def serve(args):
    os.umask(0o077)
    server = evaluator = agent = None
    run = Path(__file__).resolve().parent/'runs'/args.id
    interrupted = False
    event_id = None

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    state = {'phase': 'starting', 'created_at': now(), 'task': args.task,
             'gpu': args.gpu, 'port': args.port, 'seed': args.seed, 'layout': args.layout,
             'model': args.model, 'score': None}
    try:
        attempt = 0
        stream = None
        ack = None
        pending_native_id = None
        next_model_check = 0
        if args.adopt:
            state, (server, evaluator, agent), ack = adopt_running(run, args)
            workspace = Path(state['workspace'])
            event_id = ack['event_id']
            attempt = state.get('native_resumes', 0)
            suffix = '' if attempt == 0 else f'.resume-{attempt}'
            stream = Stream(run/f'agent{suffix}.stream.jsonl')
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(state['started_at'])).total_seconds()
            deadline = time.monotonic()+max(0, args.session_timeout-elapsed)
        else:
            record, server, evaluator = launch(args, benchmark=True)
            workspace = Path(record['workspace'])
            state.update(workspace=str(workspace), phase='loading',
                         supervisor=identity(os.getpid()), evaluator=identity(evaluator.pid),
                         policy=identity(server.pid))
            deadline = time.monotonic()+2400
        plan = read_json(run/'plan.json')
        write_json(run/'status.json', state)
        write_json(run/'summary.json', state)
        next_health = 0
        connection_health = ConnectionHealth()
        while True:
            if interrupted or (run/'stop_requested.json').exists():
                raise InterruptedError('Operator stopped this run; no fabricated score')
            event = read_json(run/'episode.json')
            if event.get('state') == 'ready' and event_id is None:
                if event['layout_id'] != args.layout or event['task'] != args.task:
                    raise RuntimeError('Evaluator episode does not match requested task/layout')
                event_id = event['event_id']
                ack = {'event_id': event_id, 'state': 'running', 'session_id': args.id}
                write_json(run/'episode_ack.json', ack)
                # Keep polling with zero physics steps while inference is
                # unavailable, retaining the live evaluator and its GPU.
                state.update(phase='waiting_for_model', started_at=now())
                deadline = time.monotonic()+args.session_timeout
            if state['phase'] == 'waiting_for_model' and time.monotonic() >= next_model_check:
                try:
                    state['model_health'] = model_health(args.base_url, args.model)
                except (OSError, ValueError, RuntimeError) as exc:
                    state['model_error'] = str(exc)
                    next_model_check = time.monotonic()+30
                else:
                    agent, stream = (start_agent(args, run, workspace, attempt, pending_native_id)
                                     if pending_native_id else start_agent(args, run, workspace))
                    state.update(phase='running', agent=identity(agent.pid))
                    state.pop('model_error', None)
            if stream:
                stream.poll()
            if event_id and event.get('event_id') != event_id:
                raise RuntimeError('Evaluator changed episode inside a private single-session workspace')
            if event_id and event.get('state') == 'ended' and state['phase'] != 'scoring':
                stop_child(agent)
                ack = {'event_id': event_id, 'state': 'closed'}
                write_json(run/'episode_ack.json', ack)
                state['phase'] = 'scoring'
                deadline = time.monotonic()+900
            elif agent is not None and agent.poll() is not None and ack['state'] == 'running':
                result = stream.result or {}
                if agent.returncode != 0 or not result or result.get('is_error'):
                    error = str(result.get('result', '')) or f'Native agent exited {agent.returncode}'
                    recoverable = any(word in error.lower() for word in
                                      ('timed out', 'timeout', 'overloaded', '503', '502', '529', 'connection'))
                    if not recoverable or attempt >= 3:
                        raise RuntimeError(error)
                    native_id = native_resume_id(workspace)
                    attempt += 1
                    pending_native_id = native_id
                    agent = None
                    next_model_check = time.monotonic()+5
                    state.update(phase='waiting_for_model', native_resumes=attempt)
                else:
                    ack['state'] = 'agent_done'
                    write_json(run/'episode_ack.json', ack)
                    state['phase'] = 'holding_for_official_score'
            if evaluator.poll() is not None:
                if evaluator.returncode != 0:
                    raise RuntimeError(f'Evaluator exited {evaluator.returncode}; see evaluator.log')
                result_path = Path(plan['chunks'][0]['result_path'])
                result = read_json(result_path)
                details = result.get('details', {})
                values = list(details.values()) if isinstance(details, dict) else details
                if (len(values) != 1 or int(values[0]['layout_id']) != args.layout
                        or int(result.get('eval_time', 0)) != 1):
                    raise RuntimeError('Official evaluator did not score the exact requested layout')
                write_json(run/'official_result.json', result)
                state.update(phase='complete', score=values[0]['score'], success=values[0]['success'],
                             result_path=str(result_path), result_sha256=digest(result_path))
                break
            if server.poll() is not None:
                raise RuntimeError('Workspace policy exited; see policy.log')
            if time.monotonic() > deadline:
                raise TimeoutError(f"Phase {state['phase']} exceeded its deadline; no fabricated score")
            if time.monotonic() >= next_health:
                snapshot = None
                try:
                    snapshot = workspace_status(workspace)
                    state['connection'] = snapshot
                    state.pop('connection_error', None)
                except (OSError, ValueError, RuntimeError) as exc:
                    state['connection_error'] = str(exc)
                fault = connection_health.check(snapshot, active=state['phase'] in
                    ('running', 'waiting_for_model', 'holding_for_official_score'))
                if fault:
                    raise RuntimeError(fault)
                calls = sorted((workspace/'tool_call_history').glob('*/result.json'))
                latest = read_json(calls[-1]) if calls else {}
                state.update(updated_at=now(), agent_alive=bool(agent and agent.poll() is None),
                             robot_tool_calls=len(calls), latest_tool=(latest.get('tool_name') or latest.get('tool') or
                                 (calls[-1].parent.name.split('_',1)[1] if calls else None)),
                             latest_tool_error=latest.get('error'))
                if stream:
                    state.update(response_model=stream.model, last_model_reply_at=stream.reply_at,
                                 native_tool_calls=stream.tools, thinking_blocks=stream.thinking)
                write_json(run/'status.json', state)
                write_json(run/'summary.json', state)
                next_health = time.monotonic()+10
            time.sleep(.5)
    except BaseException as exc:
        state.update(phase='stopped' if isinstance(exc, InterruptedError) else 'failed',
                     error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        if run.is_dir() and event_id:
            write_json(run/'episode_ack.json', {'event_id': event_id, 'state': 'error', 'error': state['error']})
    finally:
        # launch() can fail after binding the worker but before returning.
        # Retain process receipts early so this cannot orphan a policy socket.
        if not args.adopt:
            children = read_json(run/'children.json')
            if server is None and children.get('policy'):
                server = ExistingProcess(children['policy'])
            if evaluator is None and children.get('evaluator'):
                evaluator = ExistingProcess(children['evaluator'])
        for child in (agent, evaluator, server):
            stop_child(child)
        state.update(updated_at=now(), ended_at=now(), agent_alive=False)
        if run.is_dir() and (not args.adopt or server is not None):
            write_json(run/'status.json', state)
            write_json(run/'summary.json', state)
    return 0 if state['phase'] == 'complete' else 1


def main():
    parser = arguments()
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--layout', type=int, default=0)
    parser.add_argument('--model', default='Qwen3.8-Flash-Next-NVFP4')
    parser.add_argument('--base-url', default=os.environ.get('EMBODIED_ANTHROPIC_BASE_URL', 'http://127.0.0.1:31000'))
    parser.add_argument('--session-timeout', type=int, default=72*3600)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--adopt', action='store_true', help='Hand off a live supervisor without restarting its children')
    args = parser.parse_args()
    if not args.id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', args.id):
        parser.error('--id must contain 1..64 safe session characters')
    if args.detach:
        os.umask(0o077)
        logs = Path(__file__).resolve().parent/'runs/.launch'
        logs.mkdir(exist_ok=True, parents=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   *(item for item in sys.argv[1:] if item != '--detach')]
        suffix = f'.adopt-{time.time_ns()}' if args.adopt else ''
        with (logs/(args.id+suffix+'.log')).open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'supervisor': identity(child.pid), 'command': command,
                          'run_dir': str(logs.parent/args.id)}))
        return 0
    return serve(args)


if __name__ == '__main__':
    raise SystemExit(main())
