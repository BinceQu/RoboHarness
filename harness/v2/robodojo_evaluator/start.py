"""Start one independent ARX_X5 workspace and the unmodified RoboDojo evaluator."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from assets import prepare, prepare_joint_config

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robodojo', type=Path, default=os.environ.get('ROBODOJO_ROOT'), required='ROBODOJO_ROOT' not in os.environ,
                        help='RoboDojo checkout (default: $ROBODOJO_ROOT)')
    parser.add_argument('--evaluator-env', type=Path, default=os.environ.get('ROBODOJO_EVALUATOR_ENV'),
                        required='ROBODOJO_EVALUATOR_ENV' not in os.environ,
                        help='Python environment of the RoboDojo evaluator (default: $ROBODOJO_EVALUATOR_ENV)')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--task', default='general_pickup')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--id')
    parser.add_argument('--without-evaluator-planner', action='store_true',
                        help='Use native need_planner=False for joint-only evaluation')
    return parser

def launch(args, *, benchmark=False):
    os.umask(0o077)
    here = Path(__file__).resolve().parent
    template = here.parent/'ARX_X5'
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    name = args.id or 'x5-'+stamp
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', args.port))
    spec = importlib.util.spec_from_file_location('arx_session', template/'session.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = here/'runs'/name
    workspace = module.create(run/'ARX_X5', name, args.task)
    if benchmark:
        from lifecycle import prepare_plan, identity, write_json
        prepare_plan(args, run)
        write_json(run/'owner.json', identity(os.getpid()))
    assets = prepare(args.robodojo, run/'evaluator_assets')
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith('ROBODOJO_HARNESS_') or key.startswith('ROBODOJO_INTERFACE_'):
            environment.pop(key)
    for key in ['PYTHONPATH','PYTHONHOME','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        environment.pop(key, None)
    environment.update(ROBODOJO_RUN_ID=name, EVAL_NUM='1', ROBODOJO_MAX_BASH_RETRIES='0', PYTHONDONTWRITEBYTECODE='1',
                       ROBODOJO_USD_ASSET_PREFIX=environment.get('ROBODOJO_USD_ASSET_PREFIX',''))
    environment.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    environment['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    server_args = [str(workspace/'.venv/bin/python'), str(here/'server.py'), '--workspace', str(workspace),
                   '--robodojo', str(args.robodojo), '--port', str(args.port)]
    with (run/'policy.log').open('w') as output:
        policy_environment = {**environment, 'MALLOC_ARENA_MAX': '2'}
        server = subprocess.Popen(server_args, stdout=output, stderr=subprocess.STDOUT, env=policy_environment,
                                  start_new_session=True, cwd=workspace)
    from lifecycle import identity, write_json
    children = {'policy': identity(server.pid)}
    write_json(run/'children.json', children)
    # Cold imports on NFS can exceed ten seconds after a host reboot.
    # This deadline covers service readiness only, before evaluation starts.
    policy_deadline = time.monotonic() + 300
    while time.monotonic() < policy_deadline:
        if server.poll() is not None:
            raise RuntimeError(f'Policy failed: see {run}/policy.log')
        try:
            with socket.create_connection(('127.0.0.1',args.port), timeout=.3):
                break
        except OSError:
            time.sleep(.1)
    else:
        server.terminate()
        raise RuntimeError('Policy did not bind its assigned port within 300 seconds')
    environment['PATH'] = str(args.evaluator_env/'bin')+os.pathsep+environment.get('PATH','')
    # The upstream CuRobo YAML contains an installation-specific absolute path.
    # Relocate paths in a per-run overlay; do not change assets or task physics.
    environment['ROBOHARNESS_ROBOTS_PATH'] = str(assets)
    if args.without_evaluator_planner:
        config_root = prepare_joint_config(args.robodojo, run/'evaluator_config')
        environment['ROBOHARNESS_ENV_CONFIG_PATH'] = str(config_root)
        if benchmark:
            from lifecycle import digest, read_json
            plan = read_json(run/'plan.json')
            plan['evaluator_config_override'] = {
                'key': 'robot.robots[*].need_planner', 'value': False,
                'reason': 'workspace sends joint actions; evaluator IK is unused',
                'source_sha256': digest(args.robodojo/'env_cfg/robot/dual_x5.yml'),
                'effective_path': str(config_root/'robot/dual_x5.yml'),
                'effective_sha256': digest(config_root/'robot/dual_x5.yml')}
            write_json(run/'plan.json', plan)
    else:
        environment.pop('ROBOHARNESS_ENV_CONFIG_PATH', None)
    environment['PYTHONPATH'] = os.pathsep.join([str(here/'path_shim'), str(args.robodojo.resolve())])
    if benchmark:
        environment.update(ROBODOJO_HARNESS_RUN_DIR=str(run),
                           ROBODOJO_HARNESS_CHUNK=f's{args.seed}_{args.task}')
    evaluator_args = ['bash',str(args.robodojo/'scripts/robodojo.sh'),'client',
        '--policy-name','RoboDojoInterface','--policy-host','127.0.0.1','--policy-port',str(args.port),
        '--env-gpu',str(args.gpu),'--task',args.task,'--env-cfg','robodojo_interface_arx_x5',
        '--seed',str(args.seed),'--ckpt','workspace-v2','--action-type','joint','--eval-num','1']
    with (run/'evaluator.log').open('w') as output:
        evaluator = subprocess.Popen(evaluator_args, cwd=args.robodojo, stdout=output, stderr=subprocess.STDOUT,
                                     env=environment, start_new_session=True)
    children['evaluator'] = identity(evaluator.pid)
    write_json(run/'children.json', children)
    record = {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'workspace':str(workspace),
              'gpu':args.gpu,'task':args.task,'seed':args.seed,'port':args.port,
              'policy_pid':server.pid,'evaluator_pid':evaluator.pid,
              'without_evaluator_planner': args.without_evaluator_planner,
              'policy_command':server_args,'evaluator_command':evaluator_args,
              'kind':'official evaluator, selected standard episode' if benchmark else
                     'development acceptance; no score until official evaluator completes'}
    (run/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    return record, server, evaluator

def main():
    record, _, _ = launch(arguments().parse_args())
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
