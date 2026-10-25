import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifecycle import BenchmarkControl, identity, prepare_plan, read_json, write_json
from run import Stream, native_resume_id


@pytest.fixture
def planned(tmp_path):
    source = tmp_path/'source'
    task = source/'task/RoboDojo/tasks/stack_bowls.py'
    task.parent.mkdir(parents=True)
    task.write_text('class Task:\n    def init(self):\n        self.step_lim = 800\n')
    layout = source/'Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_0.json'
    layout.parent.mkdir(parents=True)
    layout.write_text('{"scene":"unchanged"}')
    prompt = tmp_path/'prompt.txt'
    prompt.write_text('Instruction: Stack bowls.\nBudget: 800 action steps.\n')
    run = tmp_path/'test-run'
    run.mkdir()
    args = SimpleNamespace(robodojo=source, task='stack_bowls', prompt=prompt,
                           gpu=1, port=19090, seed=0, layout=0, model='NVFP4',
                           base_url='http://localhost:31000', session_timeout=259200)
    return args, run


def test_plan_preserves_source_prompt_budget_and_layout(planned):
    args, run = planned
    plan = prepare_plan(args, run)
    assert plan['max_steps'] == 800
    assert plan['chunks'][0]['layout_ids'] == [0]
    assert (run/'prompt.txt').read_bytes() == args.prompt.read_bytes()
    assert plan['chunks'][0]['result_path'].endswith('/test-run/_result.json')
    assert read_json(run/'plan.json') == plan


def test_reject_wrong_budget(planned):
    args, run = planned
    args.prompt.write_text('Budget: 1600 action steps.')
    with pytest.raises(ValueError, match='budgets disagree'):
        prepare_plan(args, run)


def test_reject_missing_layout(planned):
    args, run = planned
    args.layout = 99
    with pytest.raises(ValueError, match='Missing official layout'):
        prepare_plan(args, run)


def test_resume_requires_completed_tool_results(tmp_path):
    path = tmp_path/'.runtime/claude-home/projects/project/session.jsonl'
    path.parent.mkdir(parents=True)
    use = {'message':{'content':[{'type':'tool_use','id':'one'}]}}
    result = {'message':{'content':[{'type':'tool_result','tool_use_id':'one'}]}}
    path.write_text(json.dumps(use)+'\n')
    with pytest.raises(RuntimeError, match='Unresolved'):
        native_resume_id(tmp_path)
    path.write_text(json.dumps(use)+'\n'+json.dumps(result)+'\n')
    assert native_resume_id(tmp_path) == 'session'


def test_stream_waits_for_complete_line_and_retains_real_model(tmp_path):
    path = tmp_path/'stream.jsonl'
    stream = Stream(path)
    event = {'type':'assistant','message':{'model':'actual-NVFP4','content':[
        {'type':'thinking','thinking':'test'}, {'type':'tool_use','id':'one'}]}}
    path.write_text(json.dumps(event))
    stream.poll()
    assert stream.model is None
    path.write_text(json.dumps(event)+'\n')
    stream.poll()
    stream.poll()
    assert (stream.model, stream.tools, stream.thinking) == ('actual-NVFP4',1,1)


def test_benchmark_transport_never_exposes_reward():
    events=[]
    bridge = BenchmarkControl()
    bridge.adapter = SimpleNamespace(touch=lambda:events.append('touch'),
        status=lambda:{'episode_id':'episode','reward':'hidden'}, instruction=lambda:'stack bowls')
    action={'left_arm_joint_state':[0]*6,'right_arm_joint_state':[0]*6,
            'left_ee_joint_state':[.4],'right_ee_joint_state':[.6]}
    bridge.world=SimpleNamespace(make_joint_action=lambda:action)
    bridge.scheduler=SimpleNamespace(reset=lambda:events.append('closed'))
    assert bridge.benchmark_control({'command':'identity'}) == {
        'episode_id':'episode','instruction':'stack bowls'}
    assert bridge.benchmark_control({'command':'hold'}) == action
    assert bridge.benchmark_control({'command':'close_episode'}) == {'ok':True}
    assert events[-1]=='closed'
    with pytest.raises(ValueError):
        bridge.benchmark_control({'command':'reward'})


def test_existing_official_loop_delivers_terminal_observation(tmp_path, monkeypatch):
    """Exercise the real evaluator lifecycle without simulator or model calls."""
    if 'ROBODOJO_ROOT' not in os.environ:
        pytest.skip('set ROBODOJO_ROOT to a RoboDojo checkout')
    repo = Path(os.environ['ROBODOJO_ROOT'])
    monkeypatch.syspath_prepend(str(repo))
    import robodojo_eval_harness.episode as episode
    monkeypatch.setenv('ROBODOJO_HARNESS_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('ROBODOJO_HARNESS_SELECTION_ACTIVE', '1')
    monkeypatch.setenv('ROBODOJO_HARNESS_CHUNK', 's0_stack_bowls')
    monkeypatch.delenv('ROBODOJO_INTERFACE_ALLOW_OPERATOR_RESET', raising=False)
    write_json(tmp_path/'plan.json', {'max_steps':2})
    write_json(tmp_path/'owner.json', identity(__import__('os').getpid()))
    class Task:
        env_seeds=[0]
        step_lim=2
        task_name='stack_bowls'
        count=0
        def get_obs(self): return {'sequence':self.count}
        def is_episode_end(self): return self.count>=2
        def take_action(self,action): self.count+=1
    task=Task()
    received=[]
    class Client:
        def call(self,func_name,obs=None):
            if func_name=='update_obs': received.append(obs['sequence'])
            if func_name=='get_action': return [{'joint':.1}]
            if func_name=='benchmark_control':
                if obs['command']=='identity': return {'episode_id':'one','instruction':'test'}
                if obs['command']=='hold': return {'joint':.1}
    original=episode.read_json
    def read(path,default=None):
        if path.name=='episode_ack.json':
            event=original(tmp_path/'episode.json')
            return {'event_id':event['event_id'], 'state':
                    'closed' if event['state']=='ended' else 'running'}
        return original(path,default)
    monkeypatch.setattr(episode,'read_json',read)
    episode.eval_episode(task,Client())
    assert received==[0,1,2]
    assert read_json(tmp_path/'episode.json')['state']=='ended'


@pytest.mark.parametrize('model_outage', [False, True])
def test_supervisor_waits_for_official_score_after_stopping_active_agent(tmp_path, monkeypatch, model_outage):
    """An evaluator may take several polls to write its official receipt."""
    import run as supervisor
    run = tmp_path/'runs/test-close'
    workspace = run/'ARX_X5'
    workspace.mkdir(parents=True)
    official = run/'source-result.json'
    write_json(official, {'eval_time':1, 'score':15,
                         'details':{'0':{'layout_id':0,'score':.15,'success':False}}})
    write_json(run/'plan.json', {'chunks':[{'result_path':str(official)}]})
    args = SimpleNamespace(id='test-close', task='stack_bowls', gpu=1, port=19090,
                           seed=0, layout=0, model='NVFP4', base_url='http://local',
                           session_timeout=259200, adopt=False)
    class Process:
        pid = 123
        returncode = None
        polls = 0
        def __init__(self, evaluator=False): self.evaluator=evaluator
        def poll(self):
            self.polls+=1
            if self.evaluator and self.polls>=4: self.returncode=0
            return self.returncode
    server, evaluator, agent = Process(), Process(True), Process()
    stream = SimpleNamespace(poll=lambda:None, result=None, model='NVFP4',
                             reply_at='now', tools=1, thinking=1)
    monkeypatch.setattr(supervisor, '__file__', str(tmp_path/'run.py'))
    probes = 0
    def model_health(*_):
        nonlocal probes
        probes += 1
        if model_outage and probes == 1:
            raise RuntimeError('Temporary model outage')
        return {'route':'A-test'}
    monkeypatch.setattr(supervisor, 'model_health', model_health)
    monkeypatch.setattr(supervisor, 'launch', lambda *_, **kw:(
        {'workspace':str(workspace)}, server, evaluator))
    monkeypatch.setattr(supervisor, 'start_agent', lambda *_:(agent, stream))
    monkeypatch.setattr(supervisor, 'identity', lambda pid:{'pid':pid, 'start_ticks':1})
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *_:None)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _:None)
    ticks = iter(range(0, 100000, 10))
    monkeypatch.setattr(supervisor.time, 'monotonic', lambda: next(ticks))
    stopped=[]
    def stop(process):
        if process is not None:
            stopped.append(process)
            process.returncode=-15
    monkeypatch.setattr(supervisor, 'stop_child', stop)
    original_read=supervisor.read_json
    events=0
    def read(path, default=None):
        nonlocal events
        if path.name=='episode.json':
            events+=1
            if model_outage and probes == 1:
                # Inference is unavailable, yet no evaluator is stopped and
                # its running acknowledgement permits idle heartbeat polls.
                assert stopped == []
                assert original_read(run/'episode_ack.json')['state'] == 'running'
            return {'event_id':'one','state':'ready' if events <= (2 if model_outage else 1) else 'ended',
                    'layout_id':0,'task':'stack_bowls'}
        return original_read(path,default)
    monkeypatch.setattr(supervisor, 'read_json', read)
    assert supervisor.serve(args)==0
    assert evaluator.polls>=4
    assert stopped[0] is agent
    assert read_json(run/'episode_ack.json')['state']=='closed'
    assert read_json(run/'status.json')['score']==.15
    assert read_json(run/'official_result.json')['score']==15
    assert probes == (2 if model_outage else 1)


def test_existing_process_rejects_reused_pid(monkeypatch):
    import run as supervisor
    monkeypatch.setattr(supervisor,'identity',lambda _:{'pid':42,'start_ticks':200})
    process=supervisor.ExistingProcess({'pid':42,'start_ticks':100})
    assert process.poll()==0


@pytest.mark.parametrize('closing',[False,True])
def test_live_handoff_preserves_children_and_rolls_back_if_closing(tmp_path, monkeypatch, closing):
    import run as supervisor
    import os
    import signal
    args=SimpleNamespace(id='test',task='stack_bowls',gpu=1,port=19090,
                         seed=0,layout=0,model='NVFP4')
    previous={'pid':97,'start_ticks':100}
    children={k:{'pid':n,'start_ticks':100} for k,n in
              [('policy',98),('evaluator',99),('agent',100)]}
    state={**vars(args),'phase':'running','supervisor':previous,**children}
    write_json(tmp_path/'status.json',state)
    write_json(tmp_path/'owner.json',previous)
    ack={'event_id':'one','state':'running'}
    write_json(tmp_path/'episode_ack.json',ack)
    write_json(tmp_path/'episode.json',{'event_id':'one','state':'ended' if closing else 'ready'})
    monkeypatch.setattr(supervisor,'identity',lambda pid:{'pid':pid,'start_ticks':100})
    original_bytes=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:
        str(Path(supervisor.__file__).resolve()).encode()+b'\0--id\0test\0'
        if str(p)=='/proc/97/cmdline' else original_bytes(p))
    signals=[]
    def kill(pid,sig):
        signals.append((pid,sig))
        if sig==signal.SIGKILL:
            assert read_json(tmp_path/'owner.json')['pid']==os.getpid()
    monkeypatch.setattr(supervisor.os,'kill',kill)
    if closing:
        with pytest.raises(RuntimeError,match='closing'):
            supervisor.adopt_running(tmp_path,args)
        assert signals==[(97,signal.SIGSTOP),(97,signal.SIGCONT)]
        assert read_json(tmp_path/'owner.json')==previous
    else:
        updated, processes, retained_ack=supervisor.adopt_running(tmp_path,args)
        assert signals==[(97,signal.SIGSTOP),(97,signal.SIGKILL)]
        assert [p.pid for p in processes]==[98,99,100]
        assert updated['agent']==children['agent']
        assert retained_ack==ack
    assert read_json(tmp_path/'episode_ack.json')==ack


def test_joint_config_only_omits_unused_evaluator_planners(tmp_path):
    import yaml
    from assets import prepare_joint_config
    source=tmp_path/'source/env_cfg'
    (source/'robot').mkdir(parents=True)
    (source/'camera').mkdir()
    robot={'robots':[{'robot_name':'x5','default_root_pos':[-.3,-.45,.765]},
                     {'robot_name':'x5','default_root_pos':[.3,-.45,.765]}]}
    path=source/'robot/dual_x5.yml'
    path.write_text(yaml.safe_dump(robot))
    original=path.read_bytes()
    (source/'camera/native.yml').write_text('width: 640\nheight: 480\n')
    (source/'robot/_robot_info.json').write_text('{}')
    target=prepare_joint_config(tmp_path/'source',tmp_path/'overlay')
    changed=yaml.safe_load((target/'robot/dual_x5.yml').read_text())
    for arm in changed['robots']:
        assert arm.pop('need_planner') is False
    assert changed==robot
    assert path.read_bytes()==original
    assert (target/'camera/native.yml').resolve()==source/'camera/native.yml'
    assert (target/'robot/_robot_info.json').resolve()==source/'robot/_robot_info.json'


def test_stop_reaps_worker_even_when_shell_exits_first():
    import subprocess
    import run as supervisor
    child_code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'
    parent_code=('import subprocess,sys,time; '
                 f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}],stdout=subprocess.PIPE,text=True); '
                 'p.stdout.readline(); print(p.pid,flush=True); time.sleep(60)')
    parent=subprocess.Popen([sys.executable,'-c',parent_code],start_new_session=True,
                            stdout=subprocess.PIPE,text=True)
    worker=int(parent.stdout.readline())
    try:
        assert identity(worker)['pgid']==parent.pid
        supervisor.stop_child(parent,grace_seconds=.2)
        assert parent.poll() is not None
        assert identity(worker) is None
    finally:
        supervisor.stop_child(parent,grace_seconds=.2)
