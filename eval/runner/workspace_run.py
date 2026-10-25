"""V2 runs: each case gets a fresh R1_pro workspace whose tools drive the evaluator.

Imported by workspace.py after it re-executes under the workspace Python.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import uuid

from tools.util.protocol import request
from tools.util.session import create
from tools.util.receipt import camera_observation_ready
from runner import core as runner

HERE = Path(__file__).resolve().parent
TEMPLATE = runner.ROOT / "harness/v2/R1_pro"


def workspace_sources():
    sources = {}
    for folder, directories, files in os.walk(TEMPLATE):
        directories[:] = [d for d in directories if not d.startswith(".") or d == ".claude"]
        directories[:] = [d for d in directories if d != "__pycache__"]
        for filename in sorted(files):
            path = Path(folder) / filename
            sources[str(path.relative_to(runner.ROOT))] = runner.sha(path.read_bytes())
    for name in ("eval/runner/workspace.py", "eval/runner/workspace_run.py", "harness/v2/R1_pro/requirements.lock"):
        sources[name] = runner.sha((runner.ROOT / name).read_bytes())
    return sources


class WorkspaceRun(runner.Run):
    def prepare(self):
        super().prepare()
        self.socket_dir = Path(tempfile.mkdtemp(prefix="r1-driver-"))
        self.socket = self.socket_dir / "control.sock"
        self.env["ROBOT_COMPILE_TMP"] = str(self.socket_dir / "compile")
        Path(self.env["ROBOT_COMPILE_TMP"]).mkdir()
        runner.atomic_json(self.path / "workspace-source.json", workspace_sources())
        runner.atomic_json(self.path / "driver-config.json", {"socket": str(self.socket), "policy_port": self.policy_port})

    def start_stack(self):
        self.log(f"Direct workspace control on GPU {self.plan['gpu']}, evaluator policy {self.policy_port}; read-only monitor {self.port}")
        self.spawn("driver", [self.config["workspace_python"], "-m", "tools.util.driver",
            "--policy-port", str(self.policy_port), "--socket", str(self.socket), "--status", str(self.path / "driver-status.json")], self.env)
        deadline = time.monotonic() + 120
        while not self.socket.exists():
            self.assert_alive("driver")
            if time.monotonic() > deadline:
                raise TimeoutError("Atomic driver did not start")
            time.sleep(.2)
        self.spawn("monitor", [self.config["workspace_python"], str(HERE / "workspace_monitor.py"),
            "--run-dir", str(self.path), "--socket", str(self.socket), "--port", str(self.port)], self.env)
        env = dict(self.env)
        env["OMNIGIBSON_APPDATA_PATH"] = str(Path(self.env["TMPDIR"]).parent / "appdata")
        # Kit selects the physical Vulkan card separately. CUDA consumers
        # (including Warp's device discovery) must see only that owned card.
        env.update(CUDA_VISIBLE_DEVICES=str(self.plan["gpu"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   OMNIGIBSON_GPU_ID="0", BEHAVIOR_EVAL_TEST_UNMASK_CUDA="0",
                   # Sequential instances may follow a task that deleted objects.
                   BEHAVIOR_EVAL_TEST_RESTORE_TASK_OBJECTS="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [
            str(runner.ROOT / "BEHAVIOR/OmniGibson"), str(runner.ROOT / "BEHAVIOR/bddl3"),
            str(runner.ROOT / "BEHAVIOR/joylo"), self.config["evaluator_dependencies"], self.env["PYTHONPATH"]]))
        self.spawn("evaluator", [self.config["evaluator_python"], "-m", "behavior_interface_eval_test.official_evaluator_entrypoint",
            "--task-name", self.task["task_name"], "--robot-config", str(runner.PROFILE / "evaluator_robot.yaml"),
            "--host", "127.0.0.1", "--port", str(self.policy_port), "--mode", "public_test",
            "--instance-indices", *[str(c["slot"]) for c in self.plan["cases"]], "--num-rollouts", "1",
            "--max-steps", str(self.task["max_steps"]),
            "--env-wrapper", "behavior_interface_eval_test.official_rgbd_wrapper.OfficialRGBDFullResWrapper",
            "--output-dir", str(self.path / "output")], env)

    def wait_ready(self, case):
        deadline, last_log, stable = time.monotonic() + self.config["startup_timeout_s"], 0, 0
        while time.monotonic() < deadline:
            self.assert_alive("driver", "evaluator")
            driver = request(self.socket, {"operation": "status"})
            path = self.path / "operator" / f"behavior_eval_operator_status_p{self.port}.json"
            status = runner.read_json(path) if path.exists() else {}
            ready = (status.get("current_instance_id") == case["instance_id"] and driver["connected"]
                     and driver["initialization"]["ready"] and driver["session_id"] is None)
            stable = stable + 1 if ready else 0
            if stable >= 2:
                self.log(f"instance {case['instance_id']} ready at sequence {driver['sequence']}")
                return
            if time.monotonic() - last_log > 60:
                self.log(f"Waiting for {case['instance_id']}: evaluator={status.get('current_instance_id')} initialization={driver['initialization']['state']}")
                last_log = time.monotonic()
            time.sleep(2)
        raise TimeoutError("Evaluator episode initialization did not finish")

    def start_agent(self, case):
        sid = f"{self.task['task']}i{case['instance_id']}-{uuid.uuid4().hex[:12]}"
        case_dir = self.path / f"instance_{case['instance_id']}"
        case_dir.mkdir()
        cwd = case_dir / "workspace"
        create(TEMPLATE, cwd, self.socket, session_id=sid, task_name=self.task["task_name"])
        raw = (runner.ROOT / case["prompt"]).read_text()
        prompt = re.sub(r"\b1506\d\b", str(self.port), raw)
        preamble = ("Use this isolated R1_pro workspace. Read AGENTS.md, the baseline Skill, tools/README.md, "
            "and tool files or --schema as needed. Named tools/skills are Bash calls: "
            "python tools/TOOL_NAME.py '{\"argument\":\"value\"}'. "
            "Before every click inspect the newest attached head image; use its current image_id and original "
            "720 x 720 pixels. If only a path is shown, use native Read. This session's perception/history are "
            "authoritative; do not read other episodes, evaluator/source/answers/scores or modify tools/task instructions.\n\n")
        (case_dir / "prompt.source.txt").write_text(raw)
        (case_dir / "prompt.txt").write_text(preamble + prompt)
        runner.atomic_json(case_dir / "case.json", {**case, "session_id": sid, "agent_cwd": str(cwd),
            "rendered_prompt_sha256": runner.sha((preamble + prompt).encode()), "comparison": "direct workspace transport; parent task unchanged"})
        runner.atomic_json(self.path / "active_session.json", {"port": self.port, "session_id": sid,
            "workspace": str(cwd), "owner_pid": os.getpid(), "owner_start": runner.process_start(os.getpid()), "run_id": self.token})
        env = dict(self.env)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        session_python = str(cwd / ".venv/bin/python")
        for name in list(env):
            if name.startswith(("GIT_", "ANTHROPIC_", "CLAUDE_")):
                env.pop(name)
        env.update(ROBOT_WORKSPACE=str(cwd), CLAUDE_BIN=self.config["claude_bin"],
            EMBODIED_ANTHROPIC_BASE_URL=self.config["model_url"], QWEN_MODEL=self.config["model"],
            PATH=str(Path(session_python).parent) + os.pathsep + env["PATH"])
        # Populate senses before the agent starts; no joint command is sent.
        with (case_dir / "initial-observation.json").open("w") as stream:
            subprocess.run([session_python, str(cwd / "tools/capture_head_camera.py")], cwd=cwd,
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
        initial = runner.read_json(case_dir / "initial-observation.json")
        if not camera_observation_ready(initial):
            raise RuntimeError(f"Initial workspace observation failed: {initial.get('error')}")
        command = ([session_python, str(HERE / "workspace_smoke.py")]
            if self.plan.get("smoke") else [session_python, "-m", "tools.util.launch",
            "--workspace", str(cwd), "--", "-p", "--output-format", "json"])
        process = self.spawn("agent", command, env,
            stdin=case_dir / "prompt.txt", stdout=case_dir / "agent.json", stderr=case_dir / "agent.stderr.log", cwd=cwd)
        self.write_summary("running")
        self.log(f"instance {case['instance_id']}: fresh workspace {cwd}")
        return process, case_dir

    def finish_episode(self, reason):
        super().finish_episode(reason)
        request(self.socket, {"operation": "finish"})

    def wait_score(self, case, agent, case_dir):
        path = self.path / "output/json" / f"{self.task['task_name']}_{case['instance_id']}_0.json"
        finished_at, reason, last_log = None, None, 0
        while True:
            result = runner.official_result(path, self.task, case["instance_id"]) if path.exists() else None
            if result is not None:
                break
            self.assert_alive("driver", "evaluator")
            code = agent.poll()
            if code is not None and finished_at is None:
                if code != 0:
                    raise RuntimeError(f"Agent failed ({code}): {case_dir / 'agent.stderr.log'}")
                final = runner.read_json(case_dir / "agent.json")
                if final.get("type") != ("smoke_result" if self.plan.get("smoke") else "result") or final.get("is_error"):
                    raise RuntimeError("Invalid native Claude final result")
                self.finish_episode("model_done")
                finished_at, reason = time.monotonic(), "model_done"
            if finished_at and time.monotonic() - finished_at > 600:
                raise TimeoutError("Official evaluator failed to write score after hand-in")
            if time.monotonic() - last_log > 60:
                status = request(self.socket, {"operation": "status"})
                self.log(f"instance {case['instance_id']}: agent={code if code is not None else 'running'} ticks={status['ticks']}")
                last_log = time.monotonic()
            time.sleep(2)
        runner.atomic_json(self.path / "active_session.json", {})
        if agent.poll() is None:
            os.killpg(agent.pid, signal.SIGTERM)
            try:
                agent.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(agent.pid, signal.SIGKILL)
                agent.wait(timeout=5)
        row = {"instance_id": case["instance_id"], "q": result["q_score"]["final"],
            "reference_q": case["reference_q"], "delta_q": result["q_score"]["final"] - case["reference_q"],
            "steps": result["steps"], "success": result["success"], "result": str(path.relative_to(self.path)),
            "result_sha256": runner.sha(path.read_bytes()), "finish_reason": reason or "evaluator_end"}
        initial = runner.read_json(self.path / "workspace-source.json")
        changed = []
        for name, digest in initial.items():
            if name.startswith("harness/v2/R1_pro/"):
                source = case_dir / "workspace" / name.removeprefix("harness/v2/R1_pro/")
                if not source.is_file() or runner.sha(source.read_bytes()) != digest:
                    changed.append(name)
        row["workspace_source_changes"] = changed
        runner.atomic_json(case_dir / "comparison.json", row)
        self.results.append(row)
        self.write_summary("running")
        self.log(f"instance {case['instance_id']}: Q={row['q']:.6f}")
