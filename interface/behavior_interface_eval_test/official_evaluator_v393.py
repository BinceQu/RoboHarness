"""Single-port transport adaptation for the official v3.9.3 evaluator.

The simulator, metrics, timeout, loading and finite-rollout writer remain the
official BatchedEvaluator implementation. Each process has exactly one logical
environment; only the websocket observation loses the leading batch dimension
to match the existing single-port interface.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from behavior_interface_eval_test.rollout_budget import (
    ROLLOUT_BUDGET_KEY, budget_snapshot, unavailable,
)


def route_evaluator_class(base_cls):
    class RouteEvaluator(base_cls):
        # v3.9.3 load_batch already resets the scene, policy, observations,
        # and metrics. The resident driver must not immediately do it again.
        load_task_instance_resets_rollout = True

        def __init__(self, cfg):
            if int(cfg.get("num_envs", 1)) != 1:
                raise ValueError("A route must own exactly one logical environment")
            self._budget_episode_id = uuid.uuid4().hex
            self._resident_tick_budget = None
            super().__init__(cfg)

        @property
        def route_state(self):
            return self.instance_eval_states[0]

        def _batch_obs(self):
            # Sensor fields stay untouched. Counters travel separately from
            # policy observations and never contain goal truth or object state.
            return {**self.route_state.obs, ROLLOUT_BUDGET_KEY: self._rollout_budget()}

        def _rollout_budget(self):
            try:
                if self._resident_tick_budget is not None:
                    used, total = self._resident_tick_budget
                    source = 'resident_active_steps'
                else:
                    # Read the exact counter and cap used by Timeout._step.
                    used = self.env.episode_steps[0]
                    total = self.env.task._termination_conditions['timeout']._max_steps
                    source = 'official_evaluator_episode_steps'
                return budget_snapshot(used, total, episode_id=self._budget_episode_id,
                                       instance_id=self.route_state.instance_id, source=source)
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                return unavailable('evaluator_counter_unavailable')

        def reset(self):
            result = super().reset()
            self._budget_episode_id = uuid.uuid4().hex
            if self._resident_tick_budget is not None:
                self._resident_tick_budget = (0, self._resident_tick_budget[1])
            return result

        def load_batch(self, env_idx_to_instance, **kwargs):
            if set(env_idx_to_instance) != {0}:
                raise ValueError("A route can load only logical environment 0")
            result = super().load_batch(env_idx_to_instance, **kwargs)
            self._budget_episode_id = uuid.uuid4().hex
            self._operator_current_instance_id = int(env_idx_to_instance[0])
            port = os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "")
            if port.isdigit():
                from behavior_interface_eval_test.operator_scene_control import write_status, LISTENER_NAME
                write_status(int(port), listener=LISTENER_NAME, state="ready",
                             current_instance_id=self._operator_current_instance_id, error="")
            return result

        def load_task_instance(self, instance_id):
            return self.load_batch({0: int(instance_id)})

        def step(self):
            terminated, truncated = super()._step_fn([0])
            return bool(terminated[0]), bool(truncated[0])

        def _step_fn(self, active_env_indices):
            if list(active_env_indices) != [0]:
                raise ValueError("A route can step only logical environment 0")
            # The existing operator finish hook wraps step(), and therefore
            # remains effective for both native run() and the resident loop.
            terminated, truncated = self.step()
            return [terminated], [truncated]

        def start_recording(self, path, rate):
            from omnigibson.eval.utils.obs_utils import create_video_writer
            self._set_video_writer(self.route_state, create_video_writer(
                fpath=path, resolution=(448, 672), rate=rate))

        def stop_recording(self):
            self._set_video_writer(self.route_state, None)

        def record_resident_frame(self):
            if self.route_state.video_writer is not None:
                self._write_video(self.route_state)

        def resident_result(self):
            metrics = {}
            for metric in self.route_state.metrics:
                metrics.update(metric.aggregate())
            return bool(self.route_state.env_accessor.success), metrics

    RouteEvaluator.__name__ = "OfficialSingleRouteEvaluator"
    return RouteEvaluator


def run_finite_route(eval_module, evaluator_cls):
    """Run user-selected instances serially through the native result writer."""
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    from omnigibson.macros import gm

    args = eval_module.parse_args()
    if args.num_envs != 1:
        raise ValueError("The single-port interface requires --num-envs=1")
    gm.HEADLESS = args.headless
    if args.headless and os.getenv("OMNIGIBSON_KEEP_VIEWER_CAMERA", "0") != "1":
        gm.RENDER_VIEWER_CAMERA = False
    seed = seed_everything(DEFAULT_EVAL_SEED)
    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    robot = OmegaConf.load(str(Path(args.robot_config).expanduser())) if args.robot_config else None
    model = ({"_target_": "omnigibson.eval.policies.WebsocketPolicy",
              "host": args.host, "port": args.port,
              "action_chunk_size": args.replay_action_chunk_size}
             if args.policy == "websocket" else
             {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None})
    cfg = OmegaConf.create({
        "env_wrapper": {"_target_": args.env_wrapper}, "policy_name": args.policy,
        "model": model, "headless": args.headless, "partial_scene_load": True,
        "max_steps": args.max_steps, "write_video": args.write_video,
        "mode": args.mode, "seed": seed, "num_envs": 1,
        "task": {"name": args.task_name}, "robot": robot,
    })
    output = Path(args.output_dir).expanduser()
    (output / "json").mkdir(parents=True, exist_ok=True)
    if args.write_video:
        (output / "videos").mkdir(parents=True, exist_ok=True)
    with evaluator_cls(cfg) as evaluator:
        for instance_id in instance_ids:
            for rollout_id in range(args.num_rollouts):
                evaluator.run([int(instance_id)], write_video=args.write_video,
                              video_path=str(output / "videos"),
                              metrics_dir=str(output / "json"),
                              rollout_id=rollout_id, video_fps=args.video_fps)
