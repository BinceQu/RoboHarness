# Idle-Step Gate Design and Safety Review

## Scope

This change is an independent sidecar. It does not edit, import, or monkey
patch `behavior_interface_eval_test`, the official evaluator, OmniGibson, or
the frontend. The stock interface remains the policy backend and the stock
evaluator remains the simulator owner.

## Existing synchronous boundary

The official evaluator's websocket policy loop is effectively:

```text
observation -> policy.forward(observation) -> env.step(action)
```

The policy call is blocking. Therefore the only protocol-level way to avoid an
idle simulator step without changing the evaluator is to leave the policy call
waiting. The sidecar forwards each observation to the unchanged interface once,
then decides whether to forward the interface response immediately or hold that
same response until the interface becomes active.

## New state machine

```text
evaluator observation
        |
        v
unchanged interface (one request, one response)
        |
        +-- non-action / error / active state --> send exact response
        |
        +-- complete idle hold -----------------> keep exact response pending
                                                     |
                                  tool queued/active, init/live state, error,
                                  diagnostics unavailable, or disconnect
                                                     |
                                                     v
                                             send exact response once
```

The pending response is not recomputed. The sidecar never replays an
observation, so adapter snapshots, odometry, trackers, capture bookkeeping,
and tool generators cannot advance twice.

After any response that was passed through while the interface was active, the
sidecar also passes one subsequent hold response before entering the long idle
wait. This preserves the stock post-action observation/termination transition
for contact settling and task-success checks. Model-only idle time, with no
preceding active response on that connection, is still suppressed immediately.

## Suppression predicate

An action response is suppressible only when all of these are explicitly true:

| Check | Required value |
| --- | --- |
| health | `ok == true` |
| selected action | `action_source == "hold"` |
| downstream policy | `configured == false` |
| episode initialization | `ready == true` |
| live test | state is `idle` or another terminal state |
| backend error | `last_error` is empty/`null` |
| state | `active_skill == null` |
| reset | `reset_pending` is `false` or current-runtime `null` |
| task switch | `task_switch_pending == false` |
| health/simulation | `vision_degraded == false`, `simulation_degraded == false` |

Missing or unexpected fields fail open. All nonterminal live states, including
`planning`, pass through because their next evaluator observation may be needed
to advance an observation-driven planner. A configured downstream model always
passes through because a hold action may be intentional model output.

## Invariants checked before deployment

1. Metadata and action frames are forwarded byte-for-byte.
2. Reset frames are forwarded and produce no synthetic response, matching the
   existing interface protocol. Reset detection follows key presence, as in the
   stock handler.
3. No response is suppressed during initialization, tool queue/execute,
   downstream inference, live motion/planning, reset/task switch, degradation,
   or diagnostic uncertainty; one post-activity transition hold is always
   passed through.
4. HTTP probes are read-only and run in daemon workers, so a slow endpoint
   cannot block websocket I/O. Probe failure sends the cached response.
5. A websocket close cancels an idle wait. Backend failure cannot leave the
   evaluator waiting forever for a response from a dead backend.
6. `--max-idle-wait-s 0` is the intended challenge mode. A positive bound is a
   fail-open transition aid and may emit an idle hold step.

## Important deployment condition

The existing strict HTTP tool boundary considers an observation stale after the
configured `BEHAVIOR_EVAL_TEST_OBSERVATION_MAX_AGE_S` (default five seconds).
Because the evaluator is intentionally blocked while the model thinks, start
the unchanged interface with a sufficiently large explicit value (for example
`86400`). This changes only the admission timeout; it does not refresh or
duplicate observations. Without this setting, a long model thought can be
rejected as stale even though the gate itself is working correctly.

## Rollback

Stop the sidecar and point the evaluator back to the original interface policy
port. Since the implementation is isolated, rollback requires no source
revert and leaves the existing stack untouched.

## Evaluation-policy boundary

The official documentation describes a synchronous policy/action boundary and
counts `--max-steps` in simulator steps. This design uses that protocol fact;
it does not modify the evaluator or assert organizer approval for a delayed
transport response. Validate this sidecar against the rules of the target
challenge or leaderboard before reporting results.

## Verification

The isolated test module covers codec compatibility, strict suppression
classification, planning pass-through, missing-field fail-open behavior,
bounded waits, cancellation, reset forwarding, exact response bytes, and a
fake end-to-end proxy exchange. It intentionally does not launch Isaac Sim;
the final acceptance check should compare evaluator step count/FPS and tool
completion on one real port before enabling other ports.
