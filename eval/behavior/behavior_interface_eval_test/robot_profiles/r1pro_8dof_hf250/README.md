# R1Pro 8DOF HF250 Official Profile

This directory is a self-contained OmniGibson v3.9.1 custom robot package.
It does not import robot code or assets from outside `behavior_interface_eval_test`.

- canonical evaluator model: `r1pro`
- embodiment variant: `r1pro_8dof_hf250`
- arm joints: 8 per arm, with J8 included in `arm_left` / `arm_right`
- evaluator action: 27 values; each gripper contributes two independent effort
  commands
- evaluator proprioception: 65 values, using only standard R1Pro qpos, qvel,
  EEF pose, trunk, and base velocity fields
- base link mass: 250kg, baked into USD and URDF; the stock 47.63kg
  inertia tensor is scaled by the same mass ratio
- base motion: forward/reverse, lateral translation and yaw; linear output
  limits are +/-0.75 m/s on both axes and yaw output is bounded to +/-1 rad/s
- base z, roll, and pitch: retained as passive virtual joints for the official
  6DOF base contract, with no DriveAPI and asset-level joint friction `1e9`
- torso motion: `torso_joint4` yaw is locked at zero in both controller limits
  and the policy action boundary
- torso motion effort: `1000` for q1-q4, ten times the stock `100`
- arm effort per side: J1/J2 `110`, J3/J4 `50`, J5-J7 `36`, J8 `20`;
  J1-J7 are twice the stock R1Pro limits
- finger asset effort limit: stock `100`; submitted controller commands are
  independently bounded to `+/-20N` per finger
- controlled base x/y/yaw motion effort: `10000`, ten times the official
  generic Robot runtime limit `1000`
- base velocity-drive damping: x/y `30000`, yaw `1700`; x/y still reach the
  existing `10000N` cap at full command, while all axes stay within the
  submitted mass/inertia's monotone small-error bound at 120Hz and prevent
  zero-command max-force chatter
- runtime actuator mutation: none; effort limits come only from the submitted
  static USD/URDF and standard controller configuration

The grippers use the official `MultiFingerGripperController` with
`motor_type: effort`, `mode: independent`, direct per-finger commands, and no
runtime actuator mutation. `close_gripper` starts both fingers at 0.5 N while
seeking contact. A single independently stalled
finger ramps linearly to 2 N and returns to 0.5 N on renewed motion. Bilateral
contact starts both fingers at 1 N and ramps them synchronously to the
controller limit of 20 N across the existing 12-frame stationary confirmation
window. Inward compliance restarts the stationary timer without resetting the
preload, and the 20 N carry command persists until explicit open.
The policy never observes or creates the evaluator's assisted-grasp constraint.

The uncontrolled z, roll, and pitch joints remain locked by passive
asset-level friction. The official wrapper passes the policy action through
unchanged and contains no runtime robot-joint query, actuator write, pose
teleport, object-state change, task-state change, or privileged observation
path. Include `official_rgbd_wrapper.py` for manual challenge inspection.

The installer never modifies the shared evaluator data root. Live Kit
processes keep inotify watchers on every loaded asset directory, and rewriting
`models/r1pro` underneath them made nine official evaluators abort in the same
second on 2026-09-14 (`unlock() called by non-owning thread`). Instead
`install.py --data-root <root>` prints an immutable sibling root
`<root>__r1pro-<content digest>` whose entries are symlinks into `<root>`
except for a private copy of this profile at `models/r1pro`; the launcher
exports that path as `OMNIGIBSON_DATA_PATH`. The overlay is built once per
content digest and never rewritten, so different profile variants and the
stock robot coexist without touching each other's watched directories. The
evaluator selects the variant through the submitted `evaluator_robot.yaml`,
using the documented `--robot-config` path. `--restore-stock` prints the root
whose `models/r1pro` is the bundled definition (the shared root itself, or a
`__r1pro-stock-<digest>` overlay for roots an older installer rewrote in
place).

The canonical model key is required by the unmodified v3.9.1 evaluator's
initial `BehaviorTask.reset()`: before the challenge TRO is loaded, the scene
metadata contains only an `r1pro`-specific presampled pose. A distinct custom
model registers and initializes its controllers, but that initial reset raises
`No generic or model-specific presampled robot pose`. Keeping the R1Pro model
key avoids patching the official evaluator and must be disclosed as part of
the submitted variant.

The challenge permits custom OmniGibson-supported embodiments and requires
the exact robot config and wrapper to be reproducible. Submit this asset tree,
`evaluator_robot.yaml`, and the wrapper together; do not submit it as an
unchanged bundled R1Pro configuration.

Run `patch_profile_assets.py` only when rebuilding from the copied source
asset. Run `validate.py` before installing, then `install.py --data-root ...`.
