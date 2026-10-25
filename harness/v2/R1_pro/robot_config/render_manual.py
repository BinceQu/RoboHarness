"""Generate one genuine-mesh three-position reference per controlled joint."""
import argparse
import json
import hashlib
from pathlib import Path

from tools.util.model import ACTION_NAMES, LOCKED_INDICES, joints, URDF
from tools.util.render import RobotRenderer

ROOT = Path(__file__).resolve().parents[1]

MEANINGS = {
    "torso_joint1": "Lower torso pitch; changes the trunk's lean and camera position.",
    "torso_joint2": "Middle torso pitch; combines with T1/T3 to change height.",
    "torso_joint3": "Upper torso pitch; combines with T1/T2 to orient the chest/head.",
    "torso_joint4": "Chest yaw. Locked at zero by the submitted evaluator profile.",
    "arm_joint1": "First shoulder rotation: moves the entire arm about the marked axis.",
    "arm_joint2": "Second shoulder rotation: swings the entire upper arm about its local axis.",
    "arm_joint3": "Upper-arm axial rotation: changes the elbow's bending plane.",
    "arm_joint4": "Elbow bend: rotates the forearm relative to the upper arm.",
    "arm_joint5": "Forearm axial rotation: rotates the wrist and gripper about the forearm.",
    "arm_joint6": "Wrist bend: changes fingertip direction relative to the forearm.",
    "arm_joint7": "Final wrist rotation before J8; see the marked local axis.",
    "arm_joint8": "Tool-axis roll. Direct joint commands can rotate it; baseline motion tools normally pin it at zero.",
    "gripper_finger_joint1": "First sliding finger. Position 0 m = closed, 0.05 m = open; action is effort in N.",
    "gripper_finger_joint2": "Opposing sliding finger. Position 0 m = closed, 0.05 m = open; action is effort in N.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path, default=ROOT / "robot_config")
    parser.add_argument("--output", type=Path, default=ROOT / "robot_config")
    parser.add_argument("--joint", help="Render only one joint for visual review")
    args = parser.parse_args()
    output = args.output
    (output / "joints").mkdir(parents=True, exist_ok=True)
    renderer = RobotRenderer(args.mesh_root)
    names = [args.joint] if args.joint else ACTION_NAMES[3:]
    records = []
    for name in names:
        index = ACTION_NAMES.index(name)
        joint = joints()[name]
        renderer.triptych(name).save(output / "joints" / (name + ".png"))
        short = name.removeprefix("left_").removeprefix("right_")
        meaning = MEANINGS.get(name, MEANINGS.get(short, "See the local axis and three-position diagram."))
        limits = joint["limits"]
        unit = "m" if joint["type"] == "prismatic" else "rad"
        command = "0 only (locked)" if index in LOCKED_INDICES else "force in N, −20…+20" if unit == "m" else "absolute angle in rad"
        entry = {"name": name, "action_index": index, "meaning": meaning,
                 "command": command, "position_unit": unit, "position_limits": [limits["lower"], limits["upper"]],
                 "axis_in_joint_frame": joint["axis"].tolist(), "parent": joint["parent"], "child": joint["child"],
                 "diagram": f"joints/{name}.png"}
        records.append(entry)
        description = f"# [{index}] {name}\n\n{meaning}\n\n" + \
            f"Controller slot: `action[{index}]`. Command: **{command}**.\n\n" + \
            f"Joint: `{joint['parent']}` → `{joint['child']}`. Local positive axis: `{joint['axis'].tolist()}`.\n\n" + \
            f"URDF position limits: {limits['lower']:.6g} to {limits['upper']:.6g} {unit}. " + \
            ("The evaluator locks this joint at 0 regardless of the geometric limits.\n\n" if index in LOCKED_INDICES else "\n\n") + \
            "The three panels use identical camera and scale. All other joints are zero. " + \
            "Red labels identify illustrative poses that cannot be commanded. This is a motion explanation, not a collision-safe trajectory.\n\n" + \
            f"![Three reference positions]({name}.png)\n"
        (output / "joints" / (name + ".md")).write_text(description)
        print(name, flush=True)
    if not args.joint:
        neutral = {"torso_joint1": 0., "torso_joint2": 0., "torso_joint3": 0.}
        renderer.three_views(neutral, "R1 Pro · zero-position reference").save(output / "three_views.png")
        manifest = {"urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
                    "renderer": renderer.renderer, "joints": records}
        (output / "joints.json").write_text(json.dumps(manifest, indent=2) + "\n")
        table = ["# Joint and controller index", "", "Axes are local to each joint origin, not the image. Positive rotation uses the right-hand rule.", "",
                 "| Slot | Joint | Meaning | Command |", "|---|---|---|---|",
                 "| 0 | base forward | +X forward | normalized −1…+1; ×0.75 m/s |",
                 "| 1 | base left | +Y left | normalized −1…+1; ×0.75 m/s |",
                 "| 2 | base yaw | +Z counterclockwise | normalized −1…+1; ×1 rad/s |"]
        table += [f"| {r['action_index']} | [{r['name']}](joints/{r['name']}.md) | {r['meaning']} | {r['command']} |" for r in records]
        table += ["", "The steering and wheel joints in model.urdf are driven by the evaluator's base controller. They have no independent action slot. Use slots 0–2; do not invent wheel-angle commands.", "",
                  "Finger position is sensed in metres, while finger commands are forces. Negative effort closes; positive effort opens. Keep a negative effort while carrying an object. Never use a finger position as its force command.", ""]
        (output / "joints.md").write_text("\n".join(table))
    renderer.close()


if __name__ == "__main__":
    main()
