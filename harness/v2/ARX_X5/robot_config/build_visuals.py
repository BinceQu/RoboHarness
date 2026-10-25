"""Build dual-arm URDF and joint triptychs using only local robot assets."""

import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def build_model():
    root = Path(__file__).resolve().parent
    source = root / "single_arm.urdf"
    if not source.exists():
        source.write_bytes((root / "model.urdf").read_bytes())
    model = ET.Element("robot", name="dual_arx_x5")
    ET.SubElement(model, "link", name="base_link")
    for side, x in [("left", -0.3), ("right", 0.3)]:
        for element in ET.parse(source).getroot():
            node = copy.deepcopy(element)
            if node.tag in ("joint", "link"):
                node.set("name", side + "_" + node.get("name"))
            for item in node.iter():
                if item.tag in ("parent", "child"):
                    item.set("link", side + "_" + item.get("link"))
                if item.tag == "mimic":
                    item.set("joint", side + "_" + item.get("joint"))
            model.append(node)
        mount = ET.SubElement(model, "joint", name=side + "_mount", type="fixed")
        ET.SubElement(mount, "parent", link="base_link")
        ET.SubElement(mount, "child", link=side + "_base_link")
        ET.SubElement(mount, "origin", xyz=f"{x} -0.45 0.765", rpy=f"0 0 {math.pi / 2}")
    ET.indent(model)
    ET.ElementTree(model).write(
        root / "model.urdf", encoding="utf-8", xml_declaration=True
    )


def main():
    build_model()
    from tools.util.render import RobotRenderer
    from tools.util.model import ACTION_NAMES

    root = Path(__file__).resolve().parent
    (root / "visuals").mkdir(exist_ok=True)
    renderer = RobotRenderer(root)
    try:
        for name in ACTION_NAMES:
            renderer.triptych(name).save(root / "visuals" / f"{name}.png")
        renderer.three_views(
            {}, caption="ARX X5: two fixed 6-DOF arms; all arm joints = 0 rad"
        ).save(root / "visuals/three_views.png")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
