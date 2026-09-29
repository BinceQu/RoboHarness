"""Canonical R1Pro gripper geometry calibration in the simulator EEF frame."""

from __future__ import annotations


# Simulator FK gives gripper_link -> eef_link as 60 mm along gripper -Z.
# The gripper geometry is rotated by Ry(pi) into the EEF convention.
R1PRO_GRIPPER_LINK_Z_EEF_M = -0.06000
R1PRO_FINGER_JOINT_Z_GRIPPER_M = -0.03689
R1PRO_FINGER_LINK_Z_EEF_M = (
    R1PRO_GRIPPER_LINK_Z_EEF_M - R1PRO_FINGER_JOINT_Z_GRIPPER_M
)
R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM = (
    0.0,
    0.0,
    R1PRO_GRIPPER_LINK_Z_EEF_M,
    0.0,
    0.0,
    1.0,
    0.0,
)
