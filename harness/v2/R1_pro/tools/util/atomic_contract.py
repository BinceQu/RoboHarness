"""Single submitted R1 Pro controller profile; no legacy robot variants."""
import numpy as np
from .model import ACTION_SLICES, PROPRIO_SLICES

ACTION_DIM, PROPRIO_DIM, ARM_DOF = 27, 65, 8
ROBOT_PROFILE = "r1pro_8dof_hf250"
ROBOT_MODEL = "r1pro"
BASE_FOOTPRINT_RADIUS_M = 0.42
LOCKED_ARM_JOINT_INDEX, LOCKED_ARM_JOINT_VALUE = 7, 0.0
LOCKED_TRUNK_YAW_INDEX, LOCKED_TRUNK_YAW_VALUE = 3, 0.0


def enforce_locked_arm_joint(values, *, copy=True):
    result = np.array(values, copy=copy).reshape(-1)
    result[7] = 0
    return result


def enforce_locked_trunk_joint(values, *, copy=True):
    result = np.array(values, copy=copy).reshape(-1)
    result[3] = 0
    return result


def enforce_locked_action(values, *, copy=True):
    result = np.array(values, copy=copy).reshape(-1)
    result[[6, 14, 24]] = 0
    return result
