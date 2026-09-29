"""interface 共用异常。"""


class SkillCancelled(Exception):
    """Skill 被 /api/skill/cancel 或用户中断。"""


class GraspObjPlanningError(Exception):
    """grasp_obj 规划前置条件不满足（应立刻失败并提示用户）。"""
