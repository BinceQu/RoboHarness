#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def build_context(plugin_root: Path, active_skill: str = "", revision: int = 0) -> str:
    session_context = (plugin_root / "prompts" / "session-context.md").read_text(
        encoding="utf-8"
    ).strip()
    baseline = (
        plugin_root / "skills" / "behavior-v2-baseline" / "SKILL.md"
    ).read_text(encoding="utf-8").strip()
    import skill_router

    catalog = skill_router.render_task_skill_catalog(plugin_root)
    parts = [
        skill_router.skill_state.state_notice(
            skill_router.skill_state.SkillState(active_skill or None, revision)
        ),
        session_context,
        catalog,
        (
            '<activated_skill name="behavior-v2-baseline">\n'
            + baseline
            + "\n</activated_skill>"
        ),
    ]
    # Restore only the current workflow, independently of native historical
    # Skill reattachments. Always append an explicit state, including null.
    extra = skill_router.render_activated_task_skill(plugin_root, active_skill)
    if extra:
        parts.append(extra)
    parts.append(skill_router.skill_state.state_notice(
        skill_router.skill_state.SkillState(active_skill or None, revision)
    ))
    return "\n\n".join(parts)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    # compact / resume / clear 也会再进 SessionStart。这里绝不能
    # new_attempt：否则同一 BEHAVIOR_SESSION_ID 会被切盘，直播卡片从零开始，
    # 看起来像「记忆只留到切盘那一刻」。新一轮 exec 仍由 launcher 带
    # new_attempt 开盘。任务 Skill 由 activate_skill 上报，这里不要回写成 baseline。
    plugin_root = Path(
        os.environ.get("CLAUDE_PLUGIN_ROOT")
        or os.environ.get("EMBODIED_PLUGIN_ROOT")
        or os.environ.get("PLUGIN_ROOT")
        or Path(__file__).resolve().parents[1]
    )
    import skill_router

    # 只注入，不 publish：监视器标签由 UserPromptSubmit / activate_skill 带上
    # 当前任务 skill，避免这里误写成只有 baseline。
    state = skill_router.skill_state.read_state()
    active = state.active_task_skill if state else None
    context = build_context(plugin_root, active_skill=active or "",
                            revision=state.revision if state else 0)
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
