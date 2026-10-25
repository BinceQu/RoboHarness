"""Native SessionStart context; never edits or summarizes conversation history."""

import json
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    text = (
        (root / "README.md").read_text()
        + "\n"
        + (root / "skills/robodojo-baseline/SKILL.md").read_text()
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": text,
                }
            }
        )
    )
