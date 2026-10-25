"""Python atomic API: Robot().observe() and Robot().step(full_action)."""

from tools.util.cli import request, run_cli
from tools.joint_command import joint_command, validate_command


class Robot:
    def observe(self):
        """Return the latest public RGB-D/state manifest; consume zero ticks."""
        return request(
            {"method": "call", "tool": "capture_head_camera", "arguments": {}}
        )

    def step(self, action):
        """One 14D joint command; return only after its new observation arrives."""
        return request(
            {
                "method": "call",
                "tool": "joint_command",
                "arguments": {"action": validate_command(action)},
            }
        )

    def status(self):
        return request({"method": "status"})


if __name__ == "__main__":
    run_cli("joint_command", joint_command)
