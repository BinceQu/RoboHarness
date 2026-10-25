"""Registry containing exactly the current public Interface v2 tool names."""

from __future__ import annotations

from typing import Any

from .capabilities import (
    PUBLIC_TOOLS,
    TOOL_CAPABILITIES,
    TOOL_VERSION,
    capability_report,
    validate_submission,
)
from .tools import PUBLIC_TOOL_FUNCTIONS
from .types import ToolSpec, tool_spec


class OfficialToolRegistry(dict[str, ToolSpec]):
    """Sealed registry that ignores legacy decorators trying to replace tools.

    The compatibility UI imports some production skill modules lazily.  Their
    decorators write into the shared skill registry as an import side effect.
    The official evaluator runtime installs this mapping there, so accepting
    those writes would silently route a test tool back to the simulator-backed
    implementation.
    """

    def __init__(self, entries: dict[str, ToolSpec]) -> None:
        dict.__init__(self, entries)
        self._blocked_mutation_count = 0
        self._last_blocked_mutation = ""

    def _block(self, operation: str, key: Any = None) -> None:
        self._blocked_mutation_count += 1
        suffix = "" if key is None else f" key={key!r}"
        self._last_blocked_mutation = f"{operation}{suffix}"

    @property
    def blocked_mutation_count(self) -> int:
        return int(self._blocked_mutation_count)

    @property
    def last_blocked_mutation(self) -> str:
        return str(self._last_blocked_mutation)

    def __setitem__(self, key: str, value: ToolSpec) -> None:
        current = dict.get(self, key)
        if current is value:
            return
        self._block("setitem", key)

    def __delitem__(self, key: str) -> None:
        self._block("delitem", key)

    def clear(self) -> None:
        self._block("clear")

    def pop(self, key: str, default: Any = None) -> Any:
        self._block("pop", key)
        return dict.get(self, key, default)

    def popitem(self) -> tuple[str, ToolSpec]:
        self._block("popitem")
        raise KeyError("official_v2 registry is sealed")

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return dict.__getitem__(self, key)
        self._block("setdefault", key)
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        incoming = dict(*args, **kwargs)
        for key, value in incoming.items():
            self.__setitem__(key, value)

    def __ior__(self, other: Any):
        self.update(other)
        return self


def _description(name: str) -> str:
    capability = TOOL_CAPABILITIES[name]
    return str(
        capability.get("implementation")
        or capability.get("required_rewrite")
        or name
    )


def build_registry(adapter) -> dict[str, ToolSpec]:
    del adapter
    registry: dict[str, ToolSpec] = {}
    for name in PUBLIC_TOOLS:
        fn = PUBLIC_TOOL_FUNCTIONS[name]
        registry[name] = tool_spec(name, fn, _description(name))
    return registry


def install_profile(skills_module, adapter) -> dict[str, ToolSpec]:
    registry = OfficialToolRegistry(build_registry(adapter))
    ensure_profile_installed(skills_module, registry)
    return registry


def ensure_profile_installed(
    skills_module,
    registry: OfficialToolRegistry,
) -> bool:
    """Restore and validate the exact test-owned registry before execution."""
    if tuple(registry) != PUBLIC_TOOLS:
        raise RuntimeError("official_v2 sealed registry public surface mismatch")
    for name in PUBLIC_TOOLS:
        spec = dict.__getitem__(registry, name)
        if spec.fn is not PUBLIC_TOOL_FUNCTIONS[name]:
            raise RuntimeError(
                f"official_v2 registry callable mismatch for {name!r}"
            )

    restored = getattr(skills_module, "SKILL_REGISTRY", None) is not registry
    skills_module.SKILL_REGISTRY = registry
    skills_module.PUBLIC_SKILLS = frozenset(PUBLIC_TOOLS)
    skills_module.TOOL_VERSION = TOOL_VERSION
    return bool(restored)


def profile_report() -> dict[str, Any]:
    return capability_report()


__all__ = [
    "OfficialToolRegistry",
    "build_registry",
    "ensure_profile_installed",
    "install_profile",
    "profile_report",
    "validate_submission",
]
