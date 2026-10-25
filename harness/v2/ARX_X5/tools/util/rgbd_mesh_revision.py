"""Bind each pure reconstruction to one immutable, auditable mesh revision.

No simulator/service reload or action is performed. Keeping the completed
result and its functions independent of later source edits avoids mixing an
old capture's geometry with newly mutated module globals during planning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import threading
from types import ModuleType

_SOURCE = Path(__file__).with_name("rgbd_scene_mesh_robodojo_v1.py")
_LOCK = threading.RLock()
_REVISION: tuple[str, ModuleType] | None = None


def reconstruct_scene_mesh_from_snapshot(snapshot, **options):
    global _REVISION
    payload = _SOURCE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with _LOCK:
        if _REVISION is None or _REVISION[0] != digest:
            name = f"{__package__}._rgbd_mesh_revision_{digest}"
            module = ModuleType(name)
            module.__package__ = __package__
            module.__file__ = str(_SOURCE)
            module.__dict__["__builtins__"] = __builtins__
            # dataclasses resolves postponed field annotations via __module__.
            previous = sys.modules.get(name)
            sys.modules[name] = module
            try:
                exec(compile(payload, str(_SOURCE), "exec"), module.__dict__)
                if not callable(module.reconstruct_scene_mesh_from_snapshot):
                    raise TypeError("revision has no reconstruction function")
            except Exception as error:
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
                raise RuntimeError(f"RGB-D mesh revision rejected: {error}") from error
            _REVISION = (digest, module)
        revision = _REVISION[1]
    result = revision.reconstruct_scene_mesh_from_snapshot(snapshot, **options)
    result.metadata["mesh_source_sha256"] = digest
    result.metadata["mesh_revision_binding"] = (
        "immutable_bytes_per_reconstruction_no_runtime_actions"
    )
    return result
