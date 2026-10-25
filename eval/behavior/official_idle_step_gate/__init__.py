"""Protocol-transparent idle-step gate for the official evaluator.

The package is deliberately independent from ``behavior_interface_eval_test``.
It can be deployed as a sidecar between the stock evaluator and an unchanged
policy interface.  The sidecar never imports OmniGibson and never synthesizes
or replays an action.

Exports are resolved lazily so ``python -m official_idle_step_gate.proxy`` does
not import the module once during package initialization and again as the
requested module.
"""

_EXPORTED_NAMES = {
    "ActivitySnapshot",
    "BackendActivityReader",
    "GateDecision",
    "IdleStepGate",
    "IdleStepGateProxy",
    "ProxyConfig",
}

__all__ = [
    "ActivitySnapshot",
    "BackendActivityReader",
    "GateDecision",
    "IdleStepGate",
    "IdleStepGateProxy",
    "ProxyConfig",
]


def __getattr__(name: str):
    if name not in _EXPORTED_NAMES:
        raise AttributeError(name)
    from . import proxy

    return getattr(proxy, name)
