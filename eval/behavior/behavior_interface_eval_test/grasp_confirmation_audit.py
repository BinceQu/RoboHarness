"""Test-only oracle audit for the observation-only grasp confirmation.

The privileged oracle is deliberately evaluator-side and opt-in. Its samples
must never be attached to an observation, action, tool result, or HTTP response.
The policy process writes only its already-public ``close_gripper`` result to a
different JSONL file. This module joins those files offline by host timestamp.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


ORACLE_ENABLE_ENV = "BEHAVIOR_EVAL_TEST_ENABLE_PRIVILEGED_GRASP_ORACLE"
ORACLE_TRACE_ENV = "BEHAVIOR_EVAL_TEST_PRIVILEGED_GRASP_ORACLE_PATH"
PREDICTION_TRACE_ENV = "BEHAVIOR_EVAL_TEST_GRASP_PREDICTION_TRACE_PATH"
AUDIT_SCHEMA = "behavior_grasp_confirmation_audit_v1"
TRUTH_DEFINITION = (
    "object_reference_present_and_valid_assisted_grasp_constraint_present"
)


def _absolute_env_path(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return os.path.abspath(os.path.expanduser(raw))


def privileged_oracle_trace_path() -> str | None:
    """Return the oracle path only after an explicit privileged-mode opt-in."""
    enabled = os.environ.get(ORACLE_ENABLE_ENV, "0").strip()
    if enabled not in {"0", "1"}:
        raise ValueError(f"{ORACLE_ENABLE_ENV} must be 0 or 1")
    path = _absolute_env_path(ORACLE_TRACE_ENV)
    if enabled == "0":
        return None
    if path is None:
        raise ValueError(
            f"{ORACLE_TRACE_ENV} is required when {ORACLE_ENABLE_ENV}=1"
        )
    return path


def compliant_prediction_trace_path() -> str | None:
    """Return the non-privileged policy-result trace path, if configured."""
    return _absolute_env_path(PREDICTION_TRACE_ENV)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _JSONLWriter:
    """Append complete JSON records with owner-only file permissions."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        parent = os.path.dirname(self.path)
        os.makedirs(parent, mode=0o700, exist_ok=True)
        self._fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(
                _jsonable(record),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(self._fd, remaining)
                if written <= 0:
                    raise OSError("short write while appending grasp audit JSONL")
                remaining = remaining[written:]

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


_prediction_writer_lock = threading.Lock()
_prediction_writers: dict[str, _JSONLWriter] = {}


def _prediction_writer(path: str) -> _JSONLWriter:
    with _prediction_writer_lock:
        writer = _prediction_writers.get(path)
        if writer is None:
            writer = _JSONLWriter(path)
            _prediction_writers[path] = writer
        return writer


def record_compliant_grasp_prediction(
    tool_name: str,
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Persist only the legal close result; never inspect simulator state."""
    if str(tool_name) != "close_gripper":
        return None
    path = compliant_prediction_trace_path()
    if path is None:
        return None
    record = {
        "schema": AUDIT_SCHEMA,
        "record_type": "compliant_prediction",
        "privileged_test_only": False,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "arm": str(result.get("arm") or ""),
        "grasp_confirmed": bool(result.get("grasp_confirmed", False)),
        "ok": bool(result.get("ok", False)),
        "failure_stage": result.get("failure_stage"),
        "failure_reason": result.get("failure_reason"),
        "confirmation_source": result.get("confirmation_source"),
        "action_steps": result.get("action_steps"),
        "gripper_qpos_before": result.get("gripper_qpos_before"),
        "gripper_qpos_after": result.get("gripper_qpos_after"),
        "bilateral_contact_ever_observed": result.get(
            "bilateral_contact_ever_observed"
        ),
    }
    _prediction_writer(path).append(record)
    return record


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_grasp_confirmation(
    oracle_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    max_age_s: float = 5.0,
) -> dict[str, Any]:
    """Align each prediction to the latest preceding oracle sample per arm."""
    if max_age_s <= 0:
        raise ValueError("max_age_s must be positive")
    samples: dict[str, list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]] = {}
    for record in oracle_records:
        if record.get("record_type") != "privileged_oracle_sample":
            continue
        try:
            sample_time = int(record["time_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        arms = record.get("arms")
        if not isinstance(arms, Mapping):
            continue
        for arm, truth in arms.items():
            if isinstance(truth, Mapping) and isinstance(
                truth.get("established"), bool
            ):
                samples.setdefault(str(arm), []).append(
                    (sample_time, record, truth)
                )
    for arm_samples in samples.values():
        arm_samples.sort(key=lambda item: item[0])

    paired: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    max_age_ns = int(max_age_s * 1_000_000_000)
    for index, prediction in enumerate(prediction_records, start=1):
        if prediction.get("record_type") != "compliant_prediction":
            continue
        arm = str(prediction.get("arm") or "")
        try:
            prediction_time = int(prediction["time_ns"])
        except (KeyError, TypeError, ValueError):
            unpaired.append({"prediction_index": index, "reason": "missing_time"})
            continue
        arm_samples = samples.get(arm, [])
        times = [item[0] for item in arm_samples]
        sample_index = bisect.bisect_right(times, prediction_time) - 1
        if sample_index < 0:
            unpaired.append(
                {
                    "prediction_index": index,
                    "arm": arm,
                    "time_ns": prediction_time,
                    "reason": "no_preceding_oracle_sample",
                }
            )
            continue
        sample_time, sample, truth = arm_samples[sample_index]
        age_ns = prediction_time - sample_time
        if age_ns > max_age_ns:
            unpaired.append(
                {
                    "prediction_index": index,
                    "arm": arm,
                    "time_ns": prediction_time,
                    "reason": "stale_oracle_sample",
                    "oracle_age_ms": age_ns / 1_000_000.0,
                }
            )
            continue
        predicted = bool(prediction.get("grasp_confirmed", False))
        actual = bool(truth["established"])
        outcome = (
            "tp" if predicted and actual else
            "fp" if predicted else
            "fn" if actual else
            "tn"
        )
        paired.append(
            {
                "prediction_index": index,
                "arm": arm,
                "prediction_time_ns": prediction_time,
                "oracle_time_ns": sample_time,
                "oracle_age_ms": age_ns / 1_000_000.0,
                "oracle_sequence": sample.get("sequence"),
                "predicted": predicted,
                "actual": actual,
                "outcome": outcome,
                "object_name": truth.get("object_name"),
                "object_prim_path": truth.get("object_prim_path"),
                "constraint_prim_path": truth.get("constraint_prim_path"),
                "confirmation_source": prediction.get("confirmation_source"),
                "failure_stage": prediction.get("failure_stage"),
                "failure_reason": prediction.get("failure_reason"),
            }
        )

    counts = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    for pair in paired:
        counts[pair["outcome"]] += 1
    total = len(paired)
    return {
        "schema": AUDIT_SCHEMA,
        "privileged_test_only": True,
        "truth_definition": TRUTH_DEFINITION,
        "max_oracle_age_s": max_age_s,
        "paired_predictions": total,
        "unpaired_predictions": len(unpaired),
        "confusion_matrix": counts,
        "metrics": {
            "accuracy": _rate(counts["tp"] + counts["tn"], total),
            "precision": _rate(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _rate(counts["tp"], counts["tp"] + counts["fn"]),
            "specificity": _rate(counts["tn"], counts["tn"] + counts["fp"]),
            "false_positive_rate": _rate(
                counts["fp"], counts["fp"] + counts["tn"]
            ),
            "false_negative_rate": _rate(
                counts["fn"], counts["fn"] + counts["tp"]
            ),
        },
        "mismatches": [
            pair for pair in paired if pair["outcome"] in {"fp", "fn"}
        ],
        "pairs": paired,
        "unpaired": unpaired,
    }


def analyze_trace_files(
    oracle_path: str | os.PathLike[str],
    prediction_path: str | os.PathLike[str],
    *,
    max_age_s: float = 5.0,
) -> dict[str, Any]:
    return analyze_grasp_confirmation(
        read_jsonl(oracle_path),
        read_jsonl(prediction_path),
        max_age_s=max_age_s,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline comparison of compliant grasp_confirmed results against "
            "a prohibited evaluator-side assisted-grasp oracle."
        )
    )
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--max-age-s", type=float, default=5.0)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)
    report = analyze_trace_files(
        args.oracle,
        args.predictions,
        max_age_s=args.max_age_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_mismatch and report["mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
