"""Process lifecycle and thread-safe client for the native RTAB-Map worker."""

from __future__ import annotations

import os
from pathlib import Path
import select
import shutil
import sys
import subprocess
import threading
import time
from typing import Optional, Sequence

from .official import BodyOdometry, OfficialObservation, StructuralObservationGate
from .protocol import Command, MapResult, command_packet, frame_packet, read_result


class WorkerUnavailable(RuntimeError):
    pass


class WorkerRequestTimeout(WorkerUnavailable):
    """The native worker did not finish one ordered protocol request."""


_GRACEFUL_SHUTDOWN_TIMEOUT_S = 120.0
_FORCED_STOP_TIMEOUT_S = 2.0
DEFAULT_WORKER_REQUEST_TIMEOUT_S = 30.0


def worker_request_timeout_s() -> float:
    raw = os.environ.get(
        "BEHAVIOR_RTABMAP_REQUEST_TIMEOUT_S",
        str(DEFAULT_WORKER_REQUEST_TIMEOUT_S),
    ).strip()
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise WorkerUnavailable(
            "BEHAVIOR_RTABMAP_REQUEST_TIMEOUT_S must be a number"
        ) from exc
    if not 0.0 < timeout_s <= 3600.0:
        raise WorkerUnavailable(
            "BEHAVIOR_RTABMAP_REQUEST_TIMEOUT_S must be in (0, 3600]"
        )
    return timeout_s


class _DeadlineReader:
    """Apply one monotonic deadline across all reads of a response packet."""

    def __init__(self, stream, timeout_s: float, operation: str) -> None:
        self._stream = stream
        self._deadline = time.monotonic() + float(timeout_s)
        self._timeout_s = float(timeout_s)
        self._operation = str(operation)

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        try:
            descriptor = self._stream.fileno()
        except (AttributeError, OSError):
            # In-memory streams are used by protocol unit tests. Production
            # worker stdout is an unbuffered FileIO object with a descriptor.
            return self._stream.read(size)
        while True:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0.0:
                raise WorkerRequestTimeout(
                    f"RTAB-Map worker {self._operation} timed out after "
                    f"{self._timeout_s:.3f}s"
                )
            try:
                readable, _, _ = select.select(
                    [descriptor], [], [], remaining
                )
            except InterruptedError:
                continue
            if not readable:
                raise WorkerRequestTimeout(
                    f"RTAB-Map worker {self._operation} timed out after "
                    f"{self._timeout_s:.3f}s"
                )
            return self._stream.read(size)


def worker_cpu_affinity() -> tuple[int, ...]:
    """返回 worker 可使用的 CPU；空元组表示当前平台不支持亲和性。"""

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return ()
    allowed = tuple(sorted(int(cpu) for cpu in get_affinity(0)))
    if not allowed:
        return ()

    explicit = os.environ.get("BEHAVIOR_RTABMAP_CPU_LIST", "").strip()
    if explicit:
        try:
            selected = tuple(dict.fromkeys(int(item.strip()) for item in explicit.split(",")))
        except ValueError as exc:
            raise WorkerUnavailable(
                "BEHAVIOR_RTABMAP_CPU_LIST must be a comma-separated CPU list"
            ) from exc
        if not selected or any(cpu < 0 for cpu in selected):
            raise WorkerUnavailable("BEHAVIOR_RTABMAP_CPU_LIST is empty or invalid")
        unavailable = tuple(cpu for cpu in selected if cpu not in allowed)
        if unavailable:
            raise WorkerUnavailable(
                "requested RTAB-Map CPUs are outside this process affinity: "
                + ",".join(str(cpu) for cpu in unavailable)
            )
        return selected

    raw_count = os.environ.get("BEHAVIOR_RTABMAP_CPU_COUNT", "2").strip()
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise WorkerUnavailable("BEHAVIOR_RTABMAP_CPU_COUNT must be an integer") from exc
    if count <= 0:
        raise WorkerUnavailable("BEHAVIOR_RTABMAP_CPU_COUNT must be positive")
    return allowed[: min(count, len(allowed))]


def default_worker_path() -> Path:
    configured = os.environ.get("BEHAVIOR_RTABMAP_WORKER", "").strip()
    if configured:
        return Path(configured).expanduser()

    # Production releases live outside the source tree so their RTAB-Map and
    # dependency RPATHs remain valid across /tmp cleanup. Once a release link
    # has been created, even a broken link must fail closed instead of silently
    # falling back to an older development binary under native/build.
    release_root = (
        Path(__file__).resolve().parents[2]
        / "work"
        / "rtabmap_native"
        / "current"
    )
    if release_root.exists() or release_root.is_symlink():
        return release_root / "bin" / "behavior_rtabmap_worker"
    return Path(__file__).resolve().parent / "native" / "build" / "behavior_rtabmap_worker"


def default_python_detector_path() -> Path:
    return Path(__file__).resolve().parent / "kornia_sift_detector.py"


def default_python_matcher_path() -> Path:
    return Path(__file__).resolve().parent / "kornia_sift_matcher.py"


class RtabmapClient:
    """Own one isolated RTAB-Map process and its compliant odometry hint."""

    def __init__(
        self,
        worker_path: Optional[os.PathLike[str] | str] = None,
        *,
        database_path: Optional[os.PathLike[str] | str] = None,
        log_path: Optional[os.PathLike[str] | str] = None,
        profile: str = "official",
        feature_backend: Optional[str] = None,
        cuda_device: Optional[str | int] = None,
        request_timeout_s: Optional[float] = None,
    ) -> None:
        self.worker_path = Path(worker_path) if worker_path is not None else default_worker_path()
        self.database_path = Path(database_path) if database_path else None
        self.log_path = Path(log_path) if log_path else None
        if profile not in {
            "official",
            "sparse-rgbd",
            "sparse-icp",
            "native-robust",
            "native-robust-ceres",
        }:
            raise ValueError(
                "profile must be 'official', 'sparse-rgbd', 'sparse-icp', "
                "'native-robust', or 'native-robust-ceres'"
            )
        self.profile = profile
        selected_backend = (
            feature_backend
            if feature_backend is not None
            else os.environ.get("BEHAVIOR_RTABMAP_FEATURE_BACKEND", "cpu")
        )
        self.feature_backend = str(selected_backend).strip().lower()
        if self.feature_backend not in {"cpu", "kornia-sift"}:
            raise ValueError("feature_backend must be 'cpu' or 'kornia-sift'")
        selected_device = (
            cuda_device
            if cuda_device is not None
            else os.environ.get("BEHAVIOR_RTABMAP_CUDA_DEVICE", "")
        )
        self.cuda_device = str(selected_device).strip()
        self.request_timeout_s = float(
            worker_request_timeout_s()
            if request_timeout_s is None
            else request_timeout_s
        )
        if not 0.0 < self.request_timeout_s <= 3600.0:
            raise ValueError("request_timeout_s must be in (0, 3600]")
        self.odometry = BodyOdometry()
        self._structural_gate = StructuralObservationGate()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._log_handle = None
        self._next_frame_id = 1
        self._lock = threading.RLock()
        self._place_retriever = None
        self._place_retrieval_initialized = False
        self._query_integrity_worker_failed = False
        self._closed = False

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def place_retrieval_statistics(self) -> dict[str, object]:
        if self._place_retriever is None:
            return {
                "enabled": False,
                "queries": 0,
                "representatives": 0,
                "proposals": 0,
                "accepted_proposals": 0,
                "rejected_proposals": 0,
            }
        return self._place_retriever.statistics()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise WorkerUnavailable(
                    "RTAB-Map client is closed; start a new client session"
                )
            if self._query_integrity_worker_failed:
                raise WorkerUnavailable(
                    "RTAB-Map query transaction failed after its mutation boundary; "
                    "start a new client session"
                )
            if self.running:
                return
            if self._process is not None:
                exit_code = self._process.poll()
                if exit_code == 3 or self._next_frame_id > 1:
                    self._query_integrity_worker_failed = True
                    self._stop_process()
                    raise WorkerUnavailable(
                        "RTAB-Map worker state cannot be reconstructed from frame deltas; "
                        "start a new client session"
                    )
                # Before the first accepted frame there is no native odometry
                # or graph state to reconstruct, so a failed startup may be
                # retried after reaping its process resources.
                self._stop_process()
            path = self.worker_path.resolve()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise WorkerUnavailable(
                    f"RTAB-Map worker is not executable at {path}; run native/build_worker.sh"
                )
            command = [str(path)]
            command.extend(("--profile", self.profile))
            if self.feature_backend == "kornia-sift":
                detector_path = default_python_detector_path().resolve()
                matcher_path = default_python_matcher_path().resolve()
                if not detector_path.is_file():
                    raise WorkerUnavailable(
                        f"Kornia SIFT detector is missing at {detector_path}"
                    )
                if not matcher_path.is_file():
                    raise WorkerUnavailable(
                        f"Kornia SIFT matcher is missing at {matcher_path}"
                    )
                command.extend(
                    (
                        "--feature-backend",
                        "kornia-sift",
                        "--python-detector",
                        str(detector_path),
                        "--python-matcher",
                        str(matcher_path),
                    )
                )
            if self.database_path is not None:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                command.extend(("--database", str(self.database_path.resolve())))
            stderr = subprocess.DEVNULL
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = self.log_path.open("ab", buffering=0)
                stderr = self._log_handle
            # 线程环境变量约束不了 PCL/VTK 自己的线程池；从 exec 起点硬绑 CPU，
            # 既限制真实占用，也减少并行归约在匹配阈值附近造成的重放抖动。
            affinity = worker_cpu_affinity()
            if affinity:
                taskset = shutil.which("taskset")
                if taskset is None:
                    raise WorkerUnavailable(
                        "taskset is required to enforce RTAB-Map worker CPU affinity"
                    )
                command = [
                    taskset,
                    "--cpu-list",
                    ",".join(str(cpu) for cpu in affinity),
                    *command,
                ]
            worker_environment = os.environ.copy()
            worker_environment.update(
                {
                    # PyTorch/cuBLAS read these before the first CUDA context is
                    # created.  Keep identical RGB-D replays bit-reproducible;
                    # the RTAB worker still uses only the selected GPU.
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "OMP_DYNAMIC": "FALSE",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            if self.feature_backend == "kornia-sift":
                # The worker is process-isolated. PyTorch CUDA tensors are not
                # safe to destroy through Py_FinalizeEx from RTAB-Map's static
                # singleton; let process exit release that context atomically.
                worker_environment["RTABMAP_PYTHON_SKIP_FINALIZE"] = "1"
                if self.cuda_device:
                    worker_environment["CUDA_VISIBLE_DEVICES"] = self.cuda_device
                python_home = os.environ.get(
                    "BEHAVIOR_RTABMAP_PYTHONHOME",
                    sys.prefix,
                ).strip()
                if python_home:
                    worker_environment["PYTHONHOME"] = python_home
                    site_packages = str(
                        Path(python_home) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
                    )
                    existing_python_path = worker_environment.get("PYTHONPATH", "")
                    worker_environment["PYTHONPATH"] = os.pathsep.join(
                        item for item in (site_packages, existing_python_path) if item
                    )
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    bufsize=0,
                    close_fds=True,
                    env=worker_environment,
                )
                self._write(command_packet(Command.PING))
                result = self._read_result("PING")
                if result.frame_id != 0:
                    raise WorkerUnavailable("RTAB-Map worker returned an invalid PING response")
                if result.optimizer_backend != "ceres":
                    raise WorkerUnavailable(
                        "RTAB-Map worker did not load the required Ceres optimizer "
                        f"(reported {result.optimizer_backend})"
                    )
            except Exception:
                self._stop_process()
                raise

    def _ensure_place_retriever(self) -> None:
        if self._place_retrieval_initialized:
            return
        self._place_retrieval_initialized = True
        retrieval_enabled = os.environ.get(
            "BEHAVIOR_RTABMAP_TEMPORAL_RETRIEVAL", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        if self.feature_backend != "kornia-sift" or not retrieval_enabled:
            return
        from .temporal_place import TemporalPlaceRetriever

        self._place_retriever = TemporalPlaceRetriever(
            cuda_device=self.cuda_device
        )

    def advance_odometry(self, base_qvel: Sequence[float], dt_s: float) -> None:
        """Integrate actual evaluator ``base_qvel`` in the robot body frame."""

        self.odometry.advance(base_qvel, dt_s)

    def submit(self, observation: OfficialObservation) -> MapResult:
        with self._lock:
            self.start()
            self._ensure_place_retriever()
            delta, current_odom = self.odometry.frame_delta()
            frame_id = self._next_frame_id
            structural_usable = self._structural_gate.update(observation)
            prepared_place = None
            external_hypothesis = None
            if self._place_retriever is not None:
                prepared_place = self._place_retriever.prepare(
                    observation,
                    current_odom,
                    frame_id,
                    structural_usable=structural_usable,
                )
                external_hypothesis = self._place_retriever.propose(
                    prepared_place
                )
            recovery_global_no_mode_generation = 0
            if self._place_retriever is not None and external_hypothesis is None:
                consume_recovery_no_mode = getattr(
                    self._place_retriever,
                    "consume_recovery_global_no_mode",
                    None,
                )
                if consume_recovery_no_mode is not None:
                    recovery_global_no_mode_generation = int(
                        consume_recovery_no_mode()
                    )
            normal_global_no_mode = bool(
                self._place_retriever is not None
                and external_hypothesis is None
                and recovery_global_no_mode_generation <= 0
                and self._place_retriever.consume_normal_global_no_mode()
            )
            query_generation = int(
                getattr(prepared_place, "query_generation", 0)
                if prepared_place is not None
                else 0
            )
            recovery_global_no_mode = bool(
                recovery_global_no_mode_generation > 0
                and recovery_global_no_mode_generation == query_generation
            )
            try:
                self._write(
                    frame_packet(
                        observation,
                        delta,
                        frame_id,
                        external_loop_hypothesis=external_hypothesis,
                        recovery_hold=bool(
                            prepared_place is not None
                            and prepared_place.recovery_hold
                        ),
                        structural_usable=structural_usable,
                        normal_global_no_mode=normal_global_no_mode,
                        normal_global_search_pending=bool(
                            prepared_place is not None
                            and prepared_place.normal_global_hold
                        ),
                        recovery_global_no_mode=recovery_global_no_mode,
                        query_generation=query_generation,
                    )
                )
                result = self._read_result(f"FRAME {frame_id}")
                if result.frame_id != frame_id:
                    raise WorkerUnavailable(
                        f"worker response frame {result.frame_id} does not match request {frame_id}"
                    )
            except Exception as exc:
                exit_code = (
                    self._process.poll() if self._process is not None else None
                )
                # Once the native worker has accepted any earlier frame, a new
                # process cannot recover its fused pose, raw-qvel history or an
                # active query from the next incremental delta alone. Likewise
                # a query request may have crossed its RAM mutation boundary
                # before a pipe/response failure even when poll() races and
                # still reports None. Fail this client session explicitly.
                if exit_code == 3 or frame_id >= 1 or query_generation > 0:
                    self._query_integrity_worker_failed = True
                    self._stop_process()
                    if isinstance(exc, WorkerRequestTimeout):
                        raise WorkerRequestTimeout(
                            f"{exc}; frame {frame_id} may have crossed the native "
                            "mutation boundary, so this client cannot restart "
                            "from incremental deltas"
                        ) from exc
                    raise WorkerUnavailable(
                        "RTAB-Map worker state cannot be reconstructed from frame deltas; "
                        "start a new client session"
                    ) from exc
                raise
            # RTAB-Map registers against its last accepted frame. Keep all
            # body motion pending across tracking loss so the next external
            # guess spans the same interval. A short response or dead worker
            # likewise cannot consume motion that the replacement never saw.
            if result.tracking_ok:
                self.odometry.commit_frame(current_odom)
            if self._place_retriever is not None and prepared_place is not None:
                self._place_retriever.finish(
                    prepared_place, result, external_hypothesis
                )
            self._next_frame_id += 1
            return result

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                raise WorkerUnavailable(
                    "RTAB-Map client is closed; start a new client session"
                )
            dead_process_code = (
                self._process.poll() if self._process is not None else None
            )
            if self._query_integrity_worker_failed or (
                dead_process_code is not None
                and (dead_process_code == 3 or self._next_frame_id > 1)
            ):
                self._query_integrity_worker_failed = True
                self._stop_process()
                raise WorkerUnavailable(
                    "RTAB-Map worker state cannot be reconstructed from frame deltas; "
                    "start a new client session"
                )
            if self.running:
                self._write(command_packet(Command.RESET))
                result = self._read_result("RESET")
                if result.frame_id != 0:
                    raise WorkerUnavailable("RTAB-Map worker returned an invalid RESET response")
            # Native RESET can reject an active two-phase query. Do not erase
            # its scope/generation or local odometry until the worker confirms
            # reset, otherwise the next request can never finish that exact
            # transaction.
            self.odometry.reset()
            self._structural_gate.reset()
            self._next_frame_id = 1
            if self._place_retriever is not None:
                self._place_retriever.reset()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            shutdown_sent = False
            shutdown_error: Optional[WorkerUnavailable] = None
            process_exit_code = (
                self._process.poll() if self._process is not None else None
            )
            if process_exit_code is not None and process_exit_code != 0:
                shutdown_error = WorkerUnavailable(
                    "RTAB-Map worker exited before durable shutdown "
                    f"(exit={process_exit_code})"
                )
            if self._process is not None and process_exit_code is None:
                try:
                    self._write(command_packet(Command.SHUTDOWN))
                    shutdown_sent = True
                except (BrokenPipeError, OSError, WorkerUnavailable) as exc:
                    shutdown_error = WorkerUnavailable(
                        "RTAB-Map worker lost connection before durable shutdown"
                    )
                    shutdown_error.__cause__ = exc
            try:
                self._stop_process(graceful_shutdown=shutdown_sent)
            except WorkerUnavailable as exc:
                if shutdown_error is None:
                    shutdown_error = exc
            if shutdown_error is not None:
                raise shutdown_error

    def _stdin(self):
        if self._process is None or self._process.stdin is None:
            raise WorkerUnavailable("RTAB-Map worker stdin is unavailable")
        return self._process.stdin

    def _stdout(self):
        if self._process is None or self._process.stdout is None:
            raise WorkerUnavailable("RTAB-Map worker stdout is unavailable")
        return self._process.stdout

    def _write(self, packet: bytes) -> None:
        stream = self._stdin()
        try:
            stream.write(packet)
            stream.flush()
        except (BrokenPipeError, OSError) as exc:
            code = self._process.poll() if self._process is not None else None
            raise WorkerUnavailable(f"RTAB-Map worker stopped (exit={code})") from exc

    def _read_result(self, operation: str) -> MapResult:
        return read_result(
            _DeadlineReader(
                self._stdout(),
                self.request_timeout_s,
                operation,
            )
        )

    def _stop_process(self, *, graceful_shutdown: bool = False) -> None:
        process = self._process
        self._process = None
        shutdown_error: Optional[WorkerUnavailable] = None
        streams_closed = False

        def close_streams() -> None:
            nonlocal streams_closed
            if streams_closed or process is None:
                return
            streams_closed = True
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

        try:
            if process is not None and graceful_shutdown:
                # kShutdown returns from native main only after the stack-owned
                # SlamWorker has completed RTAB-Map/SQLite destruction. Keep
                # both pipes open while waiting: a zero process exit is the
                # durable shutdown acknowledgement.
                try:
                    exit_code = process.wait(
                        timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_S
                    )
                except subprocess.TimeoutExpired:
                    shutdown_error = WorkerUnavailable(
                        "RTAB-Map worker did not finish durable shutdown within "
                        f"{_GRACEFUL_SHUTDOWN_TIMEOUT_S:.0f}s"
                    )
                except OSError as exc:
                    shutdown_error = WorkerUnavailable(
                        "RTAB-Map worker became unavailable during durable shutdown"
                    )
                    shutdown_error.__cause__ = exc
                else:
                    if exit_code != 0:
                        shutdown_error = WorkerUnavailable(
                            "RTAB-Map worker failed durable shutdown "
                            f"(exit={exit_code})"
                        )
                    close_streams()

            if process is not None and not streams_closed:
                # No SHUTDOWN was delivered, or its finite grace period
                # expired. Closing stdin first lets a responsive worker observe
                # EOF; terminate/kill are reserved for a genuinely lost worker.
                close_streams()
                try:
                    process.wait(timeout=_FORCED_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=_FORCED_STOP_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=_FORCED_STOP_TIMEOUT_S)
        finally:
            close_streams()
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
        if shutdown_error is not None:
            raise shutdown_error

    def __enter__(self) -> "RtabmapClient":
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
