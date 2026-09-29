"""Causal GPU place retrieval for RTAB-Map candidate generation.

This module can nominate a historical RTAB node, but it has no API for adding
graph links, changing poses, or writing occupancy. RTAB-Map's normal RGB-D
registration and graph checks are the sole authority that can accept a
nomination.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
import secrets
from typing import Any, Optional

import numpy as np

from behavior_interface.geometry_localization import (
    GeometryConfig,
    GpuGeometryValidator,
    GridSpec,
    TemporalPoseConfig,
    TemporalPoseHypothesisBank,
    extract_column_scan,
    wrap_deg,
)
from behavior_interface.rgbd_odometry import (
    FeatureKeyframeLocalizer,
    ImageFeatures,
    VisualLocalization,
    _KorniaSiftExtractor,
)

from .global_geometry import GpuGlobalGeometrySeeder
from .global_negative import FrozenWallTarget, GpuGlobalNegativeProver
from .official import OfficialObservation, SE2Pose, structural_observation_usable
from .protocol import ExternalLoopHypothesis, MapResult, QueryOutcome, QueryScope


_QUERY_TRANSACTION_SESSION_NONCE_BITS = 48
_QUERY_TRANSACTION_SESSION_NONCE_MAX = (
    1 << _QUERY_TRANSACTION_SESSION_NONCE_BITS
) - 1
_QUERY_TRANSACTION_COUNTER_BITS = 16
_QUERY_TRANSACTION_COUNTER_MAX = (
    1 << _QUERY_TRANSACTION_COUNTER_BITS
) - 1

# Native keeps the map read-only for a bounded number of ordinary mapping
# updates after a novelty query resumes.  Temporal mirrors this state so a
# camera-stability interruption cannot start a new query before the native
# reconciliation has had a chance to finish.


@dataclass(frozen=True)
class TemporalPlaceConfig:
    feature_count: int = 900
    visual_words: int = 128
    descriptors_per_observation: int = 96
    sequence_window: int = 6
    confirmation_queries: int = 3
    confirmation_history: int = 5
    independent_candidate_gap: int = 4
    maximum_candidates: int = 12
    minimum_sequence_score: float = 0.35
    minimum_robust_separation: float = 0.035
    minimum_motion_separation_m: float = 2.0
    rotation_equivalent_radius_m: float = 1.0
    representative_translation_m: float = 0.20
    representative_yaw_rad: float = math.radians(12.0)
    representative_view_yaw_rad: float = math.radians(25.0)
    representative_region_m: float = 0.35
    candidate_region_m: float = 1.5
    rejected_candidate_cooldown: int = 8
    accepted_region_cooldown: int = 30
    maximum_representatives: int = 2048
    minimum_feature_height_m: float = 0.15
    minimum_depth_m: float = 0.45
    maximum_depth_m: float = 3.5
    kmeans_iterations: int = 8
    metric_confirmation_queries: int = 2
    metric_confirmation_history: int = 5
    metric_independent_translation_m: float = 0.25
    metric_independent_yaw_rad: float = math.radians(8.0)
    metric_interquery_translation_slack_m: float = 0.30
    metric_interquery_yaw_slack_rad: float = math.radians(12.0)
    metric_translation_sigma_floor_m: float = 0.04
    metric_translation_sigma_ceiling_m: float = 0.15
    metric_yaw_sigma_floor_rad: float = math.radians(2.0)
    metric_yaw_sigma_ceiling_rad: float = math.radians(10.0)
    # 99% chi-square quantile for one planar pose (x, y, yaw).  This is a
    # statistical confidence level, not a scene- or route-specific threshold.
    metric_consistency_chi2: float = 11.344866730144373
    # Recovery is a sensor-observability state, not a task phase. Two rolling
    # depth scales reject one-frame aliases; two disjoint probes then have to
    # agree before a kidnapped pose may move the map-to-odometry transform.
    recovery_geometry_windows: tuple[int, ...] = (16, 32)
    recovery_evidence_gap_observations: int = 8
    recovery_max_candidates: int = 8
    recovery_grid_resolution_m: float = 0.05
    recovery_grid_half_span_m: float = 24.0
    recovery_depth_stride: int = 8
    recovery_column_m: float = 0.08
    # Retained for capture/report compatibility. Distance and observation count
    # are diagnostics only; neither is allowed to prove that a place is novel.
    recovery_release_translation_m: float = 1.75

    @property
    def bootstrap_observations(self) -> int:
        # Enough documents to estimate both a vocabulary and a temporal null
        # distribution. This scales with model capacity instead of a route.
        return max(self.visual_words // 2, self.sequence_window * 8)

    @property
    def recovery_probe_spacing(self) -> int:
        return max(self.recovery_geometry_windows) + int(
            self.recovery_evidence_gap_observations
        )

    @property
    def recovery_release_min_observations(self) -> int:
        """Legacy diagnostic horizon, not a map-write authorization."""

        return 2 * max(self.recovery_geometry_windows)


@dataclass(frozen=True)
class PreparedPlaceObservation:
    frame_id: int
    observation: OfficialObservation
    image_features: Optional[ImageFeatures]
    descriptors: np.ndarray
    path_progress_m: float
    odometry_pose: SE2Pose
    structural_usable: bool
    predicted_map_pose: SE2Pose
    recovery_hold: bool
    normal_global_hold: bool = False
    query_generation: int = 0


@dataclass
class _PlaceRecord:
    node_id: int
    pose: SE2Pose
    path_progress_m: float
    sampled_descriptors: Optional[np.ndarray]


@dataclass(frozen=True)
class PlaceProposal:
    node_id: int
    score: float
    confirmations: int


@dataclass(frozen=True)
class _MetricObservation:
    frame_id: int
    candidate_id: int
    odometry_pose: SE2Pose
    map_pose: SE2Pose
    hypothesis: ExternalLoopHypothesis
    inliers: int
    inlier_ratio: float
    rmse_m: float
    registration_translation_std_m: float = math.inf
    registration_yaw_std_rad: float = math.inf


def _sequence_endpoint_scores(similarity: Any, window: int) -> Any:
    """Batch SeqSLAM endpoint scores for the supported directions/speeds."""

    torch = __import__("torch")
    count = int(similarity.shape[0])
    endpoint_scores = torch.full(
        (count,), -1.0, device=similarity.device, dtype=similarity.dtype
    )
    endpoints = torch.arange(count, device=similarity.device)
    with torch.inference_mode():
        # ``speed=0`` is the stationary sequence model. It is essential when
        # contact blocks a commanded turn: all query views may correspond to
        # one historical keyframe even though qvel predicted motion. The
        # non-zero models retain normal forward/reverse traversal matching.
        for direction in (-1, 1):
            for speed in (0.0, 0.5, 1.0, 1.5, 2.0):
                scores = torch.zeros_like(endpoint_scores)
                valid = torch.ones(
                    count, device=similarity.device, dtype=torch.bool
                )
                for offset in range(window):
                    indices = endpoints - direction * int(round(offset * speed))
                    valid &= (indices >= 0) & (indices < count)
                    scores += similarity[
                        indices.clamp(0, count - 1), window - 1 - offset
                    ]
                scores /= float(window)
                endpoint_scores = torch.maximum(
                    endpoint_scores, torch.where(valid, scores, -1.0)
                )
    return endpoint_scores


def _wrap_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _inverse_pose(pose: SE2Pose) -> SE2Pose:
    """Return the inverse of a planar rigid transform."""

    c = math.cos(float(pose.yaw_rad))
    s = math.sin(float(pose.yaw_rad))
    return SE2Pose(
        -c * float(pose.x_m) - s * float(pose.y_m),
        s * float(pose.x_m) - c * float(pose.y_m),
        _wrap_rad(-float(pose.yaw_rad)),
    )


def _map_to_odometry_correction(
    map_pose: SE2Pose, odometry_pose: SE2Pose
) -> SE2Pose:
    """Estimate the rigid map<-odometry transform at one RGB-D query."""

    return map_pose.compose(_inverse_pose(odometry_pose))


def _motion_progress_increment(
    previous: SE2Pose,
    current: SE2Pose,
    rotation_equivalent_radius_m: float,
) -> float:
    return math.hypot(
        current.x_m - previous.x_m, current.y_m - previous.y_m
    ) + float(rotation_equivalent_radius_m) * abs(
        _wrap_rad(current.yaw_rad - previous.yaw_rad)
    )


def representative_is_novel(
    pose: SE2Pose,
    records: list[_PlaceRecord],
    config: TemporalPlaceConfig,
) -> bool:
    """Select spatial/view representatives without frame or scene labels."""

    if not records:
        return True
    previous = records[-1].pose
    if math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m) >= (
        config.representative_translation_m
    ):
        return True
    if abs(_wrap_rad(pose.yaw_rad - previous.yaw_rad)) >= config.representative_yaw_rad:
        return True
    nearest = min(
        records,
        key=lambda record: math.hypot(
            pose.x_m - record.pose.x_m, pose.y_m - record.pose.y_m
        ),
    )
    distance = math.hypot(
        pose.x_m - nearest.pose.x_m, pose.y_m - nearest.pose.y_m
    )
    return distance >= config.representative_region_m or abs(
        _wrap_rad(pose.yaw_rad - nearest.pose.yaw_rad)
    ) >= config.representative_view_yaw_rad


def _resolve_parent_cuda_device(requested: str) -> str:
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]

    # Configuration uses physical GPU identifiers, while CUDA remaps a
    # process-local device index whenever CUDA_VISIBLE_DEVICES is set.  The
    # interface intentionally restricts itself to one worker card (for
    # example physical GPU 3), so ``cuda:3`` would be invalid in that process:
    # the only visible device is local ``cuda:0``.  Resolve both the explicit
    # retrieval override and the client-wide device through the same mapping.
    override = os.environ.get("BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE", "").strip()
    token = override or str(requested).strip()
    if token.startswith("cuda:"):
        token = token[5:].strip()
    if token and visible and token in visible:
        return f"cuda:{visible.index(token)}"
    # 可见列表里没有这张物理卡时，不能把物理编号直接当成本地 cuda:N。
    # 例如 interface 只有 CUDA_VISIBLE_DEVICES=2，却把检索设成 3，
    # 会触发 invalid device ordinal，LiveMapper 一次失败就整局禁用。
    if token and visible:
        return "cuda:0"
    if override:
        return f"cuda:{token}" if token else "cuda:0"
    if token and not visible and token.isdigit():
        return f"cuda:{token}"
    return "cuda:0"


class TemporalPlaceRetriever:
    """Online structural SIFT sequence retrieval with an immutable frozen index."""

    def __init__(
        self,
        *,
        cuda_device: str = "",
        config: TemporalPlaceConfig = TemporalPlaceConfig(),
        allow_cpu_for_tests: bool = False,
        extractor: Any = None,
        transaction_session_nonce: Optional[int] = None,
    ) -> None:
        import cv2
        import torch

        self.config = config
        self._torch = torch
        self._functional = torch.nn.functional
        self.device = torch.device(_resolve_parent_cuda_device(cuda_device))
        if self.device.type != "cuda" and not allow_cpu_for_tests:
            raise RuntimeError("production temporal place retrieval is CUDA-only")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("temporal place retrieval requested, but CUDA is unavailable")
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if self.device.type == "cuda":
            torch.empty(1, device=self.device)
        self._extractor = extractor or _KorniaSiftExtractor(
            config.feature_count, device=str(self.device)
        )
        self._metric_localizer = FeatureKeyframeLocalizer(
            nfeatures=config.feature_count,
            max_keyframes=2,
            max_place_keyframes=config.maximum_representatives,
            feature_extractor=self._extractor,
        )
        self._initialize_query_transactions(transaction_session_nonce)
        self.reset()

    def _initialize_query_transactions(
        self, session_nonce: Optional[int] = None
    ) -> None:
        if session_nonce is None:
            while not session_nonce:
                session_nonce = secrets.randbits(
                    _QUERY_TRANSACTION_SESSION_NONCE_BITS
                )
        if isinstance(session_nonce, bool) or not isinstance(
            session_nonce, int
        ):
            raise TypeError("transaction session nonce must be an integer")
        if not 1 <= session_nonce <= _QUERY_TRANSACTION_SESSION_NONCE_MAX:
            raise ValueError(
                "transaction session nonce must be a nonzero 48-bit integer"
            )
        self._transaction_session_nonce = int(session_nonce)
        self._transaction_session_nonces = {int(session_nonce)}
        self._transaction_counter = 0

    def _next_query_transaction_id(self) -> int:
        counter = int(self._transaction_counter)
        if counter >= _QUERY_TRANSACTION_COUNTER_MAX:
            session_nonce = self._transaction_session_nonce
            while (
                not session_nonce
                or session_nonce in self._transaction_session_nonces
            ):
                session_nonce = secrets.randbits(
                    _QUERY_TRANSACTION_SESSION_NONCE_BITS
                )
            self._transaction_session_nonce = session_nonce
            self._transaction_session_nonces.add(session_nonce)
            counter = 0
        counter += 1
        self._transaction_counter = counter
        return (
            int(self._transaction_session_nonce)
            << _QUERY_TRANSACTION_COUNTER_BITS
        ) | counter

    def reset(self) -> None:
        self._records: list[_PlaceRecord] = []
        self._histograms: list[Any] = []
        self._query_histograms: deque[Any] = deque(
            maxlen=self.config.sequence_window
        )
        self._recent_rankings: deque[list[int]] = deque(
            maxlen=self.config.confirmation_history
        )
        self._centers: Any = None
        self._last_odometry_pose: Optional[SE2Pose] = None
        self._path_progress_m = 0.0
        self._next_refit_size = self.config.bootstrap_observations
        self._frozen = False
        self._cooldowns: dict[int, int] = {}
        self._pending_candidate_id = 0
        self._metric_history: deque[_MetricObservation] = deque(
            maxlen=self.config.metric_confirmation_history
        )
        self._recent_metric_events: deque[dict[str, object]] = deque(maxlen=16)
        self._recent_recovery_events: deque[dict[str, object]] = deque(maxlen=24)
        self._last_structural_usable: Optional[bool] = None
        self._last_result_odometry_pose: Optional[SE2Pose] = None
        self._last_result_map_pose: Optional[SE2Pose] = None
        self._recovery_pending = False
        self._recovery_usable_observations = 0
        self._recovery_next_geometry_probe = 0
        self._recovery_last_candidate_frame = -10**9
        self._recovery_last_fused_pose: Optional[SE2Pose] = None
        self._recovery_anchor_fused_pose: Optional[SE2Pose] = None
        self._recovery_anchor_map_pose: Optional[SE2Pose] = None
        self._recovery_last_local_map_pose: Optional[SE2Pose] = None
        self._recovery_last_prepared_odometry_pose: Optional[SE2Pose] = None
        self._recovery_observed_translation_m = 0.0
        self._recovery_retry_proposals: list[PlaceProposal] = []
        self._recovery_retry_attempts_remaining = 0
        self._recovery_geometry_validator: Optional[GpuGeometryValidator] = None
        self._recovery_global_seeder: Optional[GpuGlobalGeometrySeeder] = None
        self._global_negative_prover: Optional[GpuGlobalNegativeProver] = None
        self._recovery_pose_bank: Optional[TemporalPoseHypothesisBank] = None
        self._recovery_target_wall = np.empty((0, 2), dtype=np.float32)
        self._recovery_target_free = np.empty((0, 2), dtype=np.float32)
        self._recovery_negative_target: Optional[FrozenWallTarget] = None
        self._recovery_negative_intervals: list[tuple[int, int]] = []
        self._recovery_negative_evidence_binding: Optional[
            tuple[str, str]
        ] = None
        self._recovery_negative_blocked_frame = -1
        self._recovery_negative_blocked_reason = ""
        self._recovery_allowed_node_ids: frozenset[int] = frozenset()
        self._recovery_resolver_warm = False
        self._recovery_resolver_force_probe = False
        self._recovery_resolver_last_probe_frame = -1
        self._recovery_resolver_last_probe_odometry_pose: Optional[SE2Pose] = None
        self._recovery_resolver_seed_summary: dict[str, object] = {}
        self._recovery_resolver_confirmation_burst_remaining = 0
        self._recovery_resolver_confirmation_budget_remaining = 0
        self._recovery_hold_generation = 0
        self._recovery_global_no_mode_generation = 0
        self._recovery_place_database_generation = 0
        self._normal_global_pending = False
        self._verified_read_only_continuation = False
        self._native_novelty_resume_reconciliation_generation = 0
        self._native_novelty_resume_reconciliation_updates_remaining = 0
        self._normal_global_usable_observations = 0
        self._normal_global_next_probe = 0
        self._normal_global_no_mode_ready = False
        self._normal_global_no_mode_intervals: list[tuple[int, int]] = []
        self._normal_negative_evidence_binding: Optional[
            tuple[str, str]
        ] = None
        self._normal_negative_blocked_frame = -1
        self._normal_negative_blocked_reason = ""
        self._normal_negative_ready_observation = -1
        self._normal_negative_ready_odometry_pose: Optional[SE2Pose] = None
        self._normal_resolver_warm = False
        self._normal_resolver_force_probe = False
        self._normal_resolver_last_probe_frame = -1
        self._normal_resolver_last_probe_odometry_pose: Optional[SE2Pose] = None
        self._normal_resolver_seed_summary: dict[str, object] = {}
        self._normal_resolver_confirmation_burst_remaining = 0
        self._normal_resolver_confirmation_budget_remaining = 0
        self._normal_global_validator: Optional[GpuGeometryValidator] = None
        self._normal_global_seeder: Optional[GpuGlobalGeometrySeeder] = None
        self._normal_global_pose_bank: Optional[TemporalPoseHypothesisBank] = None
        self._normal_metric_history: deque[_MetricObservation] = deque(maxlen=8)
        self._normal_retry_proposals: list[PlaceProposal] = []
        self._normal_retry_attempts_remaining = 0
        self._recovery_node_metric_history: deque[_MetricObservation] = deque(maxlen=8)
        self._normal_target_wall = np.empty((0, 2), dtype=np.float32)
        self._normal_target_free = np.empty((0, 2), dtype=np.float32)
        self._normal_negative_target: Optional[FrozenWallTarget] = None
        self._normal_allowed_node_ids: frozenset[int] = frozenset()
        self._normal_hold_generation = 0
        self._normal_place_database_generation = 0
        self._recent_negative_events: deque[dict[str, object]] = deque(maxlen=24)
        self._metric_localizer.reset()
        self.queries = 0
        self.proposals = 0
        self.accepted_proposals = 0
        self.rejected_proposals = 0
        self.geometry_attempts = 0
        self.geometry_accepted = 0
        self.metric_consensus_accepted = 0
        self.structural_observations_skipped = 0
        self.structural_interruptions = 0
        self.structural_recoveries = 0
        self.forced_index_builds = 0
        self.recovery_holds = 0
        self.recovery_geometry_attempts = 0
        self.recovery_global_attempts = 0
        self.recovery_global_candidates = 0
        self.recovery_geometry_consensus = 0
        self.recovery_localizations = 0
        self.recovery_native_localizations = 0
        self.recovery_novel_releases = 0
        self.recovery_ambiguous_probes = 0
        self.recovery_negative_attempts = 0
        self.recovery_negative_certificates = 0
        self.recovery_negative_unknowns = 0
        self.recovery_negative_resets = 0
        self.normal_global_holds = 0
        self.normal_global_attempts = 0
        self.normal_global_candidates = 0
        self.normal_global_bridges = 0
        self.normal_global_native_novelty_resumes = 0
        self.normal_global_no_modes = 0
        self.normal_global_ambiguous_probes = 0
        self.normal_negative_attempts = 0
        self.normal_negative_certificates = 0
        self.normal_negative_unknowns = 0
        self.normal_negative_resets = 0
        self.last_geometry_reason = "not_attempted"

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def representative_count(self) -> int:
        return len(self._records)

    def statistics(self) -> dict[str, object]:
        return {
            "enabled": True,
            "device": str(self.device),
            "frozen": self._frozen,
            "queries": self.queries,
            "representatives": len(self._records),
            "proposals": self.proposals,
            "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejected_proposals,
            "geometry_attempts": self.geometry_attempts,
            "geometry_accepted": self.geometry_accepted,
            "metric_consensus_accepted": self.metric_consensus_accepted,
            "appearance_index_ready": self._centers is not None,
            "structural_observations_skipped": self.structural_observations_skipped,
            "structural_interruptions": self.structural_interruptions,
            "structural_recoveries": self.structural_recoveries,
            "forced_index_builds": self.forced_index_builds,
            "recovery_pending": self._recovery_pending,
            "recovery_holds": self.recovery_holds,
            "recovery_usable_observations": self._recovery_usable_observations,
            "recovery_observed_translation_m": (
                self._recovery_observed_translation_m
            ),
            "recovery_geometry_attempts": self.recovery_geometry_attempts,
            "recovery_global_attempts": self.recovery_global_attempts,
            "recovery_global_candidates": self.recovery_global_candidates,
            "recovery_geometry_consensus": self.recovery_geometry_consensus,
            "recovery_localizations": self.recovery_localizations,
            "recovery_native_localizations": (
                self.recovery_native_localizations
            ),
            "recovery_novel_releases": self.recovery_novel_releases,
            "recovery_ambiguous_probes": self.recovery_ambiguous_probes,
            "recovery_negative_attempts": self.recovery_negative_attempts,
            "recovery_negative_certificates": (
                self.recovery_negative_certificates
            ),
            "recovery_negative_unknowns": self.recovery_negative_unknowns,
            "recovery_negative_resets": self.recovery_negative_resets,
            "recovery_negative_evidence_count": len(
                self._recovery_negative_intervals
            ),
            "recovery_negative_target_digest": (
                "" if self._recovery_negative_target is None
                else self._recovery_negative_target.digest_sha256
            ),
            "recovery_negative_target_generation": (
                "" if self._recovery_negative_target is None
                else self._recovery_negative_target.generation
            ),
            "normal_global_pending": self._normal_global_pending,
            "verified_read_only_continuation": (
                self._verified_read_only_continuation
            ),
            "normal_global_holds": self.normal_global_holds,
            "normal_global_attempts": self.normal_global_attempts,
            "normal_global_candidates": self.normal_global_candidates,
            "normal_global_bridges": self.normal_global_bridges,
            "normal_global_native_novelty_resumes": (
                self.normal_global_native_novelty_resumes
            ),
            "native_novelty_resume_reconciliation_generation": int(
                getattr(
                    self,
                    "_native_novelty_resume_reconciliation_generation",
                    0,
                )
            ),
            "native_novelty_resume_reconciliation_updates_remaining": int(
                getattr(
                    self,
                    "_native_novelty_resume_reconciliation_updates_remaining",
                    0,
                )
            ),
            "native_novelty_resume_reconciliation_pending": bool(
                self._native_novelty_reconciliation_pending()
            ),
            "normal_global_no_modes": self.normal_global_no_modes,
            "normal_global_ambiguous_probes": (
                self.normal_global_ambiguous_probes
            ),
            "normal_negative_attempts": self.normal_negative_attempts,
            "normal_negative_certificates": self.normal_negative_certificates,
            "normal_negative_unknowns": self.normal_negative_unknowns,
            "normal_negative_resets": self.normal_negative_resets,
            "normal_negative_evidence_count": len(
                self._normal_global_no_mode_intervals
            ),
            "normal_negative_target_digest": (
                "" if self._normal_negative_target is None
                else self._normal_negative_target.digest_sha256
            ),
            "normal_negative_target_generation": (
                "" if self._normal_negative_target is None
                else self._normal_negative_target.generation
            ),
            "normal_negative_level_ready": bool(
                self._normal_global_no_mode_ready
            ),
            "normal_negative_ready_observation": int(
                self._normal_negative_ready_observation
            ),
            "last_geometry_reason": self.last_geometry_reason,
            "last_geometry_rows": list(
                self._metric_localizer.last_place_geometry_rows
            ),
            "recent_metric_events": list(self._recent_metric_events),
            "recent_recovery_events": list(self._recent_recovery_events),
            "recent_negative_events": list(self._recent_negative_events),
            "forbidden_inputs_consumed": [],
        }

    def prepare(
        self,
        observation: OfficialObservation,
        odometry_pose: SE2Pose,
        frame_id: int,
        *,
        structural_usable: Optional[bool] = None,
    ) -> PreparedPlaceObservation:
        if self._last_odometry_pose is not None:
            self._path_progress_m += _motion_progress_increment(
                self._last_odometry_pose,
                odometry_pose,
                self.config.rotation_equivalent_radius_m,
            )
        self._last_odometry_pose = odometry_pose
        predicted_map_pose = self._predict_map_pose(odometry_pose)
        # Production passes the client's stateful camera-stability verdict so
        # mapping and feature retrieval consume exactly the same observations.
        # The fallback preserves the standalone policy API used by tests and
        # offline callers that do not own a StructuralObservationGate.
        usable = (
            structural_observation_usable(observation)
            if structural_usable is None
            else bool(structural_usable)
        )
        if not usable:
            self.structural_observations_skipped += 1
            if self._last_structural_usable is not False:
                self.structural_interruptions += 1
                self._reset_query_sequence()
                self._clear_recovery_geometry_evidence(
                    "recovery_structural_segment_interrupted"
                )
                history = getattr(self, "_normal_metric_history", None)
                if history is not None:
                    history.clear()
                self._reset_frozen_resolver("normal")
                self._normal_retry_proposals = []
                self._normal_retry_attempts_remaining = 0
                if bool(getattr(self, "_normal_global_pending", False)):
                    self._normal_global_usable_observations = 0
                    self._normal_global_next_probe = max(
                        self.config.recovery_geometry_windows
                    )
                    self._clear_negative_evidence(
                        "normal", "normal_structural_segment_interrupted"
                    )
                    validator = getattr(
                        self, "_normal_global_validator", None
                    )
                    if validator is not None:
                        validator.clear()
                    bank = getattr(self, "_normal_global_pose_bank", None)
                    if bank is not None:
                        bank.clear()
            self._last_structural_usable = False
            self.last_geometry_reason = "structural_observation_unusable"
            return PreparedPlaceObservation(
                int(frame_id),
                observation,
                None,
                np.zeros((0, 128), dtype=np.float32),
                self._path_progress_m,
                odometry_pose,
                False,
                predicted_map_pose,
                False,
                bool(getattr(self, "_normal_global_pending", False)),
                int(getattr(self, "_normal_hold_generation", 0))
                if bool(getattr(self, "_normal_global_pending", False))
                else 0,
            )
        if self._last_structural_usable is False:
            self.structural_recoveries += 1
            # A normal soft/uncertain hold already owns an immutable target and
            # only restarts its query window after camera instability. Starting
            # a second recovery episode here would discard that hold identity.
            if (
                not bool(getattr(self, "_normal_global_pending", False))
                and not self._native_novelty_reconciliation_pending()
                and self._can_begin_recovery()
            ):
                if bool(getattr(self, "_recovery_pending", False)):
                    self._resume_recovery_segment(
                        predicted_map_pose, odometry_pose
                    )
                else:
                    self._begin_recovery(predicted_map_pose, odometry_pose)
                predicted_map_pose = self._predict_map_pose(odometry_pose)
        self._last_structural_usable = True
        image_features, descriptors = self._structural_descriptors(observation)
        recovery_hold = bool(getattr(self, "_recovery_pending", False))
        if recovery_hold:
            place_generation = int(getattr(
                self._metric_localizer, "place_database_generation", 0
            ))
            if place_generation != int(
                getattr(self, "_recovery_place_database_generation", 0)
            ):
                self._recovery_place_database_generation = place_generation
                self._recovery_usable_observations = 0
                self._recovery_next_geometry_probe = max(
                    self.config.recovery_geometry_windows
                )
                self._clear_recovery_geometry_evidence(
                    "recovery_place_database_generation_changed"
                )
                self._freeze_negative_target("recovery")
            self._recovery_usable_observations += 1
            self.recovery_holds += 1
            self._remember_recovery_geometry(
                observation, predicted_map_pose, int(frame_id)
            )
        normal_global_hold = bool(
            getattr(self, "_normal_global_pending", False)
            and not recovery_hold
        )
        if normal_global_hold:
            place_generation = int(
                self._metric_localizer.place_database_generation
            )
            if place_generation != int(self._normal_place_database_generation):
                self._normal_place_database_generation = place_generation
                self._normal_global_usable_observations = 0
                self._normal_global_next_probe = max(
                    self.config.recovery_geometry_windows
                )
                self._clear_negative_evidence(
                    "normal", "normal_place_database_generation_changed"
                )
                validator = getattr(self, "_normal_global_validator", None)
                if validator is not None:
                    validator.clear()
                bank = getattr(self, "_normal_global_pose_bank", None)
                if bank is not None:
                    bank.clear()
                self._normal_metric_history.clear()
                self._reset_frozen_resolver("normal")
                self._normal_retry_proposals = []
                self._normal_retry_attempts_remaining = 0
                # Keep the hold-entry node allowlist immutable. A database
                # generation change can invalidate evidence, but cannot make
                # newly added graph nodes part of an old target snapshot.
                self._freeze_negative_target("normal")
            self._normal_global_usable_observations += 1
            self.normal_global_holds += 1
            self._expire_normal_negative_level(odometry_pose)
            self._remember_normal_global_geometry(
                observation, predicted_map_pose, int(frame_id)
            )
        return PreparedPlaceObservation(
            int(frame_id),
            observation,
            image_features,
            descriptors,
            self._path_progress_m,
            odometry_pose,
            True,
            predicted_map_pose,
            recovery_hold,
            normal_global_hold,
            int(
                getattr(
                    self,
                    "_recovery_hold_generation"
                    if recovery_hold
                    else "_normal_hold_generation",
                    0,
                )
            )
            if recovery_hold or normal_global_hold
            else 0,
        )

    def _predict_map_pose(self, odometry_pose: SE2Pose) -> SE2Pose:
        if bool(getattr(self, "_recovery_pending", False)):
            local_pose = getattr(self, "_recovery_last_local_map_pose", None)
            local_odometry = getattr(
                self, "_recovery_last_prepared_odometry_pose", None
            )
            if local_pose is not None and local_odometry is not None:
                # Recovery owns a disconnected local coordinate chain.  Only
                # the one-frame body-odometry delta predicts the next sample;
                # every completed frame replaces the chain head with native
                # RGB-D/ICP fused odometry in ``_finish_recovery_observation``.
                # This prevents a long interruption's qvel drift from bending
                # the query submap before global relocalization.
                return local_pose.compose(
                    local_odometry.relative_to(odometry_pose)
                )
            anchor = getattr(self, "_recovery_anchor_map_pose", None)
            if anchor is not None:
                return anchor
        anchor_odom = getattr(self, "_last_result_odometry_pose", None)
        anchor_map = getattr(self, "_last_result_map_pose", None)
        if anchor_odom is None or anchor_map is None:
            return odometry_pose
        return anchor_map.compose(anchor_odom.relative_to(odometry_pose))

    def _can_begin_recovery(self) -> bool:
        records = getattr(self, "_records", ())
        target_wall = getattr(
            self, "_recovery_target_wall", np.empty((0, 2))
        )
        return (
            len(records) >= int(self.config.sequence_window)
            and len(target_wall) >= 16
            and getattr(self, "_last_result_map_pose", None) is not None
        )

    def _begin_recovery(
        self,
        predicted_map_pose: SE2Pose,
        odometry_pose: SE2Pose,
    ) -> None:
        self._verified_read_only_continuation = False
        self._recovery_pending = True
        self._recovery_usable_observations = 0
        self._recovery_next_geometry_probe = max(
            self.config.recovery_geometry_windows
        )
        self._recovery_last_candidate_frame = -10**9
        self._recovery_last_fused_pose = None
        self._recovery_anchor_fused_pose = None
        self._recovery_anchor_map_pose = predicted_map_pose
        self._recovery_last_local_map_pose = predicted_map_pose
        self._recovery_last_prepared_odometry_pose = odometry_pose
        self._recovery_observed_translation_m = 0.0
        self._recovery_retry_proposals = []
        self._recovery_retry_attempts_remaining = 0
        self._recovery_allowed_node_ids = self._historical_place_node_ids()
        self._clear_recovery_geometry_evidence("recovery_episode_started")
        self._recovery_hold_generation = self._next_query_transaction_id()
        self._recovery_place_database_generation = int(getattr(
            self._metric_localizer, "place_database_generation", 0
        ))
        self._freeze_negative_target("recovery")
        self.last_geometry_reason = "recovery_waiting_for_place_and_geometry"

    def _resume_recovery_segment(
        self,
        predicted_map_pose: SE2Pose,
        odometry_pose: SE2Pose,
    ) -> None:
        """Start a new continuous query segment without replacing the episode.

        Camera articulation invalidates adjacent-frame odometry and pairwise
        localization evidence, but it does not make the committed target map or
        the interrupted episode new. Keeping those separate prevents a second
        head motion from restarting a route-dependent probation timer.
        """

        self._recovery_next_geometry_probe = (
            int(self._recovery_usable_observations)
            + max(self.config.recovery_geometry_windows)
        )
        self._recovery_last_fused_pose = None
        self._recovery_anchor_fused_pose = None
        self._recovery_anchor_map_pose = predicted_map_pose
        self._recovery_last_local_map_pose = predicted_map_pose
        self._recovery_last_prepared_odometry_pose = odometry_pose
        self._clear_recovery_geometry_evidence(
            "recovery_query_segment_restarted"
        )
        self.last_geometry_reason = "recovery_query_segment_restarted"

    def _clear_recovery_geometry_evidence(
        self, reason: str = "recovery_geometry_evidence_reset"
    ) -> None:
        validator = getattr(self, "_recovery_geometry_validator", None)
        if validator is not None:
            validator.clear()
        bank = getattr(self, "_recovery_pose_bank", None)
        if bank is not None:
            bank.clear()
        history = getattr(self, "_recovery_node_metric_history", None)
        if history is not None:
            history.clear()
        self._reset_frozen_resolver("recovery")
        self._clear_negative_evidence("recovery", str(reason))

    def _recovery_geometry_engine(self) -> GpuGeometryValidator:
        validator = getattr(self, "_recovery_geometry_validator", None)
        if validator is None:
            validator = GpuGeometryValidator(
                GridSpec(
                    float(self.config.recovery_grid_resolution_m),
                    float(self.config.recovery_grid_half_span_m),
                ),
                config=GeometryConfig(
                    windows=tuple(self.config.recovery_geometry_windows)
                ),
                device=str(self.device),
            )
            self._recovery_geometry_validator = validator
        return validator

    def _recovery_temporal_pose_bank(self) -> TemporalPoseHypothesisBank:
        bank = getattr(self, "_recovery_pose_bank", None)
        if bank is None:
            bank = TemporalPoseHypothesisBank(TemporalPoseConfig(
                min_gap_frames=int(
                    max(
                        min(self.config.recovery_geometry_windows),
                        self.config.recovery_evidence_gap_observations,
                    )
                )
            ))
            self._recovery_pose_bank = bank
        return bank

    def _recovery_global_geometry_seeder(self) -> GpuGlobalGeometrySeeder:
        seeder = getattr(self, "_recovery_global_seeder", None)
        if seeder is None:
            seeder = GpuGlobalGeometrySeeder(
                GridSpec(
                    float(self.config.recovery_grid_resolution_m),
                    float(self.config.recovery_grid_half_span_m),
                ),
                device=str(self.device),
            )
            self._recovery_global_seeder = seeder
        return seeder

    def _global_negative_engine(self) -> GpuGlobalNegativeProver:
        """Return the shared exhaustive prover; production has no CPU path."""

        prover = getattr(self, "_global_negative_prover", None)
        if prover is None:
            prover = GpuGlobalNegativeProver(
                GridSpec(
                    float(self.config.recovery_grid_resolution_m),
                    float(self.config.recovery_grid_half_span_m),
                ),
                device=str(self.device),
            )
            self._global_negative_prover = prover
        return prover

    def _negative_generation(self, scope: str) -> str:
        if scope == "normal":
            hold = int(getattr(self, "_normal_hold_generation", 0))
            places = int(
                getattr(self, "_normal_place_database_generation", 0)
            )
        elif scope == "recovery":
            hold = int(getattr(self, "_recovery_hold_generation", 0))
            places = int(
                getattr(self, "_recovery_place_database_generation", 0)
            )
        else:
            raise ValueError(f"unknown negative-proof scope: {scope}")
        return f"{scope}:hold={hold}:places={places}"

    def _negative_target(self, scope: str) -> Optional[FrozenWallTarget]:
        return getattr(self, f"_{scope}_negative_target", None)

    def _freeze_negative_target(self, scope: str) -> bool:
        wall = np.asarray(
            getattr(self, f"_{scope}_target_wall", np.empty((0, 2))),
            dtype=np.float32,
        ).reshape(-1, 2)
        if len(wall) < 16:
            setattr(self, f"_{scope}_negative_target", None)
            return False
        try:
            target = self._global_negative_engine().freeze_target(
                wall, generation=self._negative_generation(scope)
            )
        except Exception as exc:
            setattr(self, f"_{scope}_negative_target", None)
            self._note_negative_event(
                scope,
                event="target_freeze_failed",
                reason=f"target_freeze_failed:{type(exc).__name__}",
            )
            return False
        setattr(self, f"_{scope}_negative_target", target)
        return True

    def _note_negative_event(
        self, scope: str, *, event: str, reason: str, **values: object
    ) -> None:
        events = getattr(self, "_recent_negative_events", None)
        if events is None:
            events = deque(maxlen=24)
            self._recent_negative_events = events
        events.append({
            "scope": str(scope),
            "event": str(event),
            "reason": str(reason),
            **values,
        })

    def _clear_negative_evidence(self, scope: str, reason: str) -> None:
        if scope == "normal":
            intervals = getattr(
                self, "_normal_global_no_mode_intervals", []
            )
            had_evidence = bool(
                intervals
                or getattr(self, "_normal_global_no_mode_ready", False)
            )
            intervals.clear()
            self._normal_global_no_mode_ready = False
            self._normal_negative_ready_observation = -1
            self._normal_negative_ready_odometry_pose = None
            self._normal_negative_evidence_binding = None
            counter_name = "normal_negative_resets"
        elif scope == "recovery":
            intervals = getattr(self, "_recovery_negative_intervals", [])
            had_evidence = bool(intervals)
            intervals.clear()
            self._recovery_negative_evidence_binding = None
            counter_name = "recovery_negative_resets"
        else:
            raise ValueError(f"unknown negative-proof scope: {scope}")
        if had_evidence:
            setattr(
                self,
                counter_name,
                int(getattr(self, counter_name, 0)) + 1,
            )
            self._note_negative_event(
                scope, event="evidence_cleared", reason=str(reason)
            )

    def _block_negative_for_frame(
        self, scope: str, frame_id: int, reason: str
    ) -> None:
        """Make an attempted-but-unknown positive search dominate this frame."""

        self._clear_negative_evidence(scope, str(reason))
        setattr(self, f"_{scope}_negative_blocked_frame", int(frame_id))
        setattr(self, f"_{scope}_negative_blocked_reason", str(reason))

    def _positive_metric_pending(self, scope: str) -> bool:
        if int(getattr(self, "_pending_candidate_id", 0)) > 0:
            return True
        names = (
            ("_normal_metric_history",)
            if scope == "normal"
            else ("_metric_history", "_recovery_node_metric_history")
        )
        return any(bool(getattr(self, name, ())) for name in names)

    def _count_negative_unknown(self, scope: str) -> None:
        name = f"{scope}_negative_unknowns"
        setattr(self, name, int(getattr(self, name, 0)) + 1)

    def _note_negative_interval(
        self, scope: str, interval: tuple[int, int]
    ) -> bool:
        first, last = (int(value) for value in interval)
        if last < first:
            return False
        name = (
            "_normal_global_no_mode_intervals"
            if scope == "normal"
            else "_recovery_negative_intervals"
        )
        intervals = list(getattr(self, name, []))
        gap = int(self.config.recovery_evidence_gap_observations)
        independent = next((
            old for old in intervals
            if (
                old[1] < first and first - old[1] - 1 >= gap
            ) or (
                last < old[0] and old[0] - last - 1 >= gap
            )
        ), None)
        if independent is None:
            setattr(self, name, [(first, last)])
            return False
        setattr(self, name, [independent, (first, last)])
        return True

    def _evaluate_global_negative(
        self,
        scope: str,
        rolling: dict[str, Any],
        predicted: SE2Pose,
        *,
        observation_count: int,
        odometry_pose: SE2Pose,
        query_frame_id: Optional[int] = None,
    ) -> str:
        """Consume one fail-closed certificate without authorizing a write."""

        blocked_frame = int(getattr(
            self, f"_{scope}_negative_blocked_frame", -1
        ))
        if query_frame_id is not None and blocked_frame == int(query_frame_id):
            reason = str(getattr(
                self,
                f"_{scope}_negative_blocked_reason",
                "positive_search_unknown",
            ))
            self._count_negative_unknown(scope)
            self._clear_negative_evidence(scope, reason)
            self._note_negative_event(
                scope,
                event="unknown",
                reason=reason,
                frame_id=int(query_frame_id),
            )
            self.last_geometry_reason = f"{scope}_negative_blocked:{reason}"
            return "UNKNOWN"

        interval_values = (
            rolling.get("first_frame"), rolling.get("last_frame")
        )
        if any(value is None for value in interval_values):
            self._count_negative_unknown(scope)
            self._clear_negative_evidence(scope, "negative_interval_missing")
            self._note_negative_event(
                scope, event="unknown", reason="negative_interval_missing"
            )
            return "UNKNOWN"
        interval = (int(interval_values[0]), int(interval_values[1]))
        if self._positive_metric_pending(scope):
            self._clear_negative_evidence(
                scope, "positive_metric_or_bridge_pending"
            )
            self._note_negative_event(
                scope,
                event="positive_priority",
                reason="positive_metric_or_bridge_pending",
                interval=list(interval),
            )
            return "POSITIVE"

        target = self._negative_target(scope)
        if target is None:
            self._count_negative_unknown(scope)
            self._clear_negative_evidence(scope, "frozen_target_missing")
            self._note_negative_event(
                scope,
                event="unknown",
                reason="frozen_target_missing",
                interval=list(interval),
            )
            return "UNKNOWN"
        expected_generation = self._negative_generation(scope)
        if str(target.generation) != expected_generation:
            self._count_negative_unknown(scope)
            self._clear_negative_evidence(scope, "target_generation_changed")
            self._note_negative_event(
                scope,
                event="unknown",
                reason="target_generation_changed",
                interval=list(interval),
                target_digest=str(target.digest_sha256),
                target_generation=str(target.generation),
                expected_generation=expected_generation,
            )
            return "UNKNOWN"

        attempts_name = f"{scope}_negative_attempts"
        setattr(
            self, attempts_name, int(getattr(self, attempts_name, 0)) + 1
        )
        try:
            certificate = self._global_negative_engine().prove(
                np.asarray(rolling.get("wall"), dtype=np.float32),
                target,
                pivot=(float(predicted.x_m), float(predicted.y_m)),
                source_free=np.asarray(
                    rolling.get("free"), dtype=np.float32
                ),
            )
        except Exception as exc:
            certificate = None
            failure_reason = f"negative_prover_failed:{type(exc).__name__}"
        else:
            failure_reason = str(
                getattr(certificate, "reason", "negative_result_missing")
            )

        status = str(getattr(certificate, "status", "UNKNOWN"))
        complete = bool(getattr(certificate, "search_complete", False))
        truncated = bool(
            getattr(certificate, "candidate_budget_truncated", True)
        )
        same_target = bool(
            certificate is not None
            and str(getattr(certificate, "target_digest_sha256", ""))
            == str(target.digest_sha256)
            and str(getattr(certificate, "target_generation", ""))
            == expected_generation
        )
        valid_negative = bool(
            status == "NEGATIVE"
            and getattr(certificate, "novelty_certified", False)
            and not getattr(certificate, "map_write_authorized", True)
            and getattr(certificate, "target_frozen", False)
            and not getattr(certificate, "cpu_fallback", True)
            and getattr(certificate, "source_2d_observable", False)
            and complete
            and not truncated
            and same_target
        )
        self._note_negative_event(
            scope,
            event="certificate" if valid_negative else "unknown",
            reason=failure_reason,
            status=status,
            interval=list(interval),
            target_digest=str(target.digest_sha256),
            target_generation=expected_generation,
            search_complete=complete,
            candidate_budget_truncated=truncated,
            source_2d_observable=bool(
                getattr(certificate, "source_2d_observable", False)
            ),
            upper_bound_support_fraction=float(
                getattr(certificate, "upper_bound_support_fraction", 1.0)
            ),
            map_write_authorized=bool(
                getattr(certificate, "map_write_authorized", False)
            ),
        )
        if not valid_negative:
            self._count_negative_unknown(scope)
            self._clear_negative_evidence(
                scope, f"negative_unknown:{failure_reason}"
            )
            self.last_geometry_reason = (
                f"{scope}_negative_unknown:{failure_reason}"
            )
            return "UNKNOWN"

        certificates_name = f"{scope}_negative_certificates"
        setattr(
            self,
            certificates_name,
            int(getattr(self, certificates_name, 0)) + 1,
        )
        binding_name = f"_{scope}_negative_evidence_binding"
        binding = (str(target.digest_sha256), expected_generation)
        previous_binding = getattr(self, binding_name, None)
        evidence_name = (
            "_normal_global_no_mode_intervals"
            if scope == "normal"
            else "_recovery_negative_intervals"
        )
        if bool(getattr(self, evidence_name, [])) and previous_binding != binding:
            self._clear_negative_evidence(
                scope, "negative_target_digest_or_generation_changed"
            )
        setattr(self, binding_name, binding)
        if not self._note_negative_interval(scope, interval):
            self.last_geometry_reason = f"{scope}_negative_certificate_1_of_2"
            return "NEGATIVE_PENDING"

        if scope == "normal":
            if not bool(getattr(self, "_normal_global_no_mode_ready", False)):
                self.normal_global_no_modes = int(
                    getattr(self, "normal_global_no_modes", 0)
                ) + 1
            self._normal_global_no_mode_ready = True
            self._normal_negative_ready_observation = int(observation_count)
            self._normal_negative_ready_odometry_pose = odometry_pose
            self.last_geometry_reason = "normal_global_negative_level_ready"
        else:
            self.recovery_novel_releases = int(
                getattr(self, "recovery_novel_releases", 0)
            ) + 1
            # Keep the recovery episode and immutable target alive until the
            # native worker explicitly acknowledges committed/all-known. The
            # client consumes this authorization exactly once for this
            # generation; a rejected aggregate remains quarantined.
            self._recovery_global_no_mode_generation = int(
                getattr(self, "_recovery_hold_generation", 0)
            )
            self.last_geometry_reason = "recovery_negative_novel_pending_native_ack"
        return "READY"

    def consume_recovery_global_no_mode(self) -> int:
        """Consume one generation-bound recovery negative certificate."""

        generation = int(
            getattr(self, "_recovery_global_no_mode_generation", 0)
        )
        self._recovery_global_no_mode_generation = 0
        return generation if generation > 0 else 0

    def _expire_normal_negative_level(self, odometry_pose: SE2Pose) -> None:
        if not bool(getattr(self, "_normal_global_no_mode_ready", False)):
            return
        ready_at = int(
            getattr(self, "_normal_negative_ready_observation", -1)
        )
        ready_pose = getattr(
            self, "_normal_negative_ready_odometry_pose", None
        )
        count = int(getattr(self, "_normal_global_usable_observations", 0))
        ttl = max(1, int(self.config.recovery_evidence_gap_observations))
        if ready_at < 0 or ready_pose is None:
            self._clear_negative_evidence(
                "normal", "negative_level_binding_missing"
            )
            return
        translation = math.hypot(
            float(odometry_pose.x_m) - float(ready_pose.x_m),
            float(odometry_pose.y_m) - float(ready_pose.y_m),
        )
        yaw = abs(_wrap_rad(
            float(odometry_pose.yaw_rad) - float(ready_pose.yaw_rad)
        ))
        if (
            count - ready_at >= ttl
            or translation >= float(
                self.config.metric_independent_translation_m
            )
            or yaw >= float(self.config.metric_independent_yaw_rad)
        ):
            self._clear_negative_evidence(
                "normal", "negative_level_source_expired"
            )

    def _normal_global_geometry_engine(self) -> GpuGeometryValidator:
        validator = getattr(self, "_normal_global_validator", None)
        if validator is None:
            validator = GpuGeometryValidator(
                GridSpec(
                    float(self.config.recovery_grid_resolution_m),
                    float(self.config.recovery_grid_half_span_m),
                ),
                config=GeometryConfig(
                    windows=tuple(self.config.recovery_geometry_windows)
                ),
                device=str(self.device),
            )
            self._normal_global_validator = validator
        return validator

    def _normal_global_geometry_seeder(self) -> GpuGlobalGeometrySeeder:
        seeder = getattr(self, "_normal_global_seeder", None)
        if seeder is None:
            seeder = GpuGlobalGeometrySeeder(
                GridSpec(
                    float(self.config.recovery_grid_resolution_m),
                    float(self.config.recovery_grid_half_span_m),
                ),
                device=str(self.device),
            )
            self._normal_global_seeder = seeder
        return seeder

    def _normal_global_temporal_pose_bank(self) -> TemporalPoseHypothesisBank:
        bank = getattr(self, "_normal_global_pose_bank", None)
        if bank is None:
            bank = TemporalPoseHypothesisBank(TemporalPoseConfig(
                min_gap_frames=int(
                    max(
                        min(self.config.recovery_geometry_windows),
                        self.config.recovery_evidence_gap_observations,
                    )
                )
            ))
            self._normal_global_pose_bank = bank
        return bank

    def _historical_place_node_ids(self) -> frozenset[int]:
        """Snapshot graph nodes that still have immutable historical RGB-D."""

        raw_ids = getattr(self._metric_localizer, "place_keyframe_ids", None)
        if raw_ids is None:
            return frozenset()
        backed: set[int] = set()
        try:
            for raw_id in raw_ids:
                node_id = int(raw_id)
                if node_id > 0:
                    backed.add(node_id)
        except (TypeError, ValueError):
            return frozenset()
        record_ids = {int(record.node_id) for record in self._records}
        return frozenset(backed & record_ids)

    def _reset_frozen_resolver(
        self,
        scope: str,
        *,
        keep_warm: bool = False,
        force_next_probe: bool = False,
        replenish_confirmation_budget: bool = True,
    ) -> None:
        if scope not in {"normal", "recovery"}:
            raise ValueError(f"unknown frozen resolver scope: {scope}")
        warm_name = f"_{scope}_resolver_warm"
        summary_name = f"_{scope}_resolver_seed_summary"
        setattr(
            self,
            warm_name,
            bool(getattr(self, warm_name, False)) if keep_warm else False,
        )
        if not keep_warm:
            setattr(self, summary_name, {})
        setattr(self, f"_{scope}_resolver_force_probe", bool(force_next_probe))
        setattr(self, f"_{scope}_resolver_last_probe_frame", -1)
        setattr(self, f"_{scope}_resolver_last_probe_odometry_pose", None)
        setattr(self, f"_{scope}_resolver_confirmation_burst_remaining", 0)
        if replenish_confirmation_budget:
            setattr(
                self,
                f"_{scope}_resolver_confirmation_budget_remaining",
                2 * max(1, int(self.config.recovery_evidence_gap_observations)),
            )

    def _frozen_resolver_probe_due(
        self,
        scope: str,
        prepared: PreparedPlaceObservation,
        count: int,
    ) -> bool:
        if int(getattr(self, f"_{scope}_resolver_last_probe_frame", -1)) == int(
            prepared.frame_id
        ):
            return False
        if bool(getattr(self, f"_{scope}_resolver_force_probe", False)):
            return True
        if (
            int(getattr(
                self, f"_{scope}_resolver_confirmation_burst_remaining", 0
            )) > 0
            and int(getattr(
                self,
                f"_{scope}_resolver_confirmation_budget_remaining",
                2 * max(1, int(self.config.recovery_evidence_gap_observations)),
            )) > 0
        ):
            return True
        next_name = (
            "_normal_global_next_probe"
            if scope == "normal"
            else "_recovery_next_geometry_probe"
        )
        if int(count) >= int(getattr(self, next_name, 0)):
            return True
        if not bool(getattr(self, f"_{scope}_resolver_warm", False)):
            return False
        previous = getattr(
            self, f"_{scope}_resolver_last_probe_odometry_pose", None
        )
        if previous is None:
            return False
        motion = previous.relative_to(prepared.odometry_pose)
        return bool(
            math.hypot(float(motion.x_m), float(motion.y_m))
            >= float(self.config.metric_independent_translation_m)
            or abs(float(motion.yaw_rad))
            >= float(self.config.metric_independent_yaw_rad)
        )

    def _schedule_frozen_resolver_probe(
        self,
        scope: str,
        prepared: PreparedPlaceObservation,
        count: int,
    ) -> None:
        warm = bool(getattr(self, f"_{scope}_resolver_warm", False))
        spacing = (
            int(self.config.recovery_evidence_gap_observations)
            if warm
            else int(self.config.recovery_probe_spacing)
        )
        next_name = (
            "_normal_global_next_probe"
            if scope == "normal"
            else "_recovery_next_geometry_probe"
        )
        setattr(self, next_name, int(count) + max(1, spacing))
        self._claim_frozen_resolver_probe(scope, prepared)

    def _claim_frozen_resolver_probe(
        self,
        scope: str,
        prepared: PreparedPlaceObservation,
    ) -> None:
        """Make one frame idempotent before invoking fallible GPU work."""

        setattr(self, f"_{scope}_resolver_force_probe", False)
        setattr(
            self,
            f"_{scope}_resolver_last_probe_frame",
            int(prepared.frame_id),
        )
        setattr(
            self,
            f"_{scope}_resolver_last_probe_odometry_pose",
            prepared.odometry_pose,
        )

    def consume_normal_global_no_mode(self) -> bool:
        """Return a short-lived, generation-bound negative-search level.

        The worker combines this level with its independent sustained
        multi-view novelty gate. The level itself never authorizes map writes,
        and any UNKNOWN, positive mode, segment change or source motion clears
        it before a later frame can reuse it.
        """

        if not bool(getattr(self, "_normal_global_no_mode_ready", False)):
            return False
        target = self._negative_target("normal")
        binding = getattr(self, "_normal_negative_evidence_binding", None)
        current = (
            None if target is None else (
                str(target.digest_sha256), str(target.generation)
            )
        )
        if (
            current is None
            or current != binding
            or current[1] != self._negative_generation("normal")
        ):
            self._clear_negative_evidence(
                "normal", "negative_level_target_binding_changed"
            )
            return False
        return True

    def _observation_points(
        self, observation: OfficialObservation
    ) -> np.ndarray:
        depth = np.asarray(observation.depth_m, dtype=np.float32)
        if depth.ndim != 2 or not depth.size:
            return np.empty((0, 3), dtype=np.float32)
        dx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        dy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        edge = (dx > 0.12) | (dy > 0.12)
        edge |= np.roll(edge, 1, axis=1) | np.roll(edge, 1, axis=0)
        stride = max(1, int(self.config.recovery_depth_stride))
        sample_rows = np.arange(0, depth.shape[0], stride, dtype=np.int64)
        sample_columns = np.arange(0, depth.shape[1], stride, dtype=np.int64)
        rows, columns = np.meshgrid(
            sample_rows, sample_columns, indexing="ij"
        )
        rows = rows.reshape(-1)
        columns = columns.reshape(-1)
        values = depth[rows, columns]
        valid = (
            np.isfinite(values)
            & (values > 0.0)
            & ~edge[rows, columns]
        )
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)
        values = values[valid]
        rows = rows[valid]
        columns = columns[valid]
        intrinsics = observation.intrinsics
        optical = np.column_stack((
            (columns - float(intrinsics.cx)) / float(intrinsics.fx) * values,
            (rows - float(intrinsics.cy)) / float(intrinsics.fy) * values,
            values,
        )).astype(np.float32, copy=False)
        local = observation.camera_relative_pose.rtabmap_local_transform()
        return np.ascontiguousarray(
            optical @ local[:, :3].T + local[:, 3], dtype=np.float32
        )

    def _remember_recovery_geometry(
        self,
        observation: OfficialObservation,
        predicted_pose: SE2Pose,
        frame_id: int,
    ) -> bool:
        points = self._observation_points(observation)
        scan = extract_column_scan(
            points,
            column_m=float(self.config.recovery_column_m),
            min_range_m=float(self.config.minimum_depth_m),
            max_range_m=float(self.config.maximum_depth_m),
            obstacle_z_min_m=0.15,
            obstacle_z_max_m=1.95,
            chassis_block_z_max_m=0.80,
            wall_band_count=6,
            wall_band_min_run=2,
            self_clear_forward_m=0.95,
            self_clear_half_width_m=0.55,
        )
        return self._recovery_geometry_engine().remember(
            scan,
            (
                float(predicted_pose.x_m),
                float(predicted_pose.y_m),
                math.degrees(float(predicted_pose.yaw_rad)),
            ),
            int(frame_id),
        )

    def _remember_normal_global_geometry(
        self,
        observation: OfficialObservation,
        predicted_pose: SE2Pose,
        frame_id: int,
    ) -> bool:
        points = self._observation_points(observation)
        scan = extract_column_scan(
            points,
            column_m=float(self.config.recovery_column_m),
            min_range_m=float(self.config.minimum_depth_m),
            max_range_m=float(self.config.maximum_depth_m),
            obstacle_z_min_m=0.15,
            obstacle_z_max_m=1.95,
            chassis_block_z_max_m=0.80,
            wall_band_count=6,
            wall_band_min_run=2,
            self_clear_forward_m=0.95,
            self_clear_half_width_m=0.55,
        )
        return self._normal_global_geometry_engine().remember(
            scan,
            (
                float(predicted_pose.x_m),
                float(predicted_pose.y_m),
                math.degrees(float(predicted_pose.yaw_rad)),
            ),
            int(frame_id),
        )

    def propose(
        self, prepared: PreparedPlaceObservation
    ) -> Optional[ExternalLoopHypothesis]:
        self.queries += 1
        if not prepared.structural_usable:
            self._pending_candidate_id = 0
            self.last_geometry_reason = "structural_observation_unusable"
            return None
        if self._native_novelty_reconciliation_pending():
            # Native is still reconciling ordinary mapping scans against the
            # pre-resume snapshot.  Do not create a competing Python query or
            # recovery hold while those writes are in flight.
            self._age_cooldowns()
            self._pending_candidate_id = 0
            self.last_geometry_reason = (
                "native_novelty_resume_reconciliation_pending"
            )
            return None
        if prepared.normal_global_hold:
            self._age_cooldowns()
            return self._propose_normal_global_geometry(prepared)
        if prepared.recovery_hold:
            self._age_cooldowns()
            return self._propose_recovery_geometry(prepared, [])
        self._age_cooldowns()
        if (
            self._centers is None
            or prepared.image_features is None
            or len(prepared.descriptors) == 0
        ):
            self._recent_rankings.append([])
            self._pending_candidate_id = 0
            self.last_geometry_reason = "appearance_index_not_ready"
            if prepared.recovery_hold:
                return self._propose_recovery_geometry(prepared, [])
            return None
        query_histogram = self._histogram(prepared.descriptors)
        self._query_histograms.append(query_histogram)
        if len(self._query_histograms) < self.config.sequence_window:
            self._recent_rankings.append([])
            self._pending_candidate_id = 0
            self.last_geometry_reason = "query_sequence_not_ready"
            if prepared.recovery_hold:
                return self._propose_recovery_geometry(prepared, [])
            return None
        ranked = self._rank(prepared.path_progress_m)
        ranked_indices = [index for index, _score in ranked]
        self._recent_rankings.append(ranked_indices)
        proposals = self._confirmed_proposals(ranked)
        if prepared.recovery_hold:
            # During recovery, appearance is a read-only retrieval hint rather
            # than an authority to edit the graph. Waiting for three SeqSLAM
            # queries before even attempting RGB-D can miss a brief revisited
            # corner. Send every adaptively significant top-K region to the
            # metric verifier; two independent RGB-D viewpoints and the native
            # graph transaction remain mandatory before any permanent edge.
            single_query = [
                PlaceProposal(
                    int(self._records[index].node_id), float(score), 1
                )
                for index, score in ranked
            ]
            retry = list(getattr(self, "_recovery_retry_proposals", []))
            merged: dict[int, PlaceProposal] = {}
            for proposal in proposals + single_query + retry:
                old = merged.get(int(proposal.node_id))
                if old is None or float(proposal.score) > float(old.score):
                    merged[int(proposal.node_id)] = proposal
            proposals = list(merged.values())
        if not proposals:
            self._pending_candidate_id = 0
            self.last_geometry_reason = "appearance_not_confirmed"
            if prepared.recovery_hold:
                return self._propose_recovery_geometry(prepared, [])
            return None
        if prepared.recovery_hold:
            self._recovery_last_candidate_frame = int(prepared.frame_id)
            metric_proposals = self._recovery_viewpoint_proposals(proposals)
        else:
            metric_proposals = proposals

        self.geometry_attempts += 1
        observation = prepared.observation
        localization = self._metric_localizer.localize(
            f"query-{prepared.frame_id}",
            observation.rgb,
            observation.depth_m,
            self._camera_dict(observation),
            predicted_pose=None,
            frame_index=prepared.frame_id,
            min_frame_gap=0,
            min_independent=1,
            allow_single_strong=False,
            image_features=prepared.image_features,
            place_candidates=[
                str(proposal.node_id) for proposal in metric_proposals
            ],
            require_cuda_matching=bool(prepared.recovery_hold),
        )
        if localization is None:
            if prepared.recovery_hold:
                remaining = max(
                    0,
                    int(getattr(
                        self, "_recovery_retry_attempts_remaining", 0
                    )) - 1,
                )
                self._recovery_retry_attempts_remaining = remaining
                if remaining == 0:
                    self._recovery_retry_proposals = []
            self._pending_candidate_id = 0
            self.last_geometry_reason = str(
                self._metric_localizer.last_reason or "rgbd_geometry_rejected"
            )
            if prepared.recovery_hold:
                self._block_negative_for_frame(
                    "recovery",
                    prepared.frame_id,
                    f"recovery_appearance_unknown:{self.last_geometry_reason}",
                )
                return self._propose_recovery_geometry(
                    prepared, metric_proposals
                )
            return None

        if prepared.recovery_hold:
            try:
                candidate_id = int(localization.keyframe_id)
            except (TypeError, ValueError):
                candidate_id = 0
            requested = {int(row.node_id) for row in metric_proposals}
            if candidate_id not in requested:
                self._recovery_retry_proposals = []
                self._recovery_retry_attempts_remaining = 0
                self._pending_candidate_id = 0
                self._block_negative_for_frame(
                    "recovery",
                    prepared.frame_id,
                    "recovery_appearance_unknown:candidate_contract_violation",
                )
                return self._propose_recovery_geometry(
                    prepared, metric_proposals
                )

        self.geometry_accepted += 1
        if prepared.recovery_hold:
            self._clear_negative_evidence(
                "recovery", "recovery_positive_rgbd_metric"
            )
        metric = self._metric_observation(prepared, localization)
        self._metric_history.append(metric)
        if prepared.recovery_hold:
            # Keep the successfully verified physical region warm for the next
            # few usable observations. A later query still has to pass RGB-D;
            # retaining the shortlist cannot itself authorize localization.
            ordered = [
                PlaceProposal(
                    int(metric.candidate_id),
                    max((float(row.score) for row in metric_proposals), default=1.0),
                    1,
                )
            ]
            ordered.extend(metric_proposals)
            unique: dict[int, PlaceProposal] = {}
            for proposal in ordered:
                unique.setdefault(int(proposal.node_id), proposal)
            self._recovery_retry_proposals = list(unique.values())[
                : max(1, int(self.config.recovery_max_candidates))
            ]
            self._recovery_retry_attempts_remaining = max(
                2, int(self.config.metric_confirmation_history)
            )
        hypothesis = self._confirmed_metric_hypothesis(metric)
        self._recent_metric_events.append({
            "frame_id": metric.frame_id,
            "candidate_id": metric.candidate_id,
            "inliers": metric.inliers,
            "inlier_ratio": metric.inlier_ratio,
            "rmse_m": metric.rmse_m,
            "consensus": hypothesis is not None,
        })
        if hypothesis is None:
            self._pending_candidate_id = 0
            self.last_geometry_reason = "metric_temporal_consensus_pending"
            if prepared.recovery_hold:
                return self._propose_recovery_geometry(
                    prepared, metric_proposals
                )
            return None

        if prepared.recovery_hold:
            hypothesis = ExternalLoopHypothesis(
                hypothesis.candidate_id,
                hypothesis.candidate_to_current,
                hypothesis.covariance,
                recovery_relocalization=True,
            )

        self._pending_candidate_id = hypothesis.candidate_id
        self.metric_consensus_accepted += 1
        self.proposals += 1
        self.last_geometry_reason = "metric_temporal_consensus_accepted"
        return hypothesis

    def _normal_hold_appearance_proposals(
        self, prepared: PreparedPlaceObservation
    ) -> list[PlaceProposal]:
        """Retrieve immutable historical nodes without authorizing a link.

        The adaptive appearance score is deliberately only a shortlist.  A
        retained successful region also remains only a shortlist and must
        pass self-masked GPU RGB-D registration again on every query.
        """

        proposals = list(getattr(self, "_normal_retry_proposals", []))
        if (
            self._centers is not None
            and prepared.image_features is not None
            and len(prepared.descriptors)
        ):
            query_histogram = self._histogram(prepared.descriptors)
            self._query_histograms.append(query_histogram)
            if len(self._query_histograms) >= self.config.sequence_window:
                ranked = self._rank(prepared.path_progress_m)
                self._recent_rankings.append(
                    [index for index, _score in ranked]
                )
                # No temporal appearance confirmation is claimed here.  Each
                # adaptive top-K row is only a proposal for metric RGB-D.
                proposals.extend(
                    PlaceProposal(
                        int(self._records[index].node_id), float(score), 1
                    )
                    for index, score in ranked
                )
            else:
                self._recent_rankings.append([])

        allowed = frozenset(
            int(value)
            for value in getattr(self, "_normal_allowed_node_ids", frozenset())
        )
        if not allowed:
            return []
        merged: dict[int, PlaceProposal] = {}
        for seed in proposals:
            for proposal in self._recovery_viewpoint_proposals([seed]):
                node_id = int(proposal.node_id)
                if allowed and node_id not in allowed:
                    continue
                old = merged.get(node_id)
                if old is None or float(proposal.score) > float(old.score):
                    merged[node_id] = proposal
        return list(merged.values())[
            : max(1, int(self.config.recovery_max_candidates))
        ]

    def _propose_normal_hold_appearance(
        self, prepared: PreparedPlaceObservation
    ) -> Optional[ExternalLoopHypothesis]:
        """Try a fail-closed historical RGB-D bridge during normal hold."""

        return self._propose_frozen_global_geometry(prepared, "normal")

        # Legacy implementation retained temporarily for source compatibility.

        proposals = self._normal_hold_appearance_proposals(prepared)
        if not proposals or prepared.image_features is None:
            return None
        self.geometry_attempts += 1
        observation = prepared.observation
        localization = self._metric_localizer.localize(
            f"normal-appearance-{prepared.frame_id}",
            observation.rgb,
            observation.depth_m,
            self._camera_dict(observation),
            predicted_pose=None,
            frame_index=prepared.frame_id,
            min_frame_gap=0,
            min_independent=1,
            allow_single_strong=False,
            image_features=prepared.image_features,
            place_candidates=[str(row.node_id) for row in proposals],
            require_cuda_matching=True,
        )
        if localization is None:
            remaining = max(
                0,
                int(getattr(self, "_normal_retry_attempts_remaining", 0)) - 1,
            )
            self._normal_retry_attempts_remaining = remaining
            if remaining == 0:
                self._normal_retry_proposals = []
            self._block_negative_for_frame(
                "normal",
                prepared.frame_id,
                "normal_appearance_unknown:"
                + str(
                    self._metric_localizer.last_reason
                    or "rgbd_geometry_rejected"
                ),
            )
            return None

        try:
            candidate_id = int(localization.keyframe_id)
        except (TypeError, ValueError):
            candidate_id = 0
        allowed = frozenset(
            int(value)
            for value in getattr(self, "_normal_allowed_node_ids", frozenset())
        )
        proposed = {int(row.node_id) for row in proposals}
        if candidate_id not in allowed or candidate_id not in proposed:
            # A localizer/backend contract violation is UNKNOWN.  It cannot
            # name a graph endpoint outside this hold's immutable snapshot.
            self._normal_retry_proposals = []
            self._normal_retry_attempts_remaining = 0
            self._block_negative_for_frame(
                "normal",
                prepared.frame_id,
                "normal_appearance_unknown:candidate_contract_violation",
            )
            return None

        self._clear_negative_evidence(
            "normal", "normal_positive_rgbd_metric"
        )
        self.geometry_accepted += 1
        metric = self._metric_observation(prepared, localization)
        independently_confirmed = self._node_metric_has_independent_confirmation(
            self._normal_metric_history, metric
        )
        self._normal_metric_history.append(metric)
        self._recent_metric_events.append({
            "frame_id": metric.frame_id,
            "candidate_id": metric.candidate_id,
            "inliers": metric.inliers,
            "inlier_ratio": metric.inlier_ratio,
            "rmse_m": metric.rmse_m,
            "normal_hold_appearance": True,
            "consensus": independently_confirmed,
        })

        # Keep the verified physical region warm only long enough to collect a
        # second actual viewpoint.  Retention never bypasses RGB-D matching.
        ordered = [
            PlaceProposal(
                int(metric.candidate_id),
                max((float(row.score) for row in proposals), default=1.0),
                1,
            )
        ]
        ordered.extend(proposals)
        unique: dict[int, PlaceProposal] = {}
        for proposal in ordered:
            unique.setdefault(int(proposal.node_id), proposal)
        self._normal_retry_proposals = list(unique.values())[
            : max(1, int(self.config.recovery_max_candidates))
        ]
        self._normal_retry_attempts_remaining = max(
            2, int(self.config.metric_confirmation_history)
        )

        if not independently_confirmed:
            return None
        hypothesis = ExternalLoopHypothesis(
            metric.hypothesis.candidate_id,
            metric.hypothesis.candidate_to_current,
            metric.hypothesis.covariance,
            verified_graph_bridge=True,
        )
        self._pending_candidate_id = hypothesis.candidate_id
        self.metric_consensus_accepted += 1
        self.proposals += 1
        self.normal_global_bridges += 1
        self.last_geometry_reason = "normal_appearance_verified_graph_bridge"
        return hypothesis

    def _normal_mode_records(self, pose: SE2Pose) -> list[_PlaceRecord]:
        """Return every stored RGB-D viewpoint capable of naming this mode."""

        region = max(1e-6, float(self.config.candidate_region_m))
        allowed = (
            getattr(self, "_normal_allowed_node_ids", frozenset())
            if getattr(self, "_normal_global_pending", False)
            else frozenset()
        )
        rows = [
            record for record in self._records
            if (not allowed or int(record.node_id) in allowed)
            if math.hypot(
                float(record.pose.x_m) - float(pose.x_m),
                float(record.pose.y_m) - float(pose.y_m),
            ) <= region
        ]
        return sorted(rows, key=lambda record: int(record.node_id))

    def _note_normal_no_mode_interval(
        self, interval: tuple[int, int]
    ) -> bool:
        """Require two non-overlapping exhaustive negative probes."""

        first, last = (int(value) for value in interval)
        gap = int(self.config.recovery_evidence_gap_observations)
        previous = list(getattr(self, "_normal_global_no_mode_intervals", []))
        if any(
            (old_last < first and first - old_last - 1 >= gap)
            or (last < old_first and old_first - last - 1 >= gap)
            for old_first, old_last in previous
        ):
            self._normal_global_no_mode_intervals.append((first, last))
            return True
        self._normal_global_no_mode_intervals = [(first, last)]
        return False

    def _node_metric_has_independent_confirmation(
        self,
        history: deque[_MetricObservation],
        current: _MetricObservation,
    ) -> bool:
        current_record = self._record_for_node(current.candidate_id)
        for previous in list(history):
            if int(current.frame_id) == int(previous.frame_id):
                continue
            previous_record = self._record_for_node(previous.candidate_id)
            if current_record is None or previous_record is None:
                continue
            if math.hypot(
                current_record.pose.x_m - previous_record.pose.x_m,
                current_record.pose.y_m - previous_record.pose.y_m,
            ) > 2.0 * float(self.config.candidate_region_m):
                continue
            model = self._metric_pair_consistency_model(previous, current)
            if model is None:
                continue
            if not self._metric_queries_have_independent_viewpoints(
                previous, current, consistency_model=model
            ):
                continue
            return True
        return False

    def _metric_queries_have_independent_viewpoints(
        self,
        previous: _MetricObservation,
        current: _MetricObservation,
        *,
        consistency_model: Optional[str] = None,
    ) -> bool:
        """Use observed RGB-D poses, not frame count or commanded qvel."""

        if int(previous.frame_id) == int(current.frame_id):
            return False
        model = consistency_model or self._metric_pair_consistency_model(
            previous, current
        )
        if model != "moving":
            return False
        measured = previous.map_pose.relative_to(current.map_pose)
        translation_sigma = 3.0 * (
            float(previous.registration_translation_std_m)
            + float(current.registration_translation_std_m)
        )
        yaw_sigma = 3.0 * (
            float(previous.registration_yaw_std_rad)
            + float(current.registration_yaw_std_rad)
        )
        if not math.isfinite(translation_sigma) or not math.isfinite(yaw_sigma):
            return False
        return bool(
            math.hypot(measured.x_m, measured.y_m) - translation_sigma
            >= float(self.config.metric_independent_translation_m)
            or abs(measured.yaw_rad) - yaw_sigma
            >= float(self.config.metric_independent_yaw_rad)
        )

    def _frozen_mode_record_ids(
        self,
        allowed: frozenset[int],
        predicted: SE2Pose,
        candidates: list[dict[str, Any]],
    ) -> list[int]:
        """Summarize seeder modes without restricting the RGB-D authority set."""

        region = max(1e-6, float(self.config.candidate_region_m))
        selected: set[int] = set()
        for row in candidates:
            try:
                x_m = float(predicted.x_m) + float(row["dx_m"])
                y_m = float(predicted.y_m) + float(row["dy_m"])
            except (KeyError, TypeError, ValueError):
                continue
            for record in self._records:
                if int(record.node_id) not in allowed:
                    continue
                if math.hypot(record.pose.x_m - x_m, record.pose.y_m - y_m) <= region:
                    selected.add(int(record.node_id))
        return sorted(selected)

    def _note_frozen_resolver_event(
        self, event: dict[str, object]
    ) -> None:
        events = getattr(self, "_recent_recovery_events", None)
        if events is not None:
            events.append(event)

    def _qualified_metric_confirmation(
        self,
        history: deque[_MetricObservation],
        current: _MetricObservation,
    ) -> tuple[bool, str]:
        """Keep one live-view vote until an independent consistent vote arrives."""

        previous = history[0] if history else None
        if previous is None:
            history.append(current)
            return False, "first_live_query_support"
        previous_record = self._record_for_node(previous.candidate_id)
        current_record = self._record_for_node(current.candidate_id)
        same_mode = bool(
            previous_record is not None
            and current_record is not None
            and math.hypot(
                current_record.pose.x_m - previous_record.pose.x_m,
                current_record.pose.y_m - previous_record.pose.y_m,
            ) <= 2.0 * float(self.config.candidate_region_m)
        )
        model = (
            self._metric_pair_consistency_model(previous, current)
            if same_mode
            else None
        )
        if model is None:
            history.clear()
            history.append(current)
            return False, "live_query_correction_disagreement"
        if not self._metric_queries_have_independent_viewpoints(
            previous, current, consistency_model=model
        ):
            return False, f"{model}_live_query_not_independent"
        history.append(current)
        return True, "independent_live_query_consensus"

    def _propose_frozen_global_geometry(
        self,
        prepared: PreparedPlaceObservation,
        scope: str,
    ) -> Optional[ExternalLoopHypothesis]:
        """Resolve one frozen hold with exhaustive RGB-D and current geometry."""

        if scope not in {"normal", "recovery"}:
            raise ValueError(f"unknown frozen resolver scope: {scope}")
        normal = scope == "normal"
        count = int(getattr(
            self,
            "_normal_global_usable_observations"
            if normal else "_recovery_usable_observations",
            0,
        ))
        expected_generation = int(getattr(
            self,
            "_normal_hold_generation" if normal else "_recovery_hold_generation",
            0,
        ))
        if (
            expected_generation <= 0
            or int(getattr(prepared, "query_generation", 0))
            != expected_generation
        ):
            self._block_negative_for_frame(
                scope, prepared.frame_id, "frozen_resolver_transaction_mismatch"
            )
            self.last_geometry_reason = f"{scope}_resolver_transaction_mismatch"
            return None
        if not self._frozen_resolver_probe_due(scope, prepared, count):
            self.last_geometry_reason = f"{scope}_global_window_not_ready"
            return None

        target_wall = getattr(
            self, "_normal_target_wall" if normal else "_recovery_target_wall"
        )
        target_free = getattr(
            self, "_normal_target_free" if normal else "_recovery_target_free"
        )
        validator = (
            self._normal_global_geometry_engine()
            if normal else self._recovery_geometry_engine()
        )
        rolling = validator.rolling_source()
        event: dict[str, object] = {
            "frame_id": int(prepared.frame_id),
            "scope": scope,
            "query_generation": expected_generation,
            "place_database_generation": int(getattr(
                self,
                "_normal_place_database_generation"
                if normal else "_recovery_place_database_generation",
                0,
            )),
            "rolling_interval": [
                rolling.get("first_frame"), rolling.get("last_frame")
            ],
            "rolling_sample_count": int(rolling.get("sample_count") or 0),
            "search_complete": False,
            "candidate_budget_truncated": False,
            "mode_count": 0,
            "mode_union_record_count": 0,
            "allowed_record_count": 0,
            "searched_record_count": 0,
            "localizer_reason": "not_attempted",
            "candidate_id": 0,
            "validator_reason": "not_attempted",
            "per_query_visual_support_safe": False,
            "live_independent_query_count": 0,
            "accepted": False,
        }

        def finish(reason: str) -> None:
            event["reason"] = str(reason)
            self._note_frozen_resolver_event(event)
            self.last_geometry_reason = str(reason)

        if len(target_wall) < 16 or not len(target_free):
            self._block_negative_for_frame(
                scope, prepared.frame_id, "frozen_resolver_target_missing"
            )
            self._schedule_frozen_resolver_probe(scope, prepared, count)
            finish(f"{scope}_global_frozen_map_missing")
            return None
        interval_values = (rolling.get("first_frame"), rolling.get("last_frame"))
        if any(value is None for value in interval_values):
            self._block_negative_for_frame(
                scope, prepared.frame_id, "frozen_resolver_interval_missing"
            )
            self._schedule_frozen_resolver_probe(scope, prepared, count)
            finish(f"{scope}_global_interval_missing")
            return None

        self._claim_frozen_resolver_probe(scope, prepared)
        burst_name = f"_{scope}_resolver_confirmation_burst_remaining"
        budget_name = f"_{scope}_resolver_confirmation_budget_remaining"
        burst_remaining = int(getattr(self, burst_name, 0))
        budget_remaining = int(getattr(
            self,
            budget_name,
            2 * max(1, int(self.config.recovery_evidence_gap_observations)),
        ))
        if burst_remaining > 0 and budget_remaining > 0:
            # A burst budgets probe frames, not only successful registrations.
            # Incomplete seed searches and missing query features may still run
            # GPU work, so letting them escape this counter would make a burst
            # unbounded precisely when the scene is least observable.
            burst_remaining -= 1
            budget_remaining -= 1
            setattr(self, burst_name, burst_remaining)
            setattr(self, budget_name, budget_remaining)
            event["confirmation_burst_remaining"] = burst_remaining
            event["confirmation_budget_remaining"] = budget_remaining

        predicted = prepared.predicted_map_pose
        warm_name = f"_{scope}_resolver_warm"
        summary_name = f"_{scope}_resolver_seed_summary"
        warm = bool(getattr(self, warm_name, False))
        summary = dict(getattr(self, summary_name, {}))
        seed_complete = bool(summary.get("search_complete", False))
        if not warm or not seed_complete:
            try:
                if normal:
                    self.normal_global_attempts += 1
                    seeder = self._normal_global_geometry_seeder()
                else:
                    self.recovery_global_attempts += 1
                    seeder = self._recovery_global_geometry_seeder()
                global_report = seeder.seed(
                    rolling["wall"],
                    rolling["free"],
                    target_wall,
                    target_free,
                    pivot=(float(predicted.x_m), float(predicted.y_m)),
                )
            except Exception as error:
                reason = f"seeder_exception:{type(error).__name__}"
                setattr(self, summary_name, {
                    "search_complete": False,
                    "candidate_budget_truncated": False,
                    "mode_count": 0,
                    "mode_union_record_count": 0,
                    "seeder_reason": reason,
                })
                setattr(self, warm_name, True)
                event["seeder_reason"] = reason
                self._block_negative_for_frame(
                    scope, prepared.frame_id, "frozen_geometry_seeder_failed"
                )
                self._schedule_frozen_resolver_probe(scope, prepared, count)
                finish(f"{scope}_frozen_geometry_seeder_failed:{reason}")
                return None
            candidates = list(global_report.get("candidates") or [])
            if normal:
                self.normal_global_candidates += len(candidates)
            else:
                self.recovery_global_candidates += len(candidates)
            complete = bool(global_report.get("search_complete", False))
            truncated = bool(global_report.get("candidate_budget_truncated", False))
            event.update({
                "search_complete": complete,
                "candidate_budget_truncated": truncated,
                "mode_count": len(candidates),
                "geometry_backend": global_report.get("geometry_backend"),
                "seeder_reason": str(global_report.get("reason") or ""),
            })
            if (
                complete
                and (
                    str(global_report.get("reason"))
                    != "global_candidates_generated"
                    or not candidates
                )
            ):
                disposition = self._evaluate_global_negative(
                    scope,
                    rolling,
                    predicted,
                    observation_count=count,
                    odometry_pose=prepared.odometry_pose,
                    query_frame_id=prepared.frame_id,
                )
                self._schedule_frozen_resolver_probe(scope, prepared, count)
                finish(f"{scope}_global_negative_{disposition.lower()}")
                return None
            allowed = frozenset(getattr(
                self,
                "_normal_allowed_node_ids"
                if normal else "_recovery_allowed_node_ids",
                frozenset(),
            ))
            mode_ids = self._frozen_mode_record_ids(
                allowed, predicted, candidates
            )
            event["mode_union_record_count"] = len(mode_ids)
            summary = {
                "search_complete": complete,
                "candidate_budget_truncated": truncated,
                "mode_count": len(candidates),
                "mode_union_record_count": len(mode_ids),
                "seeder_reason": str(global_report.get("reason") or ""),
            }
            setattr(self, summary_name, summary)
            setattr(self, warm_name, True)
            warm = True
        else:
            event.update(summary)
            event["warm_retry"] = True

        self._schedule_frozen_resolver_probe(scope, prepared, count)
        self._block_negative_for_frame(
            scope, prepared.frame_id, "frozen_rgbd_mode_resolution_pending"
        )
        allowed = frozenset(int(value) for value in getattr(
            self,
            "_normal_allowed_node_ids"
            if normal else "_recovery_allowed_node_ids",
            frozenset(),
        ))
        event["allowed_record_count"] = len(allowed)
        current_backed = self._historical_place_node_ids()
        missing = sorted(allowed - current_backed)
        if (
            not allowed
            or missing
            or len(allowed) > int(self.config.maximum_representatives)
        ):
            event["missing_allowed_record_count"] = len(missing)
            finish(f"{scope}_frozen_rgbd_universe_incomplete")
            return None
        if prepared.image_features is None:
            finish(f"{scope}_query_features_missing")
            return None

        candidate_ids = sorted(allowed)
        event["searched_record_count"] = len(candidate_ids)
        self.geometry_attempts += 1
        if not normal:
            self.recovery_geometry_attempts += 1
        localization = self._metric_localizer.localize(
            f"{scope}-frozen-{prepared.frame_id}",
            prepared.observation.rgb,
            prepared.observation.depth_m,
            self._camera_dict(prepared.observation),
            predicted_pose=None,
            frame_index=prepared.frame_id,
            min_frame_gap=0,
            min_independent=1,
            allow_single_strong=False,
            image_features=prepared.image_features,
            place_candidates=[str(node_id) for node_id in candidate_ids],
            require_cuda_matching=True,
        )
        event["localizer_reason"] = str(
            self._metric_localizer.last_reason or ""
        )
        if localization is None:
            finish(f"{scope}_frozen_rgbd_unresolved:{event['localizer_reason']}")
            return None
        try:
            candidate_id = int(localization.keyframe_id)
        except (TypeError, ValueError):
            candidate_id = 0
        event.update({
            "candidate_id": candidate_id,
            "winning_cluster_candidates": int(
                getattr(localization, "winning_cluster_candidates", 0)
            ),
            "winning_cluster_observable_candidates": int(
                getattr(localization, "winning_cluster_observable_candidates", 0)
            ),
            "witness_selection_reason": str(
                getattr(localization, "witness_selection_reason", "")
            ),
        })
        if candidate_id not in allowed:
            finish(f"{scope}_frozen_rgbd_candidate_contract_violation")
            return None
        record = self._record_for_node(candidate_id)
        if record is None:
            finish(f"{scope}_frozen_rgbd_record_missing")
            return None
        edge = SE2Pose(
            float(localization.keyframe_to_current_x_m),
            float(localization.keyframe_to_current_y_m),
            math.radians(float(localization.keyframe_to_current_yaw_deg)),
        )
        composed = record.pose.compose(edge)
        localized_pose = SE2Pose(
            float(localization.x_m),
            float(localization.y_m),
            math.radians(float(localization.yaw_deg)),
        )
        if (
            math.hypot(
                composed.x_m - localized_pose.x_m,
                composed.y_m - localized_pose.y_m,
            ) > 1e-4
            or abs(_wrap_rad(composed.yaw_rad - localized_pose.yaw_rad))
            > math.radians(0.01)
        ):
            finish(f"{scope}_frozen_rgbd_pose_edge_mismatch")
            return None

        visual_correction = (
            float(localized_pose.x_m) - float(predicted.x_m),
            float(localized_pose.y_m) - float(predicted.y_m),
            math.degrees(_wrap_rad(
                float(localized_pose.yaw_rad) - float(predicted.yaw_rad)
            )),
        )
        observability = {
            "independent_candidates": int(localization.independent_candidates),
            "translation_std_m": float(localization.translation_std_m),
            "yaw_std_deg": float(localization.yaw_std_deg),
            "normal_matrix_condition": float(localization.normal_matrix_condition),
            "geometry_major_span_m": float(localization.geometry_major_span_m),
            "geometry_minor_span_m": float(localization.geometry_minor_span_m),
            "pixel_coverage_x": float(localization.pixel_coverage_x),
            "pixel_coverage_y": float(localization.pixel_coverage_y),
        }
        report = validator.validate(
            target_wall,
            target_free,
            pivot=(float(predicted.x_m), float(predicted.y_m)),
            visual_correction=visual_correction,
            visual_metric=True,
            visual_observability=observability,
        )
        event.update({
            "validator_reason": str(report.get("reason") or ""),
            "per_query_visual_support_safe": bool(
                report.get("per_query_visual_support_safe", False)
            ),
            "per_query_visual_support_mode": str(
                report.get("per_query_visual_support_mode") or "none"
            ),
            "per_query_visual_support_window_count": int(
                report.get("per_query_visual_support_window_count") or 0
            ),
            "visual_observability": observability,
        })
        if not bool(report.get("per_query_visual_support_safe", False)):
            finish(f"{scope}_per_query_visual_support_unsafe")
            return None

        self._clear_negative_evidence(
            scope, f"{scope}_per_query_visual_support"
        )
        self.geometry_accepted += 1
        metric = self._metric_observation(prepared, localization)
        history = (
            self._normal_metric_history
            if normal else self._recovery_node_metric_history
        )
        confirmed, confirmation_reason = self._qualified_metric_confirmation(
            history, metric
        )
        if confirmation_reason in {
            "first_live_query_support",
            "live_query_correction_disagreement",
        }:
            budget_remaining = int(getattr(
                self,
                budget_name,
                2 * max(1, int(self.config.recovery_evidence_gap_observations)),
            ))
            setattr(
                self,
                burst_name,
                min(
                    max(1, int(self.config.recovery_evidence_gap_observations)),
                    max(0, budget_remaining),
                ),
            )
        elif confirmed:
            setattr(self, burst_name, 0)
        event.update({
            "live_independent_query_count": 2 if confirmed else len(history),
            "confirmation_reason": confirmation_reason,
            "confirmation_burst_remaining": int(
                getattr(self, burst_name, 0)
            ),
            "confirmation_budget_remaining": int(getattr(
                self,
                budget_name,
                2 * max(1, int(self.config.recovery_evidence_gap_observations)),
            )),
            "accepted": confirmed,
        })
        self._recent_metric_events.append({
            "frame_id": int(metric.frame_id),
            "candidate_id": int(metric.candidate_id),
            "inliers": int(metric.inliers),
            "inlier_ratio": float(metric.inlier_ratio),
            "rmse_m": float(metric.rmse_m),
            "scope": scope,
            "per_query_visual_support_safe": True,
            "consensus": confirmed,
            "reason": confirmation_reason,
        })
        if not confirmed:
            finish(f"{scope}_{confirmation_reason}")
            return None

        hypothesis = ExternalLoopHypothesis(
            metric.hypothesis.candidate_id,
            metric.hypothesis.candidate_to_current,
            metric.hypothesis.covariance,
            recovery_relocalization=not normal,
            verified_graph_bridge=normal,
        )
        self._pending_candidate_id = int(hypothesis.candidate_id)
        self.metric_consensus_accepted += 1
        self.proposals += 1
        if normal:
            self.normal_global_bridges += 1
        else:
            self.recovery_geometry_consensus += 1
        finish(f"{scope}_frozen_rgbd_live_consensus")
        return hypothesis

    def _propose_normal_global_geometry(
        self, prepared: PreparedPlaceObservation
    ) -> Optional[ExternalLoopHypothesis]:
        return self._propose_frozen_global_geometry(prepared, "normal")

        # Legacy implementation retained temporarily for source compatibility.
        count = int(self._normal_global_usable_observations)
        if count < int(self._normal_global_next_probe):
            self.last_geometry_reason = "normal_global_window_not_ready"
            return None
        if bool(getattr(self, "_normal_global_no_mode_ready", False)):
            self._clear_negative_evidence(
                "normal", "normal_global_next_probe_started"
            )
        self._normal_global_next_probe = (
            count + int(self.config.recovery_probe_spacing)
        )
        target_wall = self._normal_target_wall
        target_free = self._normal_target_free
        if len(target_wall) < 16 or not len(target_free):
            self._clear_negative_evidence(
                "normal", "normal_global_frozen_map_missing"
            )
            self.last_geometry_reason = "normal_global_frozen_map_missing"
            return None

        validator = self._normal_global_geometry_engine()
        rolling = validator.rolling_source()
        interval_values = (rolling.get("first_frame"), rolling.get("last_frame"))
        if any(value is None for value in interval_values):
            self._clear_negative_evidence(
                "normal", "normal_global_interval_missing"
            )
            self.last_geometry_reason = "normal_global_interval_missing"
            return None
        interval = (int(interval_values[0]), int(interval_values[1]))
        predicted = prepared.predicted_map_pose
        predicted_deg = (
            float(predicted.x_m),
            float(predicted.y_m),
            math.degrees(float(predicted.yaw_rad)),
        )
        self.normal_global_attempts += 1
        global_report = self._normal_global_geometry_seeder().seed(
            rolling["wall"], rolling["free"], target_wall, target_free,
            pivot=(float(predicted.x_m), float(predicted.y_m)),
        )
        candidates = list(global_report.get("candidates") or [])
        self.normal_global_candidates += len(candidates)
        # CUDA/backend/observability failures are UNKNOWN, never no-mode.
        if str(global_report.get("reason")) != "global_candidates_generated":
            disposition = self._evaluate_global_negative(
                "normal",
                rolling,
                predicted,
                observation_count=count,
                odometry_pose=prepared.odometry_pose,
                query_frame_id=prepared.frame_id,
            )
            if disposition == "UNKNOWN":
                self.last_geometry_reason = (
                    "normal_global_modes_unresolved:"
                    + str(global_report.get("reason") or "seed_unknown")
                )
            return None

        accepted: list[tuple[dict[str, Any], SE2Pose]] = []
        validation_reports: list[dict[str, Any]] = []
        for row in candidates:
            correction = (
                float(row["dx_m"]),
                float(row["dy_m"]),
                float(row["yaw_correction_deg"]),
            )
            report = validator.validate(
                target_wall,
                target_free,
                pivot=(float(predicted.x_m), float(predicted.y_m)),
                visual_correction=correction,
                visual_metric=False,
            )
            validation_reports.append(report)
            fused = report.get("fused_correction")
            if (
                report.get("accepted")
                and report.get("revisit_evidence_safe")
                and isinstance(fused, (list, tuple))
                and len(fused) == 3
            ):
                accepted.append((report, SE2Pose(
                    float(predicted.x_m) + float(fused[0]),
                    float(predicted.y_m) + float(fused[1]),
                    _wrap_rad(
                        float(predicted.yaw_rad)
                        + math.radians(float(fused[2]))
                    ),
                )))

        if not accepted:
            # Seeder/validator rejection alone remains UNKNOWN. Only the
            # independent complete wall-support prover may add no-mode
            # evidence, and even it needs two disjoint source windows.
            disposition = self._evaluate_global_negative(
                "normal",
                rolling,
                predicted,
                observation_count=count,
                odometry_pose=prepared.odometry_pose,
                query_frame_id=prepared.frame_id,
            )
            if disposition == "UNKNOWN":
                self.last_geometry_reason = "normal_global_modes_unresolved"
            return None
        self._clear_negative_evidence(
            "normal", "normal_global_positive_geometry_mode"
        )
        if any(
            not self._pose_hypotheses_agree(accepted[0][1], row[1])
            for row in accepted[1:]
        ):
            self.normal_global_ambiguous_probes += 1
            self.last_geometry_reason = "normal_global_competing_modes"
            return None

        report, corrected = accepted[0]

        # A raster mode has no node identity. Search every historical RGB-D
        # viewpoint in its metric region and accept only the node actually
        # returned by self-masked, depth-backed registration.
        records = self._normal_mode_records(corrected)
        if not records or prepared.image_features is None:
            self.last_geometry_reason = "normal_global_node_geometry_missing"
            return None
        localization = self._metric_localizer.localize(
            f"normal-global-{prepared.frame_id}",
            prepared.observation.rgb,
            prepared.observation.depth_m,
            self._camera_dict(prepared.observation),
            predicted_pose=None,
            frame_index=prepared.frame_id,
            min_frame_gap=0,
            min_independent=1,
            allow_single_strong=False,
            image_features=prepared.image_features,
            place_candidates=[str(record.node_id) for record in records],
            require_cuda_matching=True,
        )
        if localization is None:
            # Feature failure is never evidence of novel space: a true revisit
            # may be textureless, occluded or seen from a different viewpoint.
            self.last_geometry_reason = str(
                self._metric_localizer.last_reason
                or "normal_global_node_geometry_rejected"
            )
            return None
        try:
            candidate_id = int(localization.keyframe_id)
        except (TypeError, ValueError):
            candidate_id = 0
        requested = {int(record.node_id) for record in records}
        if candidate_id not in requested:
            self._block_negative_for_frame(
                "normal",
                prepared.frame_id,
                "normal_global_appearance_unknown:"
                "candidate_contract_violation",
            )
            self.last_geometry_reason = (
                "normal_global_node_candidate_contract_violation"
            )
            return None
        visual_pose = SE2Pose(
            float(localization.x_m), float(localization.y_m),
            math.radians(float(localization.yaw_deg)),
        )
        if not self._pose_hypotheses_agree(visual_pose, corrected):
            self.last_geometry_reason = "normal_global_node_mode_disagreement"
            return None
        self._clear_negative_evidence(
            "normal", "normal_global_positive_rgbd_metric"
        )
        metric = self._metric_observation(prepared, localization)
        independently_confirmed = self._node_metric_has_independent_confirmation(
            self._normal_metric_history, metric
        )
        self._normal_metric_history.append(metric)
        decision = self._normal_global_temporal_pose_bank().observe(
            target_group=(int(metric.candidate_id),),
            predicted_pose=predicted_deg,
            corrected_pose=(
                float(visual_pose.x_m), float(visual_pose.y_m),
                math.degrees(float(visual_pose.yaw_rad)),
            ),
            # C->Q is a node-specific RGB-D measurement of this query frame.
            # The rolling interval only gates its raster mode and may overlap
            # the next probe; it is not the temporal extent of this evidence.
            evidence_interval=(int(prepared.frame_id), int(prepared.frame_id)),
            frame_index=int(prepared.frame_id),
        )
        if not independently_confirmed or not decision.get("pose_correction_safe"):
            self.last_geometry_reason = str(
                decision.get("reason") or "normal_global_node_temporal_pending"
            )
            return None
        corrected_values = decision.get("corrected_pose")
        consensus_pose = (
            None if not isinstance(corrected_values, (list, tuple))
            or len(corrected_values) != 3
            else SE2Pose(
                float(corrected_values[0]), float(corrected_values[1]),
                math.radians(float(corrected_values[2])),
            )
        )
        if consensus_pose is None or not self._pose_hypotheses_agree(
            visual_pose, consensus_pose
        ):
            self.last_geometry_reason = "normal_global_node_temporal_disagreement"
            return None
        hypothesis = ExternalLoopHypothesis(
            metric.hypothesis.candidate_id,
            metric.hypothesis.candidate_to_current,
            metric.hypothesis.covariance,
            verified_graph_bridge=True,
        )
        self._pending_candidate_id = hypothesis.candidate_id
        self.proposals += 1
        self.normal_global_bridges += 1
        self.last_geometry_reason = "normal_global_verified_graph_bridge"
        return hypothesis

    @staticmethod
    def _pose_hypotheses_agree(left: SE2Pose, right: SE2Pose) -> bool:
        return (
            math.hypot(left.x_m - right.x_m, left.y_m - right.y_m)
            <= 0.15
            and abs(_wrap_rad(left.yaw_rad - right.yaw_rad))
            <= math.radians(2.5)
        )

    def _record_for_node(self, node_id: int) -> Optional[_PlaceRecord]:
        return next(
            (
                record for record in self._records
                if int(record.node_id) == int(node_id)
            ),
            None,
        )

    def _recovery_viewpoint_proposals(
        self,
        proposals: list[PlaceProposal],
    ) -> list[PlaceProposal]:
        """Expand an appearance place into nearby historical viewpoints.

        Sequence retrieval identifies a place, not necessarily the one old
        camera pose that overlaps the post-interruption view.  Cover the same
        metric region with a small, deterministic farthest-point sample.  The
        added entries remain proposals only: every one still has to pass RGB-D
        metric registration or the independent GPU geometry validator.
        """

        limit = max(1, int(self.config.recovery_max_candidates))
        seeds = [
            (proposal, self._record_for_node(proposal.node_id))
            for proposal in proposals
        ]
        seeds = [row for row in seeds if row[1] is not None]
        if not seeds:
            return []

        region_m = max(1e-6, float(self.config.candidate_region_m))
        candidates: dict[int, PlaceProposal] = {}
        for proposal, seed in seeds:
            assert seed is not None
            candidates[int(seed.node_id)] = proposal
        for record in self._records:
            nearest = min(
                seeds,
                key=lambda row: math.hypot(
                    record.pose.x_m - row[1].pose.x_m,
                    record.pose.y_m - row[1].pose.y_m,
                ),
            )
            proposal, seed = nearest
            assert seed is not None
            distance = math.hypot(
                record.pose.x_m - seed.pose.x_m,
                record.pose.y_m - seed.pose.y_m,
            )
            if distance > region_m:
                continue
            # The inherited score is diagnostic only.  Distance attenuation
            # keeps original appearance endpoints first without authorizing a
            # nearby viewpoint as a match.
            candidates.setdefault(
                int(record.node_id),
                PlaceProposal(
                    int(record.node_id),
                    float(proposal.score) * (1.0 - 0.25 * distance / region_m),
                    int(proposal.confirmations),
                ),
            )

        selected: list[PlaceProposal] = []
        selected_ids: set[int] = set()
        for proposal, _record in seeds:
            node_id = int(proposal.node_id)
            if node_id not in selected_ids:
                selected.append(proposal)
                selected_ids.add(node_id)
            if len(selected) >= limit:
                return selected

        def viewpoint_distance(
            left: _PlaceRecord, right: _PlaceRecord
        ) -> float:
            translation = math.hypot(
                left.pose.x_m - right.pose.x_m,
                left.pose.y_m - right.pose.y_m,
            )
            # Half a metre per radian balances spatial and view-direction
            # coverage using only the compliant historical camera trajectory.
            return translation + 0.5 * abs(
                _wrap_rad(left.pose.yaw_rad - right.pose.yaw_rad)
            )

        while len(selected) < limit:
            selected_records = [
                self._record_for_node(row.node_id) for row in selected
            ]
            remaining = [
                self._record_for_node(node_id)
                for node_id in candidates
                if node_id not in selected_ids
            ]
            remaining = [row for row in remaining if row is not None]
            if not remaining:
                break
            choice = max(
                remaining,
                key=lambda record: (
                    min(
                        viewpoint_distance(record, old)
                        for old in selected_records
                        if old is not None
                    ),
                    -int(record.node_id),
                ),
            )
            selected.append(candidates[int(choice.node_id)])
            selected_ids.add(int(choice.node_id))
        return selected

    def _propose_recovery_geometry(
        self,
        prepared: PreparedPlaceObservation,
        proposals: list[PlaceProposal],
    ) -> Optional[ExternalLoopHypothesis]:
        del proposals
        return self._propose_frozen_global_geometry(prepared, "recovery")

        # Legacy implementation retained temporarily for source compatibility.
        count = int(self._recovery_usable_observations)
        if count < int(self._recovery_next_geometry_probe):
            self.last_geometry_reason = "recovery_geometry_window_not_ready"
            return None
        self._recovery_next_geometry_probe = (
            count + int(self.config.recovery_probe_spacing)
        )
        target_wall = self._recovery_target_wall
        target_free = self._recovery_target_free
        if len(target_wall) < 16:
            self._clear_negative_evidence(
                "recovery", "recovery_map_geometry_missing"
            )
            self.last_geometry_reason = "recovery_map_geometry_missing"
            return None

        predicted = prepared.predicted_map_pose
        predicted_deg = (
            float(predicted.x_m),
            float(predicted.y_m),
            math.degrees(float(predicted.yaw_rad)),
        )
        candidate_rows: list[
            tuple[PlaceProposal, _PlaceRecord, tuple[float, float, float], str]
        ] = []
        for proposal in proposals[: max(1, int(self.config.recovery_max_candidates))]:
            record = self._record_for_node(proposal.node_id)
            if record is None:
                continue
            candidate_rows.append((
                proposal,
                record,
                (
                    float(record.pose.x_m - predicted.x_m),
                    float(record.pose.y_m - predicted.y_m),
                    wrap_deg(math.degrees(
                        float(record.pose.yaw_rad - predicted.yaw_rad)
                    )),
                ),
                "appearance_place",
            ))

        # Appearance and whole-map geometry are independent proposal sources.
        # Always run both at the same sparse recovery probe: an incorrect but
        # non-empty appearance list must not hide the correct geometric mode,
        # and a second geometric room must remain visible to the ambiguity
        # gate even when appearance strongly favors the first one.
        validator = self._recovery_geometry_engine()
        rolling = validator.rolling_source()
        self.recovery_global_attempts += 1
        global_report = self._recovery_global_geometry_seeder().seed(
            rolling["wall"],
            rolling["free"],
            target_wall,
            target_free,
            pivot=(float(predicted.x_m), float(predicted.y_m)),
        )
        global_candidates = list(global_report.get("candidates") or [])
        self.recovery_global_candidates += len(global_candidates)
        self._recent_recovery_events.append({
            "frame_id": int(prepared.frame_id),
            "candidate_id": 0,
            "candidate_source": "global_frozen_geometry",
            "accepted": False,
            "reason": str(global_report.get("reason") or ""),
            "rolling_interval": [
                rolling.get("first_frame"), rolling.get("last_frame")
            ],
            "rolling_sample_count": int(rolling.get("sample_count") or 0),
            "global_candidates": [dict(row) for row in global_candidates],
            "geometry_backend": global_report.get("geometry_backend"),
        })
        records = list(self._records)
        if global_candidates and records:
            for row in global_candidates[
                : max(1, int(self.config.recovery_max_candidates))
            ]:
                correction = (
                    float(row["dx_m"]),
                    float(row["dy_m"]),
                    float(row["yaw_correction_deg"]),
                )
                corrected_x = float(predicted.x_m) + correction[0]
                corrected_y = float(predicted.y_m) + correction[1]
                corrected_yaw = _wrap_rad(
                    float(predicted.yaw_rad) + math.radians(correction[2])
                )
                # The raster supplies the absolute metric pose; the closest
                # permanent node is only the graph reference used to express
                # candidate->current.  It does not authorize the correction.
                record = min(
                    records,
                    key=lambda item: (
                        math.hypot(
                            item.pose.x_m - corrected_x,
                            item.pose.y_m - corrected_y,
                        )
                        + 0.25 * abs(_wrap_rad(
                            item.pose.yaw_rad - corrected_yaw
                        )),
                        int(item.node_id),
                    ),
                )
                candidate_rows.append((
                    PlaceProposal(
                        int(record.node_id), float(row["score"]), 1
                    ),
                    record,
                    correction,
                    "global_frozen_geometry",
                ))
            self._recovery_last_candidate_frame = int(prepared.frame_id)

        # Only collapse seeds already within the final validator's metric
        # resolution.  Wider separated modes are deliberately retained.
        merged_rows: list[
            tuple[PlaceProposal, _PlaceRecord, tuple[float, float, float], str]
        ] = []
        for row in candidate_rows:
            correction = row[2]
            duplicate = next((
                index for index, old in enumerate(merged_rows)
                if math.hypot(
                    correction[0] - old[2][0],
                    correction[1] - old[2][1],
                ) <= 2.0 * float(self.config.recovery_grid_resolution_m)
                and abs(wrap_deg(correction[2] - old[2][2])) <= 2.5
            ), None)
            if duplicate is None:
                merged_rows.append(row)
            elif row[3] == "global_frozen_geometry":
                # Prefer the raster-derived metric seed over a nearby record
                # origin, while preserving the historical graph reference.
                merged_rows[duplicate] = row
        candidate_rows = merged_rows
        if not candidate_rows:
            disposition = self._evaluate_global_negative(
                "recovery",
                rolling,
                predicted,
                observation_count=count,
                odometry_pose=prepared.odometry_pose,
                query_frame_id=prepared.frame_id,
            )
            if disposition == "READY":
                return None
            if disposition in {"NEGATIVE_PENDING", "POSITIVE"}:
                return None
            if global_candidates and not records:
                self.last_geometry_reason = "recovery_global_reference_missing"
            else:
                self.last_geometry_reason = str(
                    global_report.get("reason")
                    or "recovery_global_and_appearance_search_rejected"
                )
            return None

        accepted: list[tuple[PlaceProposal, _PlaceRecord, dict[str, Any], SE2Pose]] = []
        for proposal, record, visual_correction, candidate_source in candidate_rows:
            self.recovery_geometry_attempts += 1
            report = validator.validate(
                target_wall,
                target_free,
                pivot=(float(predicted.x_m), float(predicted.y_m)),
                visual_correction=visual_correction,
                visual_metric=False,
            )
            self._recent_recovery_events.append({
                "frame_id": int(prepared.frame_id),
                "candidate_id": int(proposal.node_id),
                "candidate_source": candidate_source,
                "appearance_score": float(proposal.score),
                "accepted": bool(report.get("accepted")),
                "reason": str(report.get("reason") or ""),
                "visual_correction": report.get("visual_correction"),
                "geometry_correction": report.get("geometry_correction"),
                "independent_window_count": int(
                    report.get("independent_window_count") or 0
                ),
                "unique_window_count": int(
                    report.get("unique_window_count") or 0
                ),
                "rows": [
                    {
                        "window_frames": int(row.get("window_frames") or 0),
                        "valid": bool(row.get("valid")),
                        "unique_geometry": bool(row.get("unique_geometry")),
                        "uniqueness_blockers": list(
                            row.get("uniqueness_blockers") or []
                        ),
                        "best_prominence_robust": float(
                            row.get("best_prominence_robust") or 0.0
                        ),
                        "top_one_percent_cluster_ratio": float(
                            row.get("top_one_percent_cluster_ratio") or 0.0
                        ),
                        "best_at_search_boundary": bool(
                            row.get("best_at_search_boundary")
                        ),
                        "peak": (
                            dict(row["peaks"][0])
                            if row.get("peaks") else None
                        ),
                    }
                    for row in report.get("rows", [])
                ],
            })
            correction = report.get("fused_correction")
            if (
                not report.get("accepted")
                or not report.get("revisit_evidence_safe")
                or not isinstance(correction, (list, tuple))
                or len(correction) != 3
            ):
                continue
            corrected = SE2Pose(
                float(predicted.x_m) + float(correction[0]),
                float(predicted.y_m) + float(correction[1]),
                _wrap_rad(
                    float(predicted.yaw_rad)
                    + math.radians(float(correction[2]))
                ),
            )
            accepted.append((proposal, record, report, corrected))

        if not accepted:
            disposition = self._evaluate_global_negative(
                "recovery",
                rolling,
                predicted,
                observation_count=count,
                odometry_pose=prepared.odometry_pose,
                query_frame_id=prepared.frame_id,
            )
            if disposition == "UNKNOWN":
                self.last_geometry_reason = (
                    "recovery_geometry_rejected:"
                    + str(self.last_geometry_reason)
                )
            return None
        self._clear_negative_evidence(
            "recovery", "recovery_positive_geometry_mode"
        )
        if any(
            not self._pose_hypotheses_agree(accepted[0][3], row[3])
            for row in accepted[1:]
        ):
            self.recovery_ambiguous_probes += 1
            self.last_geometry_reason = "recovery_competing_geometry_places"
            return None

        proposal, record, report, corrected = accepted[0]
        interval = report.get("evidence_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            self.last_geometry_reason = "recovery_geometry_interval_missing"
            return None
        # The raster/appearance mode identifies a region, not a graph node.
        # Re-run self-masked RGB-D against every historical viewpoint in that
        # region.  Only the node actually returned by metric registration may
        # become endpoint C of a permanent recovery bridge.
        records = self._normal_mode_records(corrected)
        if not records or prepared.image_features is None:
            self.last_geometry_reason = "recovery_node_geometry_missing"
            return None
        localization = self._metric_localizer.localize(
            f"recovery-global-{prepared.frame_id}",
            prepared.observation.rgb,
            prepared.observation.depth_m,
            self._camera_dict(prepared.observation),
            predicted_pose=None,
            frame_index=prepared.frame_id,
            min_frame_gap=0,
            min_independent=1,
            allow_single_strong=False,
            image_features=prepared.image_features,
            place_candidates=[str(row.node_id) for row in records],
            require_cuda_matching=True,
        )
        if localization is None:
            self.last_geometry_reason = str(
                self._metric_localizer.last_reason
                or "recovery_node_geometry_rejected"
            )
            return None
        try:
            candidate_id = int(localization.keyframe_id)
        except (TypeError, ValueError):
            candidate_id = 0
        requested = {int(record.node_id) for record in records}
        if candidate_id not in requested:
            self._block_negative_for_frame(
                "recovery",
                prepared.frame_id,
                "recovery_global_appearance_unknown:"
                "candidate_contract_violation",
            )
            self.last_geometry_reason = (
                "recovery_node_candidate_contract_violation"
            )
            return None
        visual_pose = SE2Pose(
            float(localization.x_m), float(localization.y_m),
            math.radians(float(localization.yaw_deg)),
        )
        if not self._pose_hypotheses_agree(visual_pose, corrected):
            self.last_geometry_reason = "recovery_node_mode_disagreement"
            return None
        metric = self._metric_observation(prepared, localization)
        independently_confirmed = self._node_metric_has_independent_confirmation(
            self._recovery_node_metric_history, metric
        )
        self._recovery_node_metric_history.append(metric)
        decision = self._recovery_temporal_pose_bank().observe(
            target_group=(int(metric.candidate_id),),
            predicted_pose=predicted_deg,
            corrected_pose=(
                float(visual_pose.x_m), float(visual_pose.y_m),
                math.degrees(float(visual_pose.yaw_rad)),
            ),
            evidence_interval=(int(prepared.frame_id), int(prepared.frame_id)),
            frame_index=int(prepared.frame_id),
        )
        if not independently_confirmed or not decision.get("pose_correction_safe"):
            self.last_geometry_reason = str(
                decision.get("reason")
                or "recovery_node_temporal_pending"
            )
            return None
        corrected_values = decision.get("corrected_pose")
        consensus_pose = (
            None if not isinstance(corrected_values, (list, tuple))
            or len(corrected_values) != 3
            else SE2Pose(
                float(corrected_values[0]), float(corrected_values[1]),
                math.radians(float(corrected_values[2])),
            )
        )
        if consensus_pose is None or not self._pose_hypotheses_agree(
            visual_pose, consensus_pose
        ):
            self.last_geometry_reason = "recovery_node_temporal_disagreement"
            return None
        hypothesis = ExternalLoopHypothesis(
            metric.hypothesis.candidate_id,
            metric.hypothesis.candidate_to_current,
            metric.hypothesis.covariance,
            recovery_relocalization=True,
        )
        self._pending_candidate_id = int(hypothesis.candidate_id)
        self.recovery_geometry_consensus += 1
        self.proposals += 1
        self.last_geometry_reason = "recovery_temporal_geometry_accepted"
        return hypothesis

    @staticmethod
    def _matching_query_terminal(
        prepared: PreparedPlaceObservation,
        result: MapResult,
        expected_scope: QueryScope,
    ) -> bool:
        generation = int(getattr(prepared, "query_generation", 0))
        return bool(
            generation > 0
            and result.query_scope is expected_scope
            and int(result.query_generation) == generation
        )

    def _native_novelty_reconciliation_pending(self) -> bool:
        return int(getattr(
            self,
            "_native_novelty_resume_reconciliation_updates_remaining",
            0,
        )) > 0

    def _advance_native_novelty_reconciliation(
        self,
        prepared: PreparedPlaceObservation,
        result: MapResult,
    ) -> None:
        """Mirror native's authoritative post-novelty viewpoint window."""

        outcome = getattr(result, "query_outcome", QueryOutcome.NONE)
        result_scope = getattr(result, "query_scope", QueryScope.NONE)
        result_generation = int(getattr(result, "query_generation", 0))
        prepared_generation = int(getattr(prepared, "query_generation", 0))
        remaining = max(
            0,
            int(getattr(
                result,
                "novelty_resume_reconciliation_viewpoints_remaining",
                0,
            )),
        )
        if (
            outcome is QueryOutcome.NOVELTY_RESUMED
            and result_scope is QueryScope.NORMAL
            and result_generation > 0
            and result_generation == prepared_generation
        ):
            self._native_novelty_resume_reconciliation_generation = (
                result_generation
            )
        self._native_novelty_resume_reconciliation_updates_remaining = remaining
        if remaining == 0:
            self._native_novelty_resume_reconciliation_generation = 0

    def _discard_held_query_metric_measurements(
        self, scope: QueryScope
    ) -> None:
        """Forget C->Q measurements after a non-terminal native response.

        Candidate ids remain warm retrieval hints, but no transform or temporal
        confirmation from the rejected frame may be reused on the next frame.
        """

        history_names = (
            ("_normal_metric_history",)
            if scope is QueryScope.NORMAL
            else ("_metric_history", "_recovery_node_metric_history")
        )
        for name in history_names:
            history = getattr(self, name, None)
            if history is not None:
                history.clear()
        bank_name = (
            "_normal_global_pose_bank"
            if scope is QueryScope.NORMAL
            else "_recovery_pose_bank"
        )
        bank = getattr(self, bank_name, None)
        if bank is not None:
            bank.clear()
        resolver_scope = (
            "normal" if scope is QueryScope.NORMAL else "recovery"
        )
        self._reset_frozen_resolver(
            resolver_scope,
            keep_warm=True,
            force_next_probe=True,
            replenish_confirmation_budget=False,
        )

    def finish(
        self,
        prepared: PreparedPlaceObservation,
        result: MapResult,
        proposed_hypothesis: Optional[ExternalLoopHypothesis],
    ) -> None:
        self._advance_native_novelty_reconciliation(prepared, result)
        if (
            not result.mapping_active
            or not result.soft_localization
            or result.uncertain_hold
        ):
            # Native owns the transition out of a verified known-place
            # localization window. Building means it proved sustained novelty;
            # uncertain hold means global identity must be resolved again.
            self._verified_read_only_continuation = False
        elif (
            result.tracking_ok
            and result.localized
            and result.read_only_match
        ):
            # ``localized`` is a pulse, while KnownLocalizing continues as
            # soft+read-only on later frames. Remember the trusted transition
            # so those continuation frames are not mistaken for a fresh
            # unresolved candidate hold.
            self._verified_read_only_continuation = True
        query_retry_pending = bool(
            result.query_outcome is QueryOutcome.HOLDING
            and (
                self._matching_query_terminal(
                    prepared, result, QueryScope.RECOVERY
                )
                or self._matching_query_terminal(
                    prepared, result, QueryScope.NORMAL
                )
            )
        )
        if query_retry_pending and proposed_hypothesis is not None:
            self._discard_held_query_metric_measurements(result.query_scope)
        proposed_node_id = (
            0 if proposed_hypothesis is None else proposed_hypothesis.candidate_id
        )
        if (
            proposed_node_id > 0
            and result.ref_node_id > 0
            and not query_retry_pending
        ):
            accepted = bool(result.loop_closed or result.localized)
            if accepted:
                self.accepted_proposals += 1
                self._cool_region(proposed_node_id, self.config.accepted_region_cooldown)
            else:
                self.rejected_proposals += 1
                self._cool_region(
                    proposed_node_id, self.config.rejected_candidate_cooldown
                )
        self._pending_candidate_id = 0

        self._update_poses(result)
        self._update_recovery_map_geometry(result)
        if (
            prepared.structural_usable
            and result.map_updated
            and result.ref_node_id > 0
        ):
            self._commit(
                result.ref_node_id,
                result.current_pose,
                prepared,
            )
        if bool(getattr(self, "_recovery_pending", False)):
            self._finish_recovery_observation(
                prepared, result, proposed_hypothesis
            )
        normal_pending = bool(
            getattr(self, "_normal_global_pending", False)
        )
        if normal_pending:
            query_terminal = self._matching_query_terminal(
                prepared,
                result,
                QueryScope.NORMAL,
            ) and result.query_outcome in {
                QueryOutcome.COMMITTED,
                QueryOutcome.ALL_KNOWN,
                QueryOutcome.BRIDGE_ONLY_DISCARDED,
                QueryOutcome.NOVELTY_RESUMED,
            }
            verified_bridge = bool(
                proposed_hypothesis is not None
                and proposed_hypothesis.verified_graph_bridge
            )
            if query_terminal:
                if result.query_outcome is QueryOutcome.NOVELTY_RESUMED:
                    self.normal_global_native_novelty_resumes += 1
                    terminal_reason = "normal_query_native_novelty_resumed"
                else:
                    terminal_reason = "normal_query_native_acknowledged"
                self._end_normal_global(terminal_reason)
            elif verified_bridge and query_retry_pending:
                # HOLDING means native has not produced a terminal decision for
                # this exact transaction. Preserve the physical-region
                # shortlist and retry budget, but never the C->Q transform or
                # its consensus votes: propose() invokes RGB-D localization
                # again on the next frame.
                self.last_geometry_reason = "normal_graph_bridge_query_holding"
            elif verified_bridge and not (result.loop_closed or result.localized):
                # Native graph validation is the final authority. A rejected
                # transaction invalidates the complete temporal bridge
                # proposal; retaining either vote or its warm shortlist would
                # merely resubmit the same rejected evidence next frame.
                history = getattr(self, "_normal_metric_history", None)
                if history is not None:
                    history.clear()
                self._reset_frozen_resolver(
                    "normal",
                    keep_warm=True,
                    force_next_probe=True,
                    replenish_confirmation_budget=False,
                )
                self._normal_retry_proposals = []
                self._normal_retry_attempts_remaining = 0
                bank = getattr(self, "_normal_global_pose_bank", None)
                if bank is not None:
                    bank.clear()
                self._clear_negative_evidence(
                    "normal", "normal_graph_bridge_native_rejected"
                )
                self.last_geometry_reason = (
                    "normal_graph_bridge_native_rejected"
                )
        elif self._normal_global_should_begin(result):
            self._begin_normal_global(result)
        prepared_odometry = getattr(prepared, "odometry_pose", None)
        if result.tracking_ok and prepared_odometry is not None:
            self._last_result_odometry_pose = prepared_odometry
            self._last_result_map_pose = result.current_pose
        if not result.mapping_active and not self._frozen:
            if self._centers is None:
                self._fit_vocabulary()
                if self._centers is not None:
                    self.forced_index_builds += 1
            self._frozen = True
            self._metric_localizer.seal_place_database()
            self._reset_query_sequence()
            for record in self._records:
                record.sampled_descriptors = None

    @staticmethod
    def _map_geometry(result: MapResult) -> tuple[np.ndarray, np.ndarray]:
        if result.occupancy.ndim != 2 or not result.occupancy.size:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy()
        occupancy = np.asarray(result.occupancy)
        low = np.asarray(result.low_obstacles, dtype=bool)
        high = np.asarray(result.high_obstacles, dtype=bool)
        wall = low & (occupancy >= 65)
        free = (occupancy == 0) & ~low & ~high

        def points(mask: np.ndarray) -> np.ndarray:
            rows, columns = np.nonzero(mask)
            if not len(rows):
                return np.empty((0, 2), dtype=np.float32)
            return np.column_stack((
                float(result.x_min_m)
                + (columns.astype(np.float64) + 0.5)
                * float(result.cell_size_m),
                float(result.y_min_m)
                + (rows.astype(np.float64) + 0.5)
                * float(result.cell_size_m),
            )).astype(np.float32, copy=False)

        return points(wall), points(free)

    def _normal_global_should_begin(self, result: MapResult) -> bool:
        return bool(
            result.mapping_active
            and result.soft_localization
            and not result.recovery_hold
            and not result.localized
            and (result.read_only_match or result.uncertain_hold)
            and bool(getattr(self, "_records", ()))
            and not getattr(self, "_recovery_pending", False)
            and not getattr(
                self, "_verified_read_only_continuation", False
            )
            and not self._native_novelty_reconciliation_pending()
        )

    def _begin_normal_global(self, result: MapResult) -> None:
        wall, free = self._map_geometry(result)
        if len(wall) < 16 or not len(free):
            return
        # These arrays are immutable snapshots of the last map response at
        # hold entry. finish() never refreshes them while pending.
        self._normal_target_wall = np.array(wall, dtype=np.float32, copy=True)
        self._normal_target_free = np.array(free, dtype=np.float32, copy=True)
        self._normal_allowed_node_ids = self._historical_place_node_ids()
        self._normal_hold_generation = self._next_query_transaction_id()
        self._normal_place_database_generation = int(
            getattr(self._metric_localizer, "place_database_generation", 0)
        )
        self._verified_read_only_continuation = False
        self._normal_global_pending = True
        self._normal_global_usable_observations = 0
        self._normal_global_next_probe = max(
            self.config.recovery_geometry_windows
        )
        self._clear_negative_evidence("normal", "normal_global_hold_started")
        self._freeze_negative_target("normal")
        validator = getattr(self, "_normal_global_validator", None)
        if validator is not None:
            validator.clear()
        bank = getattr(self, "_normal_global_pose_bank", None)
        if bank is not None:
            bank.clear()
        history = getattr(self, "_normal_metric_history", None)
        if history is None:
            self._normal_metric_history = deque(maxlen=8)
        else:
            history.clear()
        self._reset_frozen_resolver("normal")
        self._normal_retry_proposals = []
        self._normal_retry_attempts_remaining = 0
        self.last_geometry_reason = "normal_global_hold_started"

    def _end_normal_global(self, reason: str) -> None:
        self._clear_negative_evidence("normal", str(reason))
        self._normal_global_pending = False
        self._normal_global_usable_observations = 0
        self._normal_global_next_probe = 0
        self._normal_target_wall = np.empty((0, 2), dtype=np.float32)
        self._normal_target_free = np.empty((0, 2), dtype=np.float32)
        self._normal_negative_target = None
        self._normal_allowed_node_ids = frozenset()
        self._reset_frozen_resolver("normal")
        self._normal_resolver_confirmation_budget_remaining = 0
        validator = getattr(self, "_normal_global_validator", None)
        if validator is not None:
            validator.clear()
        bank = getattr(self, "_normal_global_pose_bank", None)
        if bank is not None:
            bank.clear()
        history = getattr(self, "_normal_metric_history", None)
        if history is not None:
            history.clear()
        self._normal_retry_proposals = []
        self._normal_retry_attempts_remaining = 0
        self.last_geometry_reason = str(reason)

    def _update_recovery_map_geometry(self, result: MapResult) -> None:
        # A recovery proof is meaningful only against one immutable historical
        # target. The current recovery frame is read-only, so refreshing here
        # would add no valid information and would silently change the digest.
        if bool(getattr(self, "_recovery_pending", False)):
            return
        wall, free = self._map_geometry(result)
        if not len(wall) and not len(free):
            return
        self._recovery_target_wall = wall
        self._recovery_target_free = free

    def _finish_recovery_observation(
        self,
        prepared: PreparedPlaceObservation,
        result: MapResult,
        proposed_hypothesis: Optional[ExternalLoopHypothesis],
    ) -> None:
        if self._matching_query_terminal(
            prepared,
            result,
            QueryScope.RECOVERY,
        ) and result.query_outcome in {
            QueryOutcome.COMMITTED,
            QueryOutcome.ALL_KNOWN,
            QueryOutcome.BRIDGE_ONLY_DISCARDED,
        }:
            self._end_recovery("recovery_query_native_acknowledged")
            return
        recovery_metric = bool(
            proposed_hypothesis is not None
            and proposed_hypothesis.recovery_relocalization
        )
        # Localization is useful evidence, but it is not the commit record for
        # the provisional query. The hold ends only through the generation-
        # bound terminal acknowledgement above, after native graph insertion
        # and aggregate handling have both reached a safe outcome.
        if result.localized:
            self.recovery_localizations += 1
            if not recovery_metric:
                self.recovery_native_localizations += 1

        fused = result.fused_odometry_pose
        previous = self._recovery_last_fused_pose
        if fused is not None and previous is not None:
            relative = previous.relative_to(fused)
            translation = math.hypot(relative.x_m, relative.y_m)
            # A single discontinuity is a reset/anchor, not exploration.
            if translation <= 0.5 * float(self.config.maximum_depth_m):
                self._recovery_observed_translation_m += translation
        if fused is not None:
            anchor_fused = getattr(self, "_recovery_anchor_fused_pose", None)
            anchor_map = getattr(self, "_recovery_anchor_map_pose", None)
            if anchor_fused is None:
                self._recovery_anchor_fused_pose = fused
                anchor_fused = fused
            if anchor_map is None:
                anchor_map = prepared.predicted_map_pose
                self._recovery_anchor_map_pose = anchor_map
            self._recovery_last_local_map_pose = anchor_map.compose(
                anchor_fused.relative_to(fused)
            )
            self._recovery_last_fused_pose = fused
        prepared_odometry = getattr(prepared, "odometry_pose", None)
        if prepared_odometry is not None:
            self._recovery_last_prepared_odometry_pose = prepared_odometry

        # UNKNOWN is not evidence of a novel place. In particular, travel,
        # elapsed observations, an appearance miss, an ambiguous global mode,
        # or a CUDA/registration failure cannot authorize writes to the
        # committed map. A future NOVEL transition must carry two independent,
        # exhaustive no-mode certificates; until then the tentative recovery
        # chain remains read-only.

    def _end_recovery(self, reason: str) -> None:
        self._recovery_pending = False
        self._recovery_usable_observations = 0
        self._recovery_next_geometry_probe = 0
        self._recovery_last_fused_pose = None
        self._recovery_anchor_fused_pose = None
        self._recovery_anchor_map_pose = None
        self._recovery_last_local_map_pose = None
        self._recovery_last_prepared_odometry_pose = None
        self._recovery_observed_translation_m = 0.0
        self._recovery_retry_proposals = []
        self._recovery_retry_attempts_remaining = 0
        self._clear_recovery_geometry_evidence(str(reason))
        self._recovery_negative_target = None
        self._recovery_allowed_node_ids = frozenset()
        self._reset_frozen_resolver("recovery")
        self._recovery_resolver_confirmation_budget_remaining = 0
        self._reset_query_sequence()
        self.last_geometry_reason = str(reason)

    def _reset_query_sequence(self) -> None:
        self._query_histograms.clear()
        self._recent_rankings.clear()
        self._metric_history.clear()
        self._pending_candidate_id = 0
        self._metric_localizer.reset_query_sequence()

    def _structural_descriptors(
        self, observation: OfficialObservation
    ) -> tuple[Optional[ImageFeatures], np.ndarray]:
        import cv2

        gray = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2GRAY)
        features = self._extractor.extract(gray)
        if features is None or features.descriptors is None:
            return None, np.zeros((0, 128), dtype=np.float32)
        pixels = np.asarray(features.pixels, dtype=np.float64)
        descriptors = np.asarray(features.descriptors, dtype=np.float32)
        localization_features = ImageFeatures(
            pixels=np.ascontiguousarray(pixels, dtype=np.float64),
            descriptors=np.ascontiguousarray(descriptors, dtype=np.float32),
            backend=str(features.backend),
        )
        columns = np.rint(pixels[:, 0]).astype(np.int64)
        rows = np.rint(pixels[:, 1]).astype(np.int64)
        height, width = observation.depth_m.shape
        valid = (
            (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        depth = np.zeros(len(pixels), dtype=np.float64)
        selected = np.flatnonzero(valid)
        if len(selected):
            depth[selected] = observation.depth_m[rows[selected], columns[selected]]
        valid &= np.isfinite(depth)
        valid &= depth >= self.config.minimum_depth_m
        valid &= depth <= self.config.maximum_depth_m
        intrinsics = observation.intrinsics
        camera_points = np.column_stack(
            (
                (pixels[:, 0] - intrinsics.cx) / intrinsics.fx * depth,
                (pixels[:, 1] - intrinsics.cy) / intrinsics.fy * depth,
                depth,
            )
        )
        local = observation.camera_relative_pose.rtabmap_local_transform()
        robot_points = camera_points @ local[:, :3].T + local[:, 3]
        valid &= np.all(np.isfinite(robot_points), axis=1)
        valid &= robot_points[:, 2] > self.config.minimum_feature_height_m
        # Exclude the robot's own hands/body using the same compliant base-frame
        # envelope as the mapper; those features move independently of a place.
        valid &= ~(
            (robot_points[:, 0] <= 0.95)
            & (np.abs(robot_points[:, 1]) <= 0.55)
            & (robot_points[:, 2] > 0.08)
        )
        masked_descriptors = np.ascontiguousarray(
            descriptors[valid], dtype=np.float32
        )
        # Mapping-range descriptors remain the bounded appearance/BoW input.
        # Long-term RGB-D localization owns a separate 0.45-8 m stable-depth
        # gate, so it must see the same raw SIFT set for both history and query.
        return localization_features, masked_descriptors

    def _sample(self, descriptors: np.ndarray) -> np.ndarray:
        count = self.config.descriptors_per_observation
        if len(descriptors) <= count:
            return np.ascontiguousarray(descriptors, dtype=np.float32)
        indices = np.linspace(0, len(descriptors) - 1, count, dtype=np.int64)
        return np.ascontiguousarray(descriptors[indices], dtype=np.float32)

    def _commit(
        self,
        node_id: int,
        pose: SE2Pose,
        prepared: PreparedPlaceObservation,
    ) -> None:
        descriptors = prepared.descriptors
        if self._frozen or not len(descriptors):
            return
        if any(record.node_id == int(node_id) for record in self._records):
            return
        if len(self._records) >= self.config.maximum_representatives:
            return
        if not representative_is_novel(pose, self._records, self.config):
            return
        sample = self._sample(descriptors)
        self._records.append(
            _PlaceRecord(
                int(node_id), pose, float(prepared.path_progress_m), sample
            )
        )
        observation = prepared.observation
        self._metric_localizer.add_place_keyframe(
            str(int(node_id)),
            observation.rgb,
            observation.depth_m,
            self._camera_dict(observation),
            (pose.x_m, pose.y_m, math.degrees(pose.yaw_rad)),
            frame_index=prepared.frame_id,
            force=True,
            image_features=prepared.image_features,
        )
        if self._centers is not None:
            self._histograms.append(self._histogram(sample))
        if len(self._records) >= self._next_refit_size:
            self._fit_vocabulary()
            self._next_refit_size *= 2

    def _fit_vocabulary(self) -> None:
        samples = [
            record.sampled_descriptors
            for record in self._records
            if record.sampled_descriptors is not None
            and len(record.sampled_descriptors)
        ]
        if not samples:
            return
        values = self._functional.normalize(
            self._torch.from_numpy(
                np.ascontiguousarray(np.concatenate(samples), dtype=np.float32)
            ).to(device=self.device),
            dim=1,
        )
        words = min(self.config.visual_words, int(values.shape[0]))
        initial = self._torch.linspace(
            0, int(values.shape[0]) - 1, steps=words, device=self.device
        ).round().long()
        centers = values[initial].clone()
        with self._torch.inference_mode():
            for _ in range(self.config.kmeans_iterations):
                sums = self._torch.zeros_like(centers)
                counts = self._torch.zeros(
                    words, device=self.device, dtype=self._torch.float32
                )
                for first in range(0, int(values.shape[0]), 4096):
                    batch = values[first:first + 4096]
                    assignment = self._torch.argmax(batch @ centers.T, dim=1)
                    sums.index_add_(0, assignment, batch)
                    counts.index_add_(
                        0,
                        assignment,
                        self._torch.ones(len(batch), device=self.device),
                    )
                occupied = counts > 0
                centers[occupied] = sums[occupied] / counts[occupied, None]
                centers = self._functional.normalize(centers, dim=1)
        self._centers = centers
        self._histograms = [
            self._histogram(record.sampled_descriptors)
            for record in self._records
            if record.sampled_descriptors is not None
        ]
        self._query_histograms.clear()
        self._recent_rankings.clear()

    def _histogram(self, descriptors: np.ndarray) -> Any:
        histogram = self._torch.zeros(
            int(self._centers.shape[0]),
            device=self.device,
            dtype=self._torch.float32,
        )
        if len(descriptors):
            values = self._functional.normalize(
                self._torch.from_numpy(
                    np.ascontiguousarray(descriptors, dtype=np.float32)
                ).to(device=self.device),
                dim=1,
            )
            assignment = self._torch.argmax(values @ self._centers.T, dim=1)
            histogram = self._torch.bincount(
                assignment, minlength=int(self._centers.shape[0])
            ).float()
        return histogram

    def _rank(self, current_path_progress_m: float) -> list[tuple[int, float]]:
        history = self._torch.stack(self._histograms)
        query = self._torch.stack(tuple(self._query_histograms))
        document_frequency = self._torch.count_nonzero(history > 0, dim=0).float()
        inverse_frequency = self._torch.log(
            (float(history.shape[0]) + 1.0) / (document_frequency + 1.0)
        ) + 1.0
        history = self._functional.normalize(
            self._torch.log1p(history) * inverse_frequency, dim=1
        )
        query = self._functional.normalize(
            self._torch.log1p(query) * inverse_frequency, dim=1
        )
        similarity = history @ query.T
        endpoint_scores = _sequence_endpoint_scores(
            similarity, self.config.sequence_window
        )
        eligible = np.asarray(
            [
                current_path_progress_m - record.path_progress_m
                >= self.config.minimum_motion_separation_m
                and self._cooldowns.get(record.node_id, 0) <= 0
                for record in self._records
            ],
            dtype=bool,
        )
        scores = endpoint_scores.detach().cpu().numpy().astype(np.float64)
        valid_scores = scores[eligible & np.isfinite(scores) & (scores >= 0.0)]
        if not len(valid_scores):
            return []
        median = float(np.median(valid_scores))
        mad = float(np.median(np.abs(valid_scores - median)))
        minimum = max(
            self.config.minimum_sequence_score,
            median + max(self.config.minimum_robust_separation, 3.0 * mad),
        )
        order = np.argsort(-scores, kind="stable")
        selected: list[tuple[int, float]] = []
        for index in order:
            index = int(index)
            if not eligible[index] or scores[index] < minimum:
                continue
            if any(
                abs(index - old_index) < self.config.independent_candidate_gap
                for old_index, _old_score in selected
            ):
                continue
            selected.append((index, float(scores[index])))
            if len(selected) >= self.config.maximum_candidates:
                break
        return selected

    def _same_region(self, left_index: int, right_index: int) -> bool:
        if abs(left_index - right_index) <= self.config.sequence_window * 2:
            return True
        left = self._records[left_index].pose
        right = self._records[right_index].pose
        return math.hypot(left.x_m - right.x_m, left.y_m - right.y_m) <= (
            self.config.candidate_region_m
        )

    def _confirmed_proposals(
        self, ranked: list[tuple[int, float]]
    ) -> list[PlaceProposal]:
        proposals = []
        for index, score in ranked:
            confirmations = 1
            for previous in list(self._recent_rankings)[:-1]:
                if any(self._same_region(index, old) for old in previous):
                    confirmations += 1
            if confirmations >= self.config.confirmation_queries:
                proposals.append(PlaceProposal(
                    self._records[index].node_id, score, confirmations
                ))
        return proposals

    def _confirmed_proposal(
        self, ranked: list[tuple[int, float]]
    ) -> Optional[PlaceProposal]:
        proposals = self._confirmed_proposals(ranked)
        return proposals[0] if proposals else None

    @staticmethod
    def _camera_dict(observation: OfficialObservation) -> dict[str, object]:
        intrinsics = observation.intrinsics
        relative = observation.camera_relative_pose
        return {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "robot_relative_pose": {
                "pos": list(relative.position_m),
                "quat": list(relative.quaternion_xyzw),
            },
        }

    def _metric_observation(
        self,
        prepared: PreparedPlaceObservation,
        localization: VisualLocalization,
    ) -> _MetricObservation:
        candidate_id = int(localization.keyframe_id)
        rmse = max(0.0, float(localization.rmse_m))
        inliers = max(1, int(localization.inliers))
        ratio = max(0.05, min(1.0, float(localization.inlier_ratio)))
        sample_scale = max(1.0, math.sqrt(20.0 / float(inliers)))
        translation_sigma = float(np.clip(
            max(
                self.config.metric_translation_sigma_floor_m,
                rmse * sample_scale / math.sqrt(ratio),
            ),
            self.config.metric_translation_sigma_floor_m,
            self.config.metric_translation_sigma_ceiling_m,
        ))
        yaw_sigma = float(np.clip(
            max(
                self.config.metric_yaw_sigma_floor_rad,
                translation_sigma / 0.75,
            ),
            self.config.metric_yaw_sigma_floor_rad,
            self.config.metric_yaw_sigma_ceiling_rad,
        ))
        registration_translation_sigma = float(
            getattr(localization, "translation_std_m", math.inf)
        )
        registration_yaw_sigma = math.radians(float(
            getattr(localization, "yaw_std_deg", math.inf)
        ))
        if (
            not math.isfinite(registration_translation_sigma)
            or registration_translation_sigma < 0.0
        ):
            registration_translation_sigma = math.inf
        if (
            not math.isfinite(registration_yaw_sigma)
            or registration_yaw_sigma < 0.0
        ):
            registration_yaw_sigma = math.inf
        # RGB-D RANSAC maps current-frame points into the historical keyframe.
        # In pose notation this is candidate->current: the current base pose
        # expressed in the candidate base frame.
        candidate_to_current = SE2Pose(
            float(localization.keyframe_to_current_x_m),
            float(localization.keyframe_to_current_y_m),
            math.radians(float(localization.keyframe_to_current_yaw_deg)),
        )
        hypothesis = ExternalLoopHypothesis(
            candidate_id,
            candidate_to_current,
            (
                translation_sigma ** 2, 0.0, 0.0,
                0.0, translation_sigma ** 2, 0.0,
                0.0, 0.0, yaw_sigma ** 2,
            ),
        )
        return _MetricObservation(
            prepared.frame_id,
            candidate_id,
            prepared.odometry_pose,
            SE2Pose(
                float(localization.x_m),
                float(localization.y_m),
                math.radians(float(localization.yaw_deg)),
            ),
            hypothesis,
            inliers,
            ratio,
            rmse,
            registration_translation_sigma,
            registration_yaw_sigma,
        )

    @staticmethod
    def _metric_map_covariance(metric: _MetricObservation) -> np.ndarray:
        """Rotate candidate-frame RGB-D covariance into map coordinates."""

        covariance = np.asarray(
            metric.hypothesis.covariance, dtype=np.float64
        ).reshape(3, 3)
        candidate_yaw = _wrap_rad(
            float(metric.map_pose.yaw_rad)
            - float(metric.hypothesis.candidate_to_current.yaw_rad)
        )
        c = math.cos(candidate_yaw)
        s = math.sin(candidate_yaw)
        rotate = np.asarray(
            ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        return rotate @ covariance @ rotate.T

    @staticmethod
    def _compose_left_covariance_jacobian(
        left: SE2Pose, relative: SE2Pose
    ) -> np.ndarray:
        """Jacobian of ``left.compose(relative)`` with respect to ``left``."""

        c = math.cos(float(left.yaw_rad))
        s = math.sin(float(left.yaw_rad))
        x = float(relative.x_m)
        y = float(relative.y_m)
        return np.asarray(
            (
                (1.0, 0.0, -s * x - c * y),
                (0.0, 1.0, c * x - s * y),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

    def _pose_innovation_mahalanobis_sq(
        self,
        previous: _MetricObservation,
        current: _MetricObservation,
        odometry_motion: SE2Pose,
    ) -> float:
        """Score two map-pose measurements under one odometry motion model."""

        # Explicitly form map<-odometry, then evaluate it at the current query.
        # This is equivalent to transporting the previous map pose by the
        # inter-query odometry, while making the correction convention clear.
        correction = _map_to_odometry_correction(
            previous.map_pose, previous.odometry_pose
        )
        modeled_current_odometry = previous.odometry_pose.compose(
            odometry_motion
        )
        predicted = correction.compose(modeled_current_odometry)
        innovation = predicted.relative_to(current.map_pose)
        residual = np.asarray(
            (innovation.x_m, innovation.y_m, innovation.yaw_rad),
            dtype=np.float64,
        )

        previous_covariance = self._metric_map_covariance(previous)
        current_covariance = self._metric_map_covariance(current)
        jacobian = self._compose_left_covariance_jacobian(
            previous.map_pose, odometry_motion
        )
        predicted_covariance = (
            jacobian @ previous_covariance @ jacobian.T
        )

        c = math.cos(float(predicted.yaw_rad))
        s = math.sin(float(predicted.yaw_rad))
        world_to_predicted = np.asarray(
            ((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        covariance = world_to_predicted @ (
            predicted_covariance + current_covariance
        ) @ world_to_predicted.T
        # Existing inter-query slacks are treated as three-sigma process
        # uncertainty. The resulting confidence region therefore scales with
        # sensor noise instead of a particular room or trajectory.
        process_sigma = np.asarray(
            (
                float(self.config.metric_interquery_translation_slack_m) / 3.0,
                float(self.config.metric_interquery_translation_slack_m) / 3.0,
                float(self.config.metric_interquery_yaw_slack_rad) / 3.0,
            ),
            dtype=np.float64,
        )
        covariance += np.diag(np.square(process_sigma))
        covariance += np.eye(3, dtype=np.float64) * 1e-12
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(covariance, hermitian=True) @ residual
        return float(residual @ solved)

    def _metric_pair_consistency_model(
        self,
        previous: _MetricObservation,
        current: _MetricObservation,
    ) -> Optional[str]:
        """Return the supported inter-query motion model, if any."""

        if int(previous.frame_id) == int(current.frame_id):
            return None
        odometry_motion = previous.odometry_pose.relative_to(
            current.odometry_pose
        )
        score = self._pose_innovation_mahalanobis_sq(
            previous, current, odometry_motion
        )

        # base_qvel reports commanded body velocity, so contact can claim a
        # large turn while the camera remains still. Two independent RGB-D
        # map poses may establish that stationary model; a single image never
        # gets this exception.
        measured_motion = previous.map_pose.relative_to(current.map_pose)
        observed_stationary = (
            math.hypot(measured_motion.x_m, measured_motion.y_m)
            <= float(self.config.metric_interquery_translation_slack_m)
            and abs(float(measured_motion.yaw_rad))
            <= float(self.config.metric_interquery_yaw_slack_rad)
        )
        odometry_disagrees = (
            math.hypot(odometry_motion.x_m, odometry_motion.y_m)
            > math.hypot(measured_motion.x_m, measured_motion.y_m)
            + float(self.config.metric_interquery_translation_slack_m)
            or abs(float(odometry_motion.yaw_rad))
            > abs(float(measured_motion.yaw_rad))
            + float(self.config.metric_interquery_yaw_slack_rad)
        )
        if observed_stationary and odometry_disagrees:
            stationary_score = self._pose_innovation_mahalanobis_sq(
                previous, current, SE2Pose()
            )
            if stationary_score <= float(self.config.metric_consistency_chi2):
                return "stationary"
        if score <= float(self.config.metric_consistency_chi2):
            return "moving"
        return None

    def _metric_pair_is_consistent(
        self,
        previous: _MetricObservation,
        current: _MetricObservation,
    ) -> bool:
        """Check a pair without requiring the historical node IDs to match."""

        return self._metric_pair_consistency_model(previous, current) is not None

    def _confirmed_metric_hypothesis(
        self, current: _MetricObservation
    ) -> Optional[ExternalLoopHypothesis]:
        current_record = next(
            (
                record for record in self._records
                if record.node_id == current.candidate_id
            ),
            None,
        )
        confirmations = 1
        for previous in list(self._metric_history)[:-1]:
            previous_record = next(
                (
                    record for record in self._records
                    if record.node_id == previous.candidate_id
                ),
                None,
            )
            if current_record is None or previous_record is None:
                continue
            candidate_separation = math.hypot(
                current_record.pose.x_m - previous_record.pose.x_m,
                current_record.pose.y_m - previous_record.pose.y_m,
            )
            if candidate_separation > 2.0 * self.config.candidate_region_m:
                continue
            model = self._metric_pair_consistency_model(previous, current)
            if model is None:
                continue
            if not self._metric_queries_have_independent_viewpoints(
                previous, current, consistency_model=model
            ):
                continue
            confirmations += 1
        if confirmations < self.config.metric_confirmation_queries:
            return None
        return current.hypothesis

    def _update_poses(self, result: MapResult) -> None:
        if not result.poses:
            return
        poses = {
            record.node_id: SE2Pose(record.x_m, record.y_m, record.yaw_rad)
            for record in result.poses
        }
        for record in self._records:
            if record.node_id in poses:
                record.pose = poses[record.node_id]
        self._metric_localizer.update_keyframe_poses({
            str(node_id): (pose.x_m, pose.y_m, math.degrees(pose.yaw_rad))
            for node_id, pose in poses.items()
        })

    def _age_cooldowns(self) -> None:
        expired = []
        for node_id, remaining in self._cooldowns.items():
            remaining -= 1
            if remaining <= 0:
                expired.append(node_id)
            else:
                self._cooldowns[node_id] = remaining
        for node_id in expired:
            self._cooldowns.pop(node_id, None)

    def _cool_region(self, node_id: int, duration: int) -> None:
        center_index = next(
            (index for index, record in enumerate(self._records) if record.node_id == node_id),
            None,
        )
        if center_index is None:
            return
        for index, record in enumerate(self._records):
            if self._same_region(center_index, index):
                self._cooldowns[record.node_id] = max(
                    self._cooldowns.get(record.node_id, 0), duration
                )
