"""RoboDojo-only fixed tabletop anchors alongside the official dynamic tracker."""

from __future__ import annotations

from dataclasses import dataclass, fields
from copy import deepcopy
from typing import Any

import numpy as np

from tools.util.dynamic_point_tracker import DynamicPointSnapshot, DynamicPointTracker
from tools.util.tabletop_surface import TabletopSurface


@dataclass(frozen=True)
class SurfacePointSnapshot(DynamicPointSnapshot):
    surface_classification: dict[str, Any] | None = None
    fixed_tabletop: bool = False
    anchor_visible: bool | None = None

    def as_dict(self):
        return dict(
            super().as_dict(),
            surface_classification=deepcopy(self.surface_classification),
            fixed_tabletop=self.fixed_tabletop,
            anchor_visible=self.anchor_visible,
        )


def _tag(snapshot, classification, *, visible=None):
    return SurfacePointSnapshot(
        **{
            field.name: getattr(snapshot, field.name)
            for field in fields(DynamicPointSnapshot)
        },
        surface_classification=deepcopy(classification),
        fixed_tabletop=bool(classification.get("fixed")),
        anchor_visible=visible,
    )


class TabletopDynamicPointTracker(DynamicPointTracker):
    """Classify only on registration; never turn a resting object into an anchor.

    Fixed XYZ comes from the registered RGB-D point. Visibility is checked in
    later frames but does not move or retire that anchor. A hidden anchor uses
    status ``fixed`` and explicitly reports frozen rather than live geometry.
    """

    def __init__(self, *, table_plane_provider=None, **kwargs):
        super().__init__(**kwargs)
        self._table_plane_provider = table_plane_provider
        self._surface_model = None
        self._surface_classifications = {}

    def reset(self):
        super().reset()
        self._surface_model = None
        self._surface_classifications = {}

    def mark_point(self, **kwargs):
        with self._lock:
            snapshot = super().mark_point(**kwargs)
            if self._surface_model is None:
                # _ValidatedFrame and TrackerFrame use different depth names.
                from types import SimpleNamespace

                frame = self._frame
                source = SimpleNamespace(
                    depth_linear=frame.depth,
                    intrinsics=frame.intrinsics,
                    camera_to_robot_base=frame.camera_to_robot_base,
                )
                reference = None
                if self.camera_role != "head":
                    reference = (
                        self._table_plane_provider(frame)
                        if self._table_plane_provider is not None
                        else {"confident": False}
                    )
                self._surface_model = TabletopSurface.from_frame(
                    source, reference_plane=reference
                )
            report = self._surface_model.classify(snapshot.pixel_uv)
            report["registration_observation_sequence"] = snapshot.observation_sequence
            self._surface_classifications[snapshot.track_id] = report
            tagged = _tag(snapshot, report, visible=True if report["fixed"] else None)
            self._tracks[snapshot.track_id].snapshot = tagged
            return tagged

    def remove_point(self, track_id):
        with self._lock:
            self._surface_classifications.pop(track_id, None)
            return super().remove_point(track_id)

    def activate_rigid_pair(self, track_ids, **kwargs):
        if any(
            self._surface_classifications.get(key, {}).get("fixed") for key in track_ids
        ):
            raise ValueError(
                "fixed tabletop points cannot be activated as on-hand rigid points"
            )
        return super().activate_rigid_pair(track_ids, **kwargs)

    def ingest(self, frame, **kwargs):
        with self._lock:
            fixed = {
                key: state
                for key, state in self._tracks.items()
                if self._surface_classifications.get(key, {}).get("fixed")
            }
            previous_tracks = self._tracks
            previous_episode = self._episode_id
            self._tracks = {
                key: state for key, state in self._tracks.items() if key not in fixed
            }
            try:
                super().ingest(frame, **kwargs)
            except Exception:
                self._tracks = previous_tracks
                raise
            if previous_episode != self._episode_id:
                self._surface_classifications.clear()
                fixed = {}
            self._surface_model = None
            for key, state in self._tracks.items():
                report = self._surface_classifications.get(key)
                if report:
                    state.snapshot = _tag(state.snapshot, report)
            for key, state in fixed.items():
                state = self._clone_track_state(state)
                state.snapshot = self._fixed_snapshot(state, self._frame)
                state.anchor_px = np.asarray(state.snapshot.pixel_uv or state.anchor_px)
                state.last_observation_method = "fixed_tabletop_registration"
                self._tracks[key] = state
            return self.list_points()

    def _fixed_snapshot(self, state, frame):
        report = self._surface_classifications[state.track_id]
        anchor = state.registration_point_policy
        point_camera = self._transform_point(
            np.linalg.inv(frame.camera_to_policy), anchor
        )
        point_base = self._transform_point(frame.camera_to_robot_base, point_camera)
        pixel, expected_depth = self._project_policy_point(anchor, frame)
        uv = (
            None
            if pixel is None
            else self._pixel_to_relative(
                pixel, frame.intrinsics.width, frame.intrinsics.height
            )
        )
        visible = False
        if pixel is not None:
            px, py = np.rint(pixel).astype(int)
            depth = float(frame.depth[py, px])
            if np.isfinite(depth) and depth > 0:
                measured = self._transform_point(
                    frame.camera_to_robot_base,
                    self._unproject(
                        np.array([px, py])
                        + report["plane"].get("depth_pixel_center_offset", 0.0),
                        depth,
                        frame.intrinsics,
                    ),
                )
                a, b, c = report["plane"]["plane_coefficients_z_from_xy"]
                residual = (
                    measured[2] - a * measured[0] - b * measured[1] - c
                ) / np.sqrt(1 + a * a + b * b)
                visible = abs(residual) <= report["plane"]["table_tolerance_m"]
        if visible:
            state.last_seen_sequence = frame.sequence
        # status=fixed keeps "not currently observed" distinct from a lost
        # object. XYZ is a persistent scene reference, not a new RGB-D sample.
        snapshot = DynamicPointSnapshot(
            track_id=state.track_id,
            label=state.label,
            camera_role=state.camera_role,
            status="observed" if visible else "fixed",
            uv=uv,
            pixel_uv=None if pixel is None else tuple(pixel),
            depth_m=expected_depth,
            point_camera_m=tuple(point_camera),
            point_robot_base_m=tuple(point_base),
            point_policy_local_m=tuple(anchor),
            velocity_policy_local_m_s=(0.0, 0.0, 0.0),
            predicted_uv=None,
            predicted_depth_m=None,
            confidence=1.0,
            observation_sequence=frame.sequence,
            last_seen_sequence=state.last_seen_sequence,
            missed_steps=0,
            episode_id=frame.episode_id,
            depth_source="fixed_tabletop_registration_projected_into_current_camera",
        )
        return _tag(snapshot, report, visible=visible)
