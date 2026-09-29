#pragma once

#include <cstdint>

namespace behavior_rtabmap {

constexpr std::uint16_t kProtocolVersion = 16;
constexpr std::uint64_t kMaxPacketBytes = 128ULL * 1024ULL * 1024ULL;

enum class Command : std::uint16_t {
  kReset = 1,
  kFrame = 2,
  kPing = 3,
  kShutdown = 4,
};

enum class Status : std::uint16_t {
  kOk = 0,
  kTrackingLost = 1,
  kBadRequest = 2,
  kInternalError = 3,
};

enum class QueryOutcome : std::uint8_t {
  kNone = 0,
  kHolding = 1,
  kCommitted = 2,
  kAllKnown = 3,
  kBridgeOnlyDiscarded = 4,
  kNoveltyResumed = 5,
};

enum class QueryScope : std::uint8_t {
  kNone = 0,
  kRecovery = 1,
  kNormal = 2,
};

#pragma pack(push, 1)
struct Prefix {
  char magic[4];
  std::uint16_t version;
  std::uint16_t code;
  std::uint64_t payload_size;
};

struct FrameMeta {
  std::uint64_t frame_id;
  double stamp;
  std::uint32_t width;
  std::uint32_t height;
  double fx;
  double fy;
  double cx;
  double cy;
  double camera_to_base[12];
  double odom_dx;
  double odom_dy;
  double odom_dyaw;
  // bit 0: suppress permanent mapping while recovering after a structural
  // camera interruption; bit 1: the external metric pose has independent
  // multi-window RGB-D recovery evidence; bit 2: the client-side stateful
  // camera-pose gate says this observation is stable enough for structure;
  // bit 3: independently verified ordinary graph bridge; bit 4: an ordinary
  // whole-map probe completed with no historical mode, allowing the existing
  // multi-view novelty gate to release its read-only hold; bit 5: that
  // ordinary external whole-map search is still pending on this frame; bit 6:
  // two independent exhaustive recovery searches certified no historical mode.
  std::uint32_t frame_flags;
  // Candidate plus an independently depth-registered metric observation.
  // RTAB-Map still recomputes RGB-D/ICP registration before accepting a link.
  std::int32_t external_loop_candidate_id;
  // Current base pose expressed in the historical candidate base frame.
  // In RTAB-Map pose notation this is candidate->current.
  double external_loop_candidate_to_current[3];
  double external_loop_covariance[9];
  // Monotonic client-side identity of the recovery/normal quarantine episode.
  // Zero is reserved for frames outside either query hold.
  std::uint64_t query_generation;
};

struct ResponseMeta {
  std::uint64_t frame_id;
  std::uint8_t tracking_ok;
  std::uint8_t map_updated;
  std::uint8_t loop_closed;
  // bit 0: global localization mode, bit 1: localized this frame,
  // bit 2: visual feature evidence, bit 3: depth-geometry evidence,
  // bit 4: transient read-only revisit match while mapping remains enabled,
  // bit 5: recovery hold kept this frame out of the permanent map,
  // bit 6: soft localization/candidate hold suppressed permanent writes while
  // RTAB-Map remained incremental, bit 7: that hold has uncertain global pose.
  std::uint8_t mode_flags;
  // Two-phase acknowledgement for a generation-bound provisional query.
  // Only a generation-bound terminal outcome authorizes the client to end its
  // hold. Bridge-only-discarded means the verified A-C bridge committed while
  // unsafe provisional coverage was intentionally dropped.
  std::uint8_t query_outcome;
  // RTAB-Map Optimizer::Type.  The Python client rejects anything but Ceres.
  std::uint8_t optimizer_backend;
  // Echoes the exact quarantine transaction acknowledged above. The Python
  // state machine ignores terminal outcomes bound to another generation.
  std::uint8_t query_scope;
  std::uint64_t query_generation;
  std::uint32_t width;
  std::uint32_t height;
  double x_min;
  double y_min;
  double cell_size;
  double pose_x;
  double pose_y;
  double pose_yaw;
  double native_pose_x;
  double native_pose_y;
  double native_pose_yaw;
  double fused_odom_pose_x;
  double fused_odom_pose_y;
  double fused_odom_pose_yaw;
  std::uint32_t node_count;
  std::uint32_t loop_count;
  std::uint32_t inliers;
  std::uint32_t features;
  std::uint32_t grid_bytes;
  std::uint32_t pose_count;
  std::int32_t ref_node_id;
  // Native-authoritative progress through the post-novelty occupancy
  // reconciliation window. Only physically independent accepted mapping
  // viewpoints consume this counter.
  std::uint32_t novelty_resume_reconciliation_viewpoints_remaining;
};

struct PoseRecord {
  std::int32_t node_id;
  double x;
  double y;
  double yaw;
};
#pragma pack(pop)

static_assert(sizeof(Prefix) == 16, "Python/C++ Prefix layout mismatch");
static_assert(sizeof(FrameMeta) == 288, "Python/C++ FrameMeta layout mismatch");
static_assert(sizeof(ResponseMeta) == 159, "Python/C++ ResponseMeta layout mismatch");
static_assert(sizeof(PoseRecord) == 28, "Python/C++ PoseRecord layout mismatch");

}  // namespace behavior_rtabmap
