#include "protocol.hpp"

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Compression.h>
#include <rtabmap/core/Features2d.h>
#include <rtabmap/core/LaserScan.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/LocalGrid.h>
#include <rtabmap/core/LocalGridMaker.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Registration.h>
#include <rtabmap/core/RegistrationInfo.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Statistics.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/global_map/OccupancyGrid.h>
#include <rtabmap/utilite/ULogger.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace br = behavior_rtabmap;

namespace {

constexpr float kMinRangeM = 0.45F;
constexpr float kMaxRangeM = 3.5F;
constexpr float kDepthEdgeJumpM = 0.12F;
constexpr float kFloorMaxM = 0.08F;
constexpr float kObstacleMinM = 0.15F;
constexpr float kObstacleMaxM = 1.95F;
constexpr std::uint8_t kWallBandCount = 6;
constexpr std::uint8_t kWallMinGroundedRun = 2;
constexpr float kSelfForwardM = 0.95F;
constexpr float kSelfHalfWidthM = 0.55F;
constexpr float kGridCellM = 0.05F;
// The R1Pro base is about 0.42 m wide. Occupancy recovered from the larger
// arm envelope may not overwrite cells the physical base has demonstrably
// traversed. Include half a grid diagonal so rasterization cannot close the
// passage again at a cell boundary.
constexpr float kTraversedFreeCellRadiusM = 0.25F;
// Mapping nodes are normally much denser than this at commanded chassis
// speeds. Do not draw a free corridor across a discontinuous relocalization.
constexpr float kTraversedFreeMaxNodeStepM = 0.75F;
constexpr int kDepthStride = 4;
constexpr int kSelfSupportRowStride = 8;
constexpr int kSelfGridMinX = -70;
constexpr int kSelfGridMaxX = 19;
constexpr int kSelfGridMinY = -11;
constexpr int kSelfGridMaxY = 11;
constexpr std::size_t kSelfGridWidth =
    static_cast<std::size_t>(kSelfGridMaxX - kSelfGridMinX + 1);
constexpr std::size_t kSelfGridHeight =
    static_cast<std::size_t>(kSelfGridMaxY - kSelfGridMinY + 1);
constexpr std::size_t kSelfGridCells = kSelfGridWidth * kSelfGridHeight;
constexpr std::uint32_t kMaxImageDimension = 4096;
constexpr std::uint32_t kFrameRecoveryHold = 1U;
constexpr std::uint32_t kFrameRecoveryRelocalization = 2U;
constexpr std::uint32_t kFrameStructuralObservationUsable = 4U;
constexpr std::uint32_t kFrameVerifiedGraphBridge = 8U;
constexpr std::uint32_t kFrameNormalGlobalNoMode = 16U;
constexpr std::uint32_t kFrameNormalGlobalSearchPending = 32U;
constexpr std::uint32_t kFrameRecoveryGlobalNoMode = 64U;
constexpr std::uint32_t kKnownFrameFlags =
    kFrameRecoveryHold | kFrameRecoveryRelocalization |
    kFrameStructuralObservationUsable | kFrameVerifiedGraphBridge |
    kFrameNormalGlobalNoMode | kFrameNormalGlobalSearchPending |
    kFrameRecoveryGlobalNoMode;

void test_only_exit_after_all_known_ledger() {
  const char *value =
      std::getenv("BEHAVIOR_RTABMAP_TEST_ONLY_EXIT_AFTER_ALL_KNOWN_LEDGER");
  if (value != nullptr && std::strcmp(value, "1") == 0) {
    std::_Exit(86);
  }
}

void test_only_exit_after_novelty_resumed_ledger() {
  const char *value = std::getenv(
      "BEHAVIOR_RTABMAP_TEST_ONLY_EXIT_AFTER_NOVELTY_RESUMED_LEDGER");
  if (value != nullptr && std::strcmp(value, "1") == 0) {
    std::_Exit(87);
  }
}

constexpr float kQueryKeyframeTranslationM = 2.0F * kGridCellM;
constexpr float kQueryKeyframeYawRad = 0.034906585F;  // two degrees
constexpr float kQueryChunkMaxTravelM = kMaxRangeM;
constexpr float kQueryChunkMaxTurnRad = 1.570796327F;
constexpr std::size_t kQueryMaxKeyframesPerChunk = 32;
constexpr std::size_t kQueryMaxKeyframesHard = 4096;
constexpr std::size_t kQueryMaxBytes = 64ULL * 1024ULL * 1024ULL;
constexpr float kQueryMaxPromotionUncertaintyM = 0.50F;
// The uncertainty-domain veto is constructed only at query release. Bound its
// temporary raster independently from the 64 MiB evidence buffer; exceeding
// this limit rejects the Q payload instead of approximating or dropping an
// unsafe observation.
constexpr std::size_t kQueryMaxUncertaintyDomainCells = 4ULL * 1024ULL * 1024ULL;
constexpr std::size_t kQueryMaxUncertaintyDomainWorkCells =
    64ULL * 1024ULL * 1024ULL;
// 每个合规 RGB-D 观测都交给 RTAB-Map 并保留成可重开的图节点。外层
// 抽帧会漏掉原地旋转和受阻时的重访特征；RTAB 内部再删除小位移节点
// 则会让在线栅格与持久图不一致。
constexpr double kSlamUpdateIntervalS = 0.0;
// 建筑平面图只接收能看到水平结构的头部相机观测。头部被机械臂
// 带到低位、过度俯视或远离底盘时，当前帧仍可供里程计跟踪，但不能把
// 机械臂和手持物写入永久栅格。阈值表达的是视线与机器人几何约束，
// 不依赖任务、路线或数据集帧号。
// Only a stable, chassis-centred head view may contribute permanent
// structural cells. During manipulation the head can travel down and forward
// with the arm; those transition views are still valid for odometry and later
// localization, but their depth is dominated by the held object/arm. The
// Python StructuralObservationGate applies the strict-to-recovery hysteresis;
// the native bounds mirror its settled recovery envelope so a valid recovered
// frame is not discarded a second time. A transient frame is still rejected
// by the protocol flag before it reaches this worker.
constexpr float kMappingViewMinHorizontal = 0.75F;
constexpr float kMappingCameraMinHeightM = 1.15F;
constexpr float kMappingCameraMaxPlanarOffsetM = 0.60F;
constexpr int kRegistrationMinDepthPixels = 32;
constexpr float kFusionMinPredictedTranslationM = 0.01F;
constexpr float kFusionMinPredictedYawRad = 0.008726646F;  // 0.5 degrees
// Treat RGB-D as a measurement of the innovation around the continuous qvel
// prediction.  Sub-centimetre / sub-half-degree ICP residuals are dominated by
// quantization and correspondence churn; accumulating them frame by frame
// produces a biased random walk.  A blocked command or real wheel slip exceeds
// this gate and is still corrected from the compliant camera observation.
constexpr float kFusionTranslationInnovationM = 0.01F;
constexpr float kFusionYawInnovationRad = 0.008726646F;  // 0.5 degrees
constexpr int kIcpMinTranslationCorrespondences = 200;
// Point-to-plane structural complexity is the smallest normalized information
// eigenvalue.  Below this value, corridor-parallel translation is not
// observable even though wall orientation (and therefore planar yaw) still is.
constexpr float kIcpMinStructuralComplexity = 0.15F;
constexpr int kVisualOdomMinInliers = 15;
constexpr float kVisualOdomMinInlierRatio = 0.20F;
constexpr float kVisualOdomMinDistribution = 0.001F;
// Keep graph revisits at the same conservative geometric minimum as frame
// odometry.  A separate read-only relocalizer may probe sparse historical
// representatives, but sparse matches must never be allowed to create mapping
// constraints and deform the house.
constexpr int kVisualLoopMinInliers = 15;
// A read-only appearance seed denotes a small historical graph region, not an
// exact camera instant. Probe the seed and its immediate odometry neighbors so
// a representative selected at node 176 can recover the metrically stronger
// node 177 observation without opening a whole-map alias search in native code.
constexpr std::size_t kReadOnlyRgbdMaxCandidates = 3;
constexpr float kReadOnlyRgbdMinInlierDistribution = 0.001F;
// Kidnapped-pose recovery searches the whole historical map, so use
// RTAB-Map's upstream general-purpose default instead of the lower local-loop
// threshold used while a continuous odometry chain is available.
constexpr float kOdomTranslationExcessLimitM = 0.15F;
constexpr float kOdomYawExcessLimitRad = 0.261799388F;  // 15 degrees
constexpr int kStaticObservationStride = 4;
constexpr int kStaticObservationMinOverlap = 200;
constexpr float kStaticObservationMinOverlapRatio = 0.65F;
constexpr float kStaticDepthP90M = 0.02F;
constexpr std::uint8_t kStaticRgbP90 = 1;
constexpr std::uint8_t kStaticTextureDifference = 8;
constexpr float kStaticMinTexturedRatio = 0.05F;
constexpr float kStaticCameraTranslationM = 0.01F;
constexpr float kStaticCameraRotationRad = 0.004363323F;  // 0.25 degrees
// A fully static RGB-D stream must not create a database transaction for every
// camera tick.  Keep the detector conservative and reversible: a small amount
// of accumulated qvel, a camera/exposure change, or a structural query exits
// the hold immediately. The worker still receives every frame while idle so
// wake-up does not depend on a timer or on restarting the client.
constexpr std::uint32_t kIdleEnterStaticFrames = 8;
constexpr float kIdlePerFrameWakeTranslationM = 0.002F;
constexpr float kIdlePerFrameWakeYawRad = 0.002F;
constexpr float kIdleAccumulatedWakeTranslationM = 0.01F;
constexpr float kIdleAccumulatedWakeYawRad = 0.01F;
// 冻结依据是在线 SLAM 证据：RTAB-Map 原生 RGB-D 特征在多个
// 空间区域通过 3-D 几何验证，同时栅格信息增量已低且图优化
// 已稳定。它不检查绘制后的墙是否“像户型图”，也不读取路线标签。
// 积累的 SE(2) 运动只用来排除局部相邻观测；真正的“已见过”仍要求
// RTAB-Map 接受视觉闭环或视觉空间近邻边。
// A native visual+metric closure is already a strong geometric observation.
// Express angular progress at a sensor-scale radius so a compact-room turn is
// independent evidence too. This depends on observed SE(2) motion, not frame
// rate, route length or room size.
constexpr float kFreezeRotationEquivalentRadiusM = 1.0F;
constexpr float kFreezeMinLoopMotionSeparationM = 1.25F;
// Do not declare a whole map converged from repeated closures in one small
// corner. At least one accepted native cycle must span a material fraction of
// both the retained pose graph and the traversed path. Ratios make this gate
// independent of room size, route length, frame rate and node-id density.
constexpr float kFreezeMinLoopNodeSpanRatio = 0.30F;
constexpr float kFreezeMinLoopMotionSpanRatio = 0.35F;
// A capture can revisit the same doorway/turning bay from two different
// headings.  Require more than one independent loop event, but do not make
// freezing depend on those events falling into arbitrary 2 m bins.
constexpr std::size_t kFreezeMinVisualLoopRegions = 1;
constexpr std::size_t kFreezeMinAcceptedLoopEvents = 2;
constexpr std::uint32_t kFreezeEvidenceWindowFrames = 120;
constexpr std::uint32_t kFreezeRecentGrowthWindowFrames = 30;
constexpr std::uint32_t kFreezeVisualRevisitWindowFrames = 180;
// Graph optimization itself is synchronous.  Three independent convergence
// samples are enough to show that its correction has stopped moving; keeping
// a sixty-frame tail used to write the already-known return corridor into the
// permanent map long after a metric closure had settled.
constexpr std::uint32_t kFreezeGraphQuietFrames = 15;
constexpr std::uint32_t kFreezeLoopSettleFrames = 15;
// A ready sample is only a candidate.  Keep observing long enough to catch a
// robot that is about to leave the currently explored room; any new coverage
// or graph motion during this debounce interval cancels the candidate.
// Candidate debounce is measured in mapping observations, not a case-specific
// frame number. The ready gate already contains 120-frame coverage, a
// three-sample graph window and 30-frame growth window, so two additional
// metric periods are enough
// to catch a late novelty burst without postponing localization past the
// revisit that established convergence.
constexpr std::uint32_t kFreezeCandidateHoldFrames = 10;
constexpr std::uint32_t kFreezeMetricPeriodFrames = 5;
// A convergence sample can occur during a short manipulation pause or an
// in-place panoramic turn. Neither proves that the robot crossed a visible
// doorway. After a candidate starts, require one sensor-scale *translational*
// exploration window before switching. Rotation remains valid loop evidence,
// but it cannot by itself declare the whole environment explored.
constexpr float kFreezeCandidateMotionRangeFraction = 0.75F;
constexpr float kFreezeCandidateMotionPathFraction = 0.15F;
constexpr float kFreezeMaxKnownGrowthRatio = 0.05F;
constexpr float kFreezeMaxFrontierRatio = 0.05F;
// Compare the current compliant depth endpoints with the previous native
// raster while a freeze candidate is being debounced. This catches motion into
// an unseen room before that frame has been inserted into the map.
constexpr float kFreezeMaxObservationNoveltyRatio = 0.30F;
// A freeze candidate normally completes after another native revisit.  Some
// settled maps will not create a new unique loop pair (RTAB-Map deduplicates
// nearby links), so one continuously ready debounce period is equivalent
// evidence. Candidate invalidation below remains active on every observation.
constexpr std::uint32_t kFreezeCandidateQuietCompletionFrames = 10;
constexpr float kFreezeRaySampleStepM = 0.10F;
constexpr float kFreezeRayStartM = 0.35F;
constexpr float kFreezeRayEndBackoffM = 0.15F;
constexpr float kFreezeGraphCorrectionM = 0.10F;
constexpr float kFreezeGraphCorrectionRad = 0.034906585F;  // 2 degrees
// A repeated read-only scan alignment is used to refresh the live odometry
// chain, not to optimize the graph.  Its temporal disagreement must be no
// larger than one map cell at the farthest accepted depth; otherwise a pair
// of individually plausible ICP results can ratchet a wall by several cells.
// atan(0.05 / 3.5) = 0.014284... rad (0.818 degrees).
constexpr float kReadOnlyScanMatchMaxDisagreementYawRad = 0.014284F;
constexpr std::size_t kFreezeGraphMinCommonNodes = 60;
constexpr float kFreezeGraphPercentile = 0.95F;
constexpr float kFreezeLoopRegionM = 2.0F;
constexpr int kFreezeLoopPairDedupNodes = 10;
// qvel 是合规且连续的 odom 预测。邻帧 RGB-D/ICP 只判断观测是否
// 支持运动；历史地点的特征+深度约束才通过位姿图估计 map->odom。
constexpr double kQvelFallbackTranslationVariance = 0.0004;  // 2 cm sigma
constexpr double kQvelFallbackYawVariance = 0.001218469679;   // 2 deg sigma
// Raw base_qvel is an integrated body twist, so its process uncertainty is a
// density over actual motion, not a charge paid once per camera frame.  Using
// the fallback variances above on every observation makes an unchanged pose
// less certain merely because the camera runs faster and makes a frozen query
// exceed its safety bound after only a few frames.  These rates preserve the
// same conservative 2 cm / 2 degree scale after one metre / radian while
// remaining invariant to how that motion is split across observations.
constexpr double kRawQvelTranslationVariancePerMeter = 0.0004;
constexpr double kRawQvelYawVariancePerRadian = 0.001218469679;
// A global appearance proposal is accepted only when its metric transform is
// statistically compatible with the compliant odometry chain.  The threshold
// is the 99% quantile of chi-square(3), for planar x/y/yaw innovation.  It is
// independent of route length: uncertainty is propagated through every SE(2)
// increment below, including yaw-to-position coupling.
constexpr double kExternalLoopInnovationChiSquare = 11.3448667301;
// A confirmed RGB-D revisit is handled as a transient localization sample
// before it reaches Rtabmap::process().  These limits are physical sensor
// limits, not route/frame thresholds: a candidate must agree with the
// body-velocity chain and may correct only a bounded accumulated drift.
constexpr double kReadOnlyRevisitMaxInnovationChiSquare =
    kExternalLoopInnovationChiSquare;
constexpr float kReadOnlyRevisitMaxCorrectionM = 1.50F;
constexpr float kReadOnlyRevisitMaxCorrectionYawRad = 0.785398163F; // 45 deg
constexpr float kReadOnlyRevisitMaxCandidateDistanceM = 5.0F;
constexpr float kReadOnlyRevisitMaxTranslationHoldM = 6.0F;
// Kidnapped recovery is the only path allowed to exceed the ordinary local
// correction cap. Its bound expands only with motion accumulated while the
// structural camera was unusable and remains capped to a residential-scale
// local map. The metric pose still needs independent appearance and rolling
// depth consensus before the Python protocol can set the recovery bit.
constexpr float kRecoveryRelocalizationMaxCorrectionM = 12.0F;
constexpr float kRecoveryRelocalizationMaxCorrectionYawRad = 3.141592654F;
// Unknown-ray ratios are noisy at depth discontinuities.  Require a short
// consecutive burst before leaving the read-only window so one bad depth
// sample cannot reopen mapping at a known corner.
constexpr float kReadOnlyRevisitMaxNoveltyRatio = 0.35F;
constexpr std::uint32_t kReadOnlyRevisitNoveltyExitStreak = 3;
constexpr std::uint32_t kReadOnlyRevisitNoObservationGrace = 8;
constexpr std::size_t kReadOnlyRevisitMinMapNodes = 4;
// A native novelty terminal is sent before the Python-side mapping flags can
// observe the transition. Keep the terminal frame and one following eligible
// update transient so that this protocol lag cannot materialize a stale corner
// into the map. The counter is consumed only by an actual mapping update.
constexpr std::uint32_t kNativeNoveltyResumeGuardUpdates = 2;
// Keep the committed-raster veto active for a bounded sequence of physically
// independent mapping viewpoints after novelty resumes. Counting accepted
// nodes would let stationary duplicates exhaust the safety window without
// adding any new spatial evidence. Sixteen viewpoints cover two complete
// local scan cycles while remaining a fixed, auditable progress target.
constexpr std::uint32_t kNativeNoveltyResumeReconciliationViewpoints = 16;
// Free space in a committed raster is part of the localization contract, not
// disposable transition state. Keep its union for the worker session so a
// later ordinary mapping scan cannot turn already-observed free cells into a
// duplicate wall after the bounded reconciliation window has elapsed.
constexpr std::size_t kNativeNoveltyResumeMaxProtectedCells =
    4ULL * 1024ULL * 1024ULL;
// Unknown obstacle cells remain provisional after the reconciliation window.
// Each cell needs two independently positioned, uncertainty-bounded supports
// before it can enter ordinary mapping. Exhausting this sparse cache only
// delays new walls, so it is safe to reject new candidates without discarding
// an already-established certificate or stopping the worker.
constexpr std::size_t kNativeNoveltyResumeMaxObstacleCandidateCells =
    256ULL * 1024ULL;
// A provisional obstacle starts collecting evidence only after the transition
// raster has closed. Require a sensor-scale baseline between its two supports;
// adjacent mapping keyframes are not independent evidence for a permanent wall.
constexpr float kNativeNoveltyResumeObstacleIndependentTranslationM = 0.25F;
constexpr float kNativeNoveltyResumeObstacleIndependentYawRad =
    0.139626340F;  // 8 degrees
// Reopening mapping is deliberately stricter than detecting one novel depth
// image.  The unknown structure must persist across time and across two
// physically distinct, non-static RGB-D views.  This keeps blocked turns and
// moving foreground objects from turning localization failures into duplicate
// map nodes.
constexpr std::uint32_t kSoftNoveltyMinViewpoints = 2;
constexpr double kSoftNoveltyMinDurationS = 0.50;
constexpr float kSoftNoveltyViewTranslationM = 2.0F * kGridCellM;
constexpr float kSoftNoveltyViewYawRad = 0.087266463F;  // 5 degrees
// Rotation reveals another bearing, but it does not provide translational
// parallax and cannot distinguish a genuinely new room from a familiar wall
// observed under a drifted heading.  Reopening writes therefore also needs a
// sensor-scale displacement from the first sustained-novelty observation.
constexpr float kSoftNoveltyMinTranslationSpanM =
    kNativeNoveltyResumeObstacleIndependentTranslationM;
// A low-novelty raster overlap is only a cheap trigger for historical metric
// registration. It can never suppress a signature on its own. The actual
// decision is made by RTAB-Map's point-to-plane ICP against depth geometry
// from old optimized nodes, excluding a trailing motion window so adjacent
// observations cannot masquerade as a revisit.
constexpr float kReadOnlyCandidatePrefilterNoveltyRatio = 0.25F;
// A geometrically plausible revisit is staged without applying its ambiguous
// ICP transform. Give the asynchronous appearance matcher one sensor-scale
// motion window to establish place identity; after that, mapping resumes.
constexpr float kReadOnlyCandidateMaxMotionM = 0.75F;
constexpr float kReadOnlyScanMatchMaxAcceptedNoveltyRatio = 0.30F;
constexpr std::size_t kReadOnlyScanMatchMinMapNodes = 64;
constexpr std::size_t kReadOnlyScanMatchMinLoopEvidence = 1;
constexpr float kReadOnlyScanMatchReferenceMotionGapM = 1.75F;
constexpr float kReadOnlyScanMatchMaxCorrectionM = 0.50F;
constexpr float kReadOnlyScanMatchMaxCorrectionYawRad = 0.261799388F;
constexpr float kReadOnlyScanMatchReferenceRadiusM =
    kMaxRangeM + kReadOnlyScanMatchMaxCorrectionM;
constexpr float kReadOnlyScanMatchReferenceNodeRadiusM =
    2.0F * kMaxRangeM + kReadOnlyScanMatchMaxCorrectionM;
constexpr float kReadOnlyScanMatchMinInlierRatio = 0.35F;
constexpr int kReadOnlyScanMatchMinCorrespondences = 160;
constexpr std::size_t kReadOnlyScanMatchMinReferenceNodes = 3;
constexpr double kReadOnlyScanMatchRetryIntervalS = 0.50;
constexpr std::uint32_t kReadOnlyScanMatchMaxFailures = 3;
// A periodic scan-to-map result is weaker than the RGB-D place anchor that
// established identity.  Never let one raster alignment overwrite that
// anchor.  Confirm it from a second, physically independent view and compare
// both absolute targets after transporting the first with the intervening
// fused odometry.  These are sensor-scale baselines, not route/frame gates.
constexpr float kReadOnlyScanMatchIndependentTranslationM = 0.25F;
constexpr float kReadOnlyScanMatchIndependentYawRad =
    0.139626340F;  // 8 degrees
// When RGB-D observes an axis, keep a finite sensor floor instead of the much
// looser qvel fallback used by the graph edge.  This lets repeated texture be
// rejected by a well-observed yaw sequence while corridor-parallel translation
// remains uncertain. A confirmed static RGB-D observation contributes no
// process noise: uncertainty must follow motion, not camera sample count.
constexpr double kObservedTranslationVarianceFloor = 0.000025;  // 5 mm sigma
constexpr double kObservedYawVarianceFloor = 0.0000761524227;    // 0.5 deg sigma
enum class WorkerProfile {
  kOfficial,
  kSparseRgbd,
  kSparseIcp,
  // Native RTAB-Map visual+ICP graph registration with dense RGB-D
  // descriptors. This is an independent tuning profile; official remains
  // the conservative default used by the live interface.
  kNativeRobust,
  // Same registration policy as kNativeRobust, but use Ceres with RTAB-Map's
  // native robust loss. This keeps the experiment independent from the
  // minimal TORO build and is the candidate for deployments requiring Ceres.
  kNativeRobustCeres,
};
enum class FeatureBackend { kCpu, kKorniaSift };
enum class MappingMode { kMapping, kLocalization };
enum class SoftMappingState {
  kBuilding,
  kCandidateHold,
  kKnownLocalizing,
  kUncertainHold,
};

constexpr bool mapping_view_is_usable(float horizontal_view,
                                      float camera_height_m,
                                      float planar_offset_m) {
  return horizontal_view >= kMappingViewMinHorizontal &&
         camera_height_m >= kMappingCameraMinHeightM &&
         planar_offset_m <= kMappingCameraMaxPlanarOffsetM;
}

static_assert(mapping_view_is_usable(0.92F, 1.57F, 0.22F));
static_assert(mapping_view_is_usable(0.792F, 1.346F, 0.383F));
static_assert(!mapping_view_is_usable(0.28F, 0.50F, 0.90F));

constexpr float absolute_value(float value) {
  return value < 0.0F ? -value : value;
}

constexpr bool plausible_motion_envelope(float predicted_translation,
                                         float predicted_yaw,
                                         float measured_translation,
                                         float measured_yaw) {
  return measured_translation <=
             predicted_translation + kOdomTranslationExcessLimitM &&
         absolute_value(measured_yaw) <=
             absolute_value(predicted_yaw) + kOdomYawExcessLimitRad;
}

bool plausible_measured_increment(const rtabmap::Transform &prediction,
                                  const rtabmap::Transform &measured) {
  return plausible_motion_envelope(
      std::hypot(prediction.x(), prediction.y()), prediction.theta(),
      std::hypot(measured.x(), measured.y()), measured.theta());
}

struct PlanarOdometryStep {
  rtabmap::Transform increment;
  cv::Matx33d covariance;
};

struct ReadOnlyScanMatchProposal {
  rtabmap::Transform target_map_pose;
  rtabmap::Transform fused_odom_pose;
  cv::Mat planar_covariance;
  std::uint64_t frame_id = 0;

  bool active() const {
    return frame_id > 0 && !target_map_pose.isNull() &&
           !fused_odom_pose.isNull();
  }

  void clear() {
    target_map_pose.setNull();
    fused_odom_pose.setNull();
    planar_covariance.release();
    frame_id = 0;
  }
};

double planar_variance(const cv::Mat &covariance, int axis,
                       double fallback) {
  if (covariance.rows == 6 && covariance.cols == 6 &&
      covariance.type() == CV_64FC1) {
    const double value = covariance.at<double>(axis, axis);
    if (std::isfinite(value) && value > 0.0 && value < 9999.0) {
      return value;
    }
  }
  return fallback;
}

cv::Matx33d loop_prior_step_covariance(
    const cv::Mat &selected_covariance,
    const cv::Mat &icp_covariance,
    const cv::Mat &visual_covariance,
    bool static_observation,
    bool translation_observed,
    bool yaw_observed,
    bool visual_translation,
    bool visual_yaw) {
  const cv::Mat &translation_measurement =
      visual_translation ? visual_covariance : icp_covariance;
  const cv::Mat &yaw_measurement =
      visual_yaw ? visual_covariance : icp_covariance;
  const double selected_x = planar_variance(
      selected_covariance, 0, kQvelFallbackTranslationVariance);
  const double selected_y = planar_variance(
      selected_covariance, 1, kQvelFallbackTranslationVariance);
  const double selected_yaw = planar_variance(
      selected_covariance, 5, kQvelFallbackYawVariance);
  const double x = static_observation
                       ? 0.0
                       : translation_observed
                             ? std::max(kObservedTranslationVarianceFloor,
                                        planar_variance(
                                            translation_measurement, 0,
                                            selected_x))
                             : selected_x;
  const double y = static_observation
                       ? 0.0
                       : translation_observed
                             ? std::max(kObservedTranslationVarianceFloor,
                                        planar_variance(
                                            translation_measurement, 1,
                                            selected_y))
                             : selected_y;
  const double yaw = static_observation
                         ? 0.0
                         : yaw_observed
                               ? std::max(kObservedYawVarianceFloor,
                                          planar_variance(
                                              yaw_measurement, 5,
                                              selected_yaw))
                               : selected_yaw;
  return cv::Matx33d(x, 0.0, 0.0,
                     0.0, y, 0.0,
                     0.0, 0.0, yaw);
}

void propagate_planar_odometry(const PlanarOdometryStep &step,
                               rtabmap::Transform *relative_pose,
                               cv::Matx33d *covariance) {
  const double yaw = relative_pose->theta();
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  const double dx = step.increment.x();
  const double dy = step.increment.y();
  const cv::Matx33d state_jacobian(
      1.0, 0.0, -sine * dx - cosine * dy,
      0.0, 1.0,  cosine * dx - sine * dy,
      0.0, 0.0, 1.0);
  const cv::Matx33d noise_jacobian(
      cosine, -sine, 0.0,
      sine, cosine, 0.0,
      0.0, 0.0, 1.0);
  *covariance = state_jacobian * *covariance * state_jacobian.t() +
                noise_jacobian * step.covariance * noise_jacobian.t();
  *relative_pose = (*relative_pose * step.increment).to3DoF();
}

cv::Matx33d raw_qvel_step_covariance(
    const rtabmap::Transform &increment) {
  const double travel_m = std::hypot(increment.x(), increment.y());
  const double turn_rad = std::abs(increment.theta());
  return cv::Matx33d(
      kRawQvelTranslationVariancePerMeter * travel_m, 0.0, 0.0,
      0.0, kRawQvelTranslationVariancePerMeter * travel_m, 0.0,
      0.0, 0.0, kRawQvelYawVariancePerRadian * turn_rad);
}

static_assert(plausible_motion_envelope(0.08F, 0.03F, 0.215F, 0.067F));
static_assert(!plausible_motion_envelope(0.10F, 0.05F, 1.20F, 0.05F));

constexpr bool mapping_has_revisit_evidence(
    std::size_t visual_loop_regions,
    std::size_t accepted_loop_events) {
  return visual_loop_regions >= kFreezeMinVisualLoopRegions &&
         accepted_loop_events >= kFreezeMinAcceptedLoopEvents;
}

constexpr bool loop_has_independent_motion(float motion_separation_m) {
  return motion_separation_m >= kFreezeMinLoopMotionSeparationM;
}

constexpr bool loop_spans_map(float node_span_ratio,
                              float motion_span_ratio) {
  return node_span_ratio >= kFreezeMinLoopNodeSpanRatio &&
         motion_span_ratio >= kFreezeMinLoopMotionSpanRatio;
}

constexpr bool freeze_candidate_can_switch(bool hold_complete,
                                           bool post_candidate_revisit,
                                           bool quiet_completion,
                                           bool viewpoint_challenge_complete,
                                           bool ready_at_switch) {
  // A second closure at the same compact turning spot is not evidence that
  // unexplored doors or corridors do not exist. Both completion paths must
  // survive one sensor-scale motion window, during which online novelty can
  // cancel the irreversible switch to localization.
  return hold_complete && ready_at_switch && viewpoint_challenge_complete &&
         (post_candidate_revisit || quiet_completion);
}

constexpr bool read_only_revisit_can_start(bool appearance_place_verified,
                                           bool metric_transform_verified) {
  return appearance_place_verified && metric_transform_verified;
}

constexpr bool read_only_candidate_can_hold(
    float motion_m, std::uint32_t unknown_streak,
    std::uint32_t no_observation_streak) {
  return motion_m <= kReadOnlyCandidateMaxMotionM &&
         unknown_streak < kReadOnlyRevisitNoveltyExitStreak &&
         no_observation_streak <= kReadOnlyRevisitNoObservationGrace;
}

constexpr bool soft_novelty_can_resume_mapping(
    std::uint32_t novel_observations, std::uint32_t distinct_viewpoints,
    float translation_span_m, double duration_s) {
  return novel_observations >= kReadOnlyRevisitNoveltyExitStreak &&
         distinct_viewpoints >= kSoftNoveltyMinViewpoints &&
         translation_span_m >= kSoftNoveltyMinTranslationSpanM &&
         duration_s >= kSoftNoveltyMinDurationS;
}

// Once the graph has at least one accepted loop, a frame with no commanded
// base translation is a revisit probe, not evidence that a new wall exists.
// Keeping this decision independent of the RGB-D/ICP increment prevents
// registration jitter during an in-place turn from reopening mapping.
constexpr bool known_revisit_stationary_hold(
    bool mapping, std::size_t accepted_loop_events,
    bool predicted_translation_motion, bool query_active,
    bool external_metric_valid) {
  return mapping && accepted_loop_events >= 1U &&
         !predicted_translation_motion && !query_active &&
         !external_metric_valid;
}

static_assert(mapping_has_revisit_evidence(1, 2));
static_assert(!mapping_has_revisit_evidence(1, 1));
static_assert(loop_has_independent_motion(2.0F));
static_assert(!loop_has_independent_motion(0.5F));
static_assert(loop_spans_map(0.45F, 0.50F));
static_assert(!loop_spans_map(0.10F, 0.80F));
static_assert(freeze_candidate_can_switch(true, true, false, true, true));
static_assert(!freeze_candidate_can_switch(true, true, false, false, true));
static_assert(freeze_candidate_can_switch(true, false, true, true, true));
static_assert(!freeze_candidate_can_switch(true, false, true, false, true));
static_assert(!freeze_candidate_can_switch(true, false, false, true, true));
static_assert(!freeze_candidate_can_switch(true, true, false, true, false));
static_assert(read_only_revisit_can_start(true, true));
static_assert(!read_only_revisit_can_start(false, true));
static_assert(!read_only_revisit_can_start(true, false));
static_assert(read_only_candidate_can_hold(0.75F, 0, 0));
static_assert(!read_only_candidate_can_hold(0.751F, 0, 0));
static_assert(!read_only_candidate_can_hold(
    0.1F, kReadOnlyRevisitNoveltyExitStreak, 0));
static_assert(!read_only_candidate_can_hold(
    0.1F, 0, kReadOnlyRevisitNoObservationGrace + 1));
static_assert(soft_novelty_can_resume_mapping(3, 2, 0.25F, 0.50));
static_assert(!soft_novelty_can_resume_mapping(2, 2, 0.25F, 0.50));
static_assert(!soft_novelty_can_resume_mapping(3, 1, 0.25F, 0.50));
static_assert(!soft_novelty_can_resume_mapping(3, 20, 0.0F, 10.0));
static_assert(!soft_novelty_can_resume_mapping(3, 2, 0.249F, 0.50));
static_assert(!soft_novelty_can_resume_mapping(3, 2, 0.25F, 0.49));
static_assert(known_revisit_stationary_hold(true, 1, false, false, false));
static_assert(!known_revisit_stationary_hold(true, 1, true, false, false));
static_assert(!known_revisit_stationary_hold(true, 0, false, false, false));
static_assert(!known_revisit_stationary_hold(true, 1, false, true, false));
static_assert(!known_revisit_stationary_hold(true, 1, false, false, true));
bool trace_enabled() {
  const char *value = std::getenv("BEHAVIOR_RTABMAP_TRACE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

float read_only_scan_match_max_correction_yaw_rad() {
  return kReadOnlyScanMatchMaxCorrectionYawRad;
}

float statistic_value(const rtabmap::Statistics &statistics,
                      const std::string &key) {
  const auto found = statistics.data().find(key);
  return found == statistics.data().end() ? 0.0F : found->second;
}

bool read_all(std::istream &stream, void *target, std::size_t size) {
  auto *bytes = static_cast<char *>(target);
  std::size_t received = 0;
  while (received < size) {
    stream.read(bytes + received, static_cast<std::streamsize>(size - received));
    const std::streamsize count = stream.gcount();
    if (count <= 0) {
      return false;
    }
    received += static_cast<std::size_t>(count);
  }
  return true;
}

void write_all(std::ostream &stream, const void *source, std::size_t size) {
  stream.write(static_cast<const char *>(source), static_cast<std::streamsize>(size));
  if (!stream) {
    throw std::runtime_error("failed to write worker response");
  }
}

void append_bytes(std::vector<std::uint8_t> &output, const void *source, std::size_t size) {
  const auto *begin = static_cast<const std::uint8_t *>(source);
  output.insert(output.end(), begin, begin + size);
}

void send_packet(br::Status status, const std::vector<std::uint8_t> &payload) {
  br::Prefix prefix{{'B', '1', 'R', 'S'}, br::kProtocolVersion,
                    static_cast<std::uint16_t>(status), payload.size()};
  write_all(std::cout, &prefix, sizeof(prefix));
  if (!payload.empty()) {
    write_all(std::cout, payload.data(), payload.size());
  }
  std::cout.flush();
}

void send_error(br::Status status, const std::string &message) {
  std::vector<std::uint8_t> payload(message.begin(), message.end());
  send_packet(status, payload);
}

rtabmap::ParametersMap slam_parameters(
    WorkerProfile profile,
    FeatureBackend feature_backend,
    const std::string &python_detector_path,
    const std::string &python_matcher_path) {
  using rtabmap::ParametersPair;
  rtabmap::ParametersMap parameters;
  const auto set = [&parameters](const std::string &name, const std::string &value) {
    parameters.insert(ParametersPair(name, value));
  };

  // F2F keeps only the previous observation. base_qvel is supplied as the
  // body-frame prediction on every call; RGB-D registration estimates the
  // observed motion around that prediction.
  set("Odom/Strategy", "1");
  set("Odom/GuessMotion", "false");
  set("Odom/ResetCountdown", "0");
  set("Odom/ImageDecimation", "1");
  // The graph backend uses native metric registration. Frame odometry remains
  // an independent ICP instance configured in reset().
  const bool native_robust = profile == WorkerProfile::kNativeRobust ||
                             profile == WorkerProfile::kNativeRobustCeres;
  const bool native_robust_ceres =
      profile == WorkerProfile::kNativeRobustCeres;
  const bool kornia_sift = feature_backend == FeatureBackend::kKorniaSift;
  // Keep mutable mapping geometric: official proximity candidates first need
  // feature correspondences and then must agree with compliant depth ICP
  // before they can deform historical occupancy. Global appearance proposals
  // remain disabled while mapping, so repeated doors cannot create an edge
  // without already lying in the odometric proximity basin. Read-only
  // localization uses the same VisIcp verifier over frozen history.
  set("Reg/Strategy", profile == WorkerProfile::kSparseRgbd
                          ? "0"
                          : profile == WorkerProfile::kOfficial ||
                                    native_robust
                                ? "2"
                                : "1");
  set("Reg/RepeatOnce", "true");
  set("Reg/Force3DoF", "true");
  // Upstream VisIcp normally runs its ICP child from the odometry guess when
  // visual registration fails. That behavior is useful for odometry but is
  // unsafe for historical constraints: a repetitive corridor can then move
  // the graph without any place-recognition evidence. The pinned upstream
  // patch keeps the default behavior and exposes this strict opt-in.
  set("Reg/ChildFallbackOnFailure",
      profile == WorkerProfile::kOfficial || native_robust ? "false" : "true");
  // Both observations carry compliant metric depth.  Estimate a 3-D-to-3-D
  // transform so a repeated texture cannot pass on image geometry alone;
  // VisIcp then verifies the proposal against the structural depth scan.
  set("Vis/EstimationType", "0");
  set("Vis/InlierDistance", native_robust ? "0.10" : "0.10");
  set("Vis/MinInliers", std::to_string(kVisualLoopMinInliers));
  set("Vis/MinInliersDistribution", "0.001");
  set("Vis/MaxDepth", "3.5");
  // Spatial mapping closures have a continuous qvel/RGB-D graph prediction.
  // Match features around that projection instead of globally matching every
  // repeated door or corridor texture. Lost-tracker localization explicitly
  // removes this window only during its separately confirmed recovery mode.
  set("Vis/CorGuessWinSize", "40");
  if (kornia_sift) {
    set("Kp/DetectorStrategy", "15");
  } else if (native_robust) {
    set("Kp/DetectorStrategy", "8");
  } else {
    // Keep the official detector choice explicit.  Besides documenting the
    // production profile, this makes profile drift visible to the isolation
    // test instead of hiding it in a conditional expression.
    set("Kp/DetectorStrategy", "1");
  }
  set("Kp/MaxFeatures", kornia_sift || native_robust ? "1400" : "800");
  set("Kp/MaxDepth", "3.5");
  // PyDetector buffers the descriptors produced by its most recent call.
  // RTAB-Map's generic grid dispatcher calls a detector once per cell, then
  // asks for descriptors for the combined keypoint list. More than one cell
  // therefore leaves PyDetector with only the final cell's descriptors and
  // the whole signature is discarded. Invoke it once on the complete image.
  set("Kp/GridRows", kornia_sift || native_robust ? "1" : "2");
  set("Kp/GridCols", kornia_sift || native_robust ? "1" : "2");
  // Vocabulary features below this base-frame height are dominated by floor
  // texture, which repeats across otherwise unrelated rooms.  This filter is
  // only applied while constructing the place-recognition signature; visual
  // registration and occupancy keep their lower geometric thresholds below.
  set("Mem/DepthMaskFloorThr", native_robust ? "0.0" : "0.12");
  // Detect on the original head image, then let RTAB-Map remap keypoints and
  // decimate RGB-D only for persistence.  This preserves long-baseline
  // feature geometry without multiplying database image/depth storage.
  set("Mem/ImagePreDecimation", kornia_sift || native_robust ? "1" : "2");
  set("Mem/ImagePostDecimation", native_robust ? "1" : "2");
  set("SIFT/ContrastThreshold", native_robust ? "0.04" : "0.01");
  set("SIFT/RootSIFT", native_robust ? "false" : "true");
  if (kornia_sift) {
    set("Vis/FeatureType", "15");
    set("PyDetector/Path", python_detector_path);
    set("PyDetector/Cuda", "true");
    set("PyMatcher/Path", python_matcher_path);
    set("PyMatcher/Cuda", "true");
    set("PyMatcher/Threshold", "0.80");
  } else if (native_robust) {
    // Keep the detector/descriptor pair identical for memory and graph
    // registration.  ``6`` is GFTT/BRIEF while the robust profile's detector
    // ``8`` is GFTT/ORB; mixing them makes the stored BoW vocabulary and the
    // registration matcher use different descriptor families, which can
    // suppress otherwise valid long-baseline visual revisits.
    set("Vis/FeatureType", "8");
  } else {
    set("Vis/FeatureType", "1");
  }
  set("Vis/GridRows", kornia_sift || native_robust ? "1" : "2");
  set("Vis/GridCols", kornia_sift || native_robust ? "1" : "2");
  set("Vis/DepthMaskFloorThr", native_robust ? "0.0" : "0.12");
  // A null-guess historical closure needs global descriptor matching. Use a
  // CUDA mutual-ratio matcher for the full-resolution SIFT descriptors so a
  // one-way repetitive-texture match cannot seed 3-D RANSAC. With a local
  // qvel guess, RTAB-Map takes its projected-window branch before this matcher.
  set("Vis/CorNNType", kornia_sift ? "6" : "1");
  set("Vis/CorNNDR", "0.80");
  set("Kp/BadSignRatio", native_robust ? "0.5" : "0.025");
  set("Odom/ScanKeyFrameThr", "0.85");

  // Adjacent odometry and the incremental graph keep this conservative ICP
  // profile for the whole session. Soft localization uses the dedicated
  // historical scan matcher and never reparses RTAB-Map parameters.
  set("Icp/Strategy", "0");
  set("Icp/VoxelSize", "0.05");
  set("Icp/RangeMin", "0.45");
  set("Icp/RangeMax", "3.5");
  const bool sparse_icp = profile == WorkerProfile::kSparseIcp;
  set("Icp/MaxCorrespondenceDistance", sparse_icp ? "0.30" : "0.12");
  set("Icp/Iterations", "35");
  set("Icp/CorrespondenceRatio", "0.12");
  // Mapping links stay in the same five-centimetre residual basin as the
  // occupancy grid.  The read-only localizer installs its separate wider
  // bound after freezing, where a bad hypothesis cannot deform structure.
  set("Icp/MaxTranslation", sparse_icp ? "0.75" :
                            native_robust ? "0.20" :
                            profile == WorkerProfile::kSparseRgbd ? "0.25" : "0.05");
  set("Icp/MaxRotation", sparse_icp ? "0.50" :
                         native_robust ? "0.35" :
                         profile == WorkerProfile::kSparseRgbd ? "0.15" : "0.20");
  set("Icp/ReciprocalCorrespondences", "true");
  set("Icp/PointToPlane", "true");
  set("Icp/PointToPlaneK", "10");
  set("Icp/PointToPlaneMinComplexity", "0.15");
  // 走廊等退化结构回退到 RTAB-Map 原生的受约束 point-to-point，未观测
  // 方向沿用 qvel 先验；直接拒绝会让长轨迹退化成纯底盘积分。
  set("Icp/PointToPlaneLowComplexityStrategy", "1");

  // Mapping and conservative loop closure. Global appearance retrieval is
  // supplied causally by the independent GPU sequence index. Disable RTAB's
  // single-image Bayesian proposal here: repetitive floors and furniture can
  // dominate that posterior before metric registration sees the correct old
  // node. External proposals still follow RTAB-Map's unchanged RGB-D/ICP,
  // covariance and graph-consistency acceptance path.
  set("Rtabmap/DetectionRate",
      profile == WorkerProfile::kOfficial || native_robust ? "0" : "2.0");
  set("Rtabmap/LoopThr",
      profile == WorkerProfile::kOfficial ? "1.0" : "0.30");
  set("Rtabmap/MaxRetrieved",
      profile == WorkerProfile::kOfficial ? "5" : native_robust ? "10" : "5");
  set("Rtabmap/PublishStats", "true");
  set("Rtabmap/TimeThr", "0");
  // Query terminal ACKs are crash contracts. RTAB-Map defaults SQLite to an
  // in-memory journal with synchronous writes disabled, under which a returned
  // COMMIT can disappear or corrupt the graph after SIGKILL. DELETE+FULL makes
  // the explicit query flush below durable; this is scene-independent storage
  // correctness, not a mapping threshold.
  set(rtabmap::Parameters::kDbSqlite3JournalMode(), "0");
  set(rtabmap::Parameters::kDbSqlite3Synchronous(), "2");
  set("Mem/IncrementalMemory", "true");
  // Small-displacement observations are deliberately removed from the graph
  // by RTAB-Map after registration.  Keeping those unlinked debug signatures
  // persisted a full compressed RGB, float-depth image and scan for every
  // submitted frame (tens of GB for a graph with only a few dozen nodes) and
  // eventually killed the worker at COMMIT with SQLITE_FULL.  Retain binary
  // data for accepted graph nodes, but physically discard rejected signatures.
  set("Mem/NotLinkedNodesKept", "false");
  set("Mem/IntermediateNodeDataKept", "false");
  // The worker publishes every accepted structural grid and may use the
  // latest accepted node as a QuerySubmap anchor.  Neither can be reconstructed
  // if RTAB-Map removes a node while closing the database.
  set("Mem/ReduceGraph", "false");
  // The worker owns frame-to-frame odometry separately and passes its pose as
  // external odometry to Rtabmap.  Do not ask the graph memory to borrow an
  // odometry feature map configured with a different detector/descriptor pair;
  // that mismatch makes RTAB-Map disable the path at startup and silently
  // removes valid RGB-D words from long-term retrieval.
  set("Mem/UseOdomFeatures", "false");
  set("Mem/LocalizationDataSaved", "false");
  set("Mem/LocalizationReadOnly", "false");
  set("Mem/InitWMWithAllNodes", "false");
  set("Mem/LoadVisualLocalFeaturesOnInit", "true");
  set("Kp/IncrementalDictionary", "true");
  // Keep replay and live decisions reproducible.  RTAB-Map's default
  // parallel feature/compression workers can complete in a different order,
  // which changes the BoW candidate set on otherwise identical observations.
  // This only affects scheduling; it does not add any information source.
  set("Kp/Parallelized", "false");
  set("Mem/CompressionParallelized", "false");
  set("Mem/STMSize", "15");
  // Rehearsal emits GlobalClosure-typed merge records and removes the merged
  // signature from the optimized pose table. Keep every accepted observation
  // explicit while mapping; the irreversible freeze bounds long-run growth.
  set("Mem/RehearsalSimilarity", "1.0");
  set("RGBD/Enabled", "true");
  set("RGBD/ForceOdom3DoF", "true");
  // Preserve RTAB-Map's graph sampling boundary. The worker independently
  // verifies below that a processed signature survived this displacement gate
  // before publishing its pose or local grid.
  set("RGBD/LinearUpdate", "0.05");
  set("RGBD/AngularUpdate", "0.034906585");
  set("RGBD/LocalRadius", "16.0");
  // A proximity lookup is only a proposal.  During mapping keep it single-pair
  // and conservative: overlapping path windows create many correlated links
  // from one physical revisit and can overpower odometry.  The independent
  // GPU rolling-submap verifier supplies long-baseline candidates instead.
  set("RGBD/ProximityBySpace",
      profile == WorkerProfile::kOfficial ? "true" : "false");
  set("RGBD/ProximityByTime", "false");
  // Spatial proximity is already bounded by LocalRadius, MaxPaths and the
  // candidate count. A temporal graph-depth cap would hide an old place after
  // a long traversal even when odometry returns physically near it. RTAB-Map
  // defines zero as "ignore this depth cap"; every resulting candidate still
  // has to pass official VisIcp and graph-consistency verification.
  set("RGBD/ProximityMaxGraphDepth",
      profile == WorkerProfile::kOfficial || native_robust ? "0" : "50");
  set("RGBD/ProximityMaxPaths", "3");
  set("RGBD/ProximityPathMaxNeighbors", "0");
  set("RGBD/ProximityPathRawPosesUsed", "true");
  set("RGBD/ProximityPathFilteringRadius", "4.0");
  // One-to-one spatial candidates are local tracking constraints, not
  // kidnapped-robot hypotheses. Seed native VisIcp from the compliant
  // qvel/RGB-D graph so visual correspondences are projected into the local
  // window above and ICP refines only observable residual axes.
  set("RGBD/ProximityOdomGuess", "true");
  set("RGBD/ProximityAngle", "45");
  // Retrieve a small set rather than only the two nearest local signatures.
  // Repeated corridor observations can otherwise occupy both slots and hide
  // the older view that closes the route. Every retrieved candidate still
  // passes VisIcp and the graph-consistency gate below.
  set("RGBD/MaxLocalRetrieved", "5");
  set("RGBD/ScanMatchingIdsSavedInLinks", "true");
  // Preserve the map frame established by the first node. A loop closure must
  // correct the current robot pose; pinning the newest node instead would move
  // the whole house (including the start) to follow accumulated qvel drift.
  set("RGBD/OptimizeFromGraphEnd", "false");
  // Historical closures may remove more accumulated drift than adjacent-frame
  // odometry, but the resulting graph must fit within half of the link's
  // reported standard deviation. Keeping this stricter than the upstream
  // default blocks internally plausible corridor aliases before commit while
  // the separate 20 cm ICP basin admits genuine long-baseline corrections.
  set("RGBD/OptimizeMaxError", "0.5");
  // Bound the residual basin before a loop/localization is inserted. A first
  // perceptual alias can be internally self-consistent and therefore evade
  // OptimizeMaxError; RTAB-Map's native distance gate prevents such a remote
  // registration from becoming the constraint that bends the graph.
  set("RGBD/MaxLoopClosureDistance", "0.75");
  // Metric registration can report sub-millimetre covariance from a small
  // set of correspondences. A loop must never be more confident than the
  // best odometry edge seen by the same graph, otherwise one accepted revisit
  // can rigidly warp the map and make a later true revisit inconsistent.
  set("RGBD/LoopCovLimited", "true");
  set("RGBD/MaxOdomCacheSize", native_robust ? "10" : "30");
  set("RGBD/LocalizationSmoothing", "true");
  set("RGBD/ProximityGlobalScanMap", "false");
  // The official profile keeps the persisted GPU descriptors used to propose
  // revisits. Other experimental profiles may re-extract registration
  // features from the stored compliant RGB-D observation.
  set("RGBD/LoopClosureReextractFeatures",
      profile == WorkerProfile::kOfficial ? "false" : "true");
  set("RGBD/AggressiveLoopThr",
      profile == WorkerProfile::kOfficial ? "1.0" : "0.30");
  set("RGBD/CreateOccupancyGrid", "true");
  if (native_robust) {
    set("Optimizer/Strategy", native_robust_ceres ? "3" : "0");
  } else {
    set("Optimizer/Strategy", "3");
  }
  set("Optimizer/Iterations", "50");
  set("Optimizer/Epsilon", "0.000001");
  set("Optimizer/Robust", native_robust_ceres ? "true" : "false");
  set("Optimizer/PriorsIgnored", "true");

  // The worker supplies a filtered, compliant 3-D scan in base coordinates.
  // Keep RTAB-Map's grid source on that explicit scan; Grid/Sensor=1 would
  // project the RGB-D image independently and bypass the same range/self
  // filtering used by registration.
  set("Grid/Sensor", "0");
  set("Grid/DepthDecimation", "4");
  set("Grid/RangeMin", "0.45");
  set("Grid/RangeMax", "3.5");
  set("Grid/CellSize", "0.05");
  set("Grid/PreVoxelFiltering", "true");
  set("Grid/NormalsSegmentation", "false");
  set("Grid/MinGroundHeight", "-0.08");
  set("Grid/MaxGroundHeight", "0.08");
  set("Grid/MaxObstacleHeight", "1.95");
  set("Grid/3D", "false");
  set("Grid/RayTracing", "true");
  set("Grid/FootprintLength", "1.90");
  set("Grid/FootprintWidth", "1.10");
  set("Grid/FootprintHeight", "1.95");
  set("GridGlobal/UpdateError", "0.005");
  set("GridGlobal/FootprintRadius", "0.55");
  set("GridGlobal/MinSize", "16.0");
  set("GridGlobal/Eroded", "false");
  set("GridGlobal/OccupancyThr", "0.55");
  return parameters;
}

struct HeightPoint {
  float x;
  float y;
  std::uint8_t band;
};

struct FrameInput {
  br::FrameMeta meta{};
  cv::Mat rgb;
  // Keep the compliant camera depth before structural edge/self filtering.
  // RTAB-Map's appearance and metric RGB-D registration need the original
  // depth support; the filtered copy remains reserved for ICP/grid geometry.
  cv::Mat raw_depth;
  cv::Mat depth;
  rtabmap::Transform camera_to_base;
  rtabmap::LaserScan odometry_scan;
  rtabmap::LaserScan depth_scan;
  std::vector<HeightPoint> recovered_height_points;
  std::vector<HeightPoint> height_points;
  std::vector<HeightPoint> occupancy_height_points;
};

enum class QueryScope : std::uint8_t {
  kNone = 0,
  kRecovery = 1,
  kNormal = 2,
};

// Positive RGB-D bridge measurements are valid only for the frame that
// produced them.  A formal exhaustive no-mode certificate is the only release
// that may survive until a later usable structural frame.
enum class QueryReleaseKind : std::uint8_t {
  kNone = 0,
  kPositiveMetric = 1,
  kNegativeNoMode = 2,
};

struct QueryView {
  std::uint64_t frame_id = 0;
  rtabmap::Transform fused_pose;
  std::size_t odometry_history_index = 0;
  cv::Mat ground;
  cv::Mat obstacles;
  cv::Mat empty;
  std::vector<HeightPoint> height_points;
  std::size_t bytes = 0;
};

struct QueryChunk {
  std::vector<QueryView> views;
  float travel_m = 0.0F;
  float turn_rad = 0.0F;
};

struct ProvisionalQueryBuffer {
  QueryScope scope = QueryScope::kNone;
  std::uint64_t generation = 0;
  int anchor_node_id = 0;
  std::size_t anchor_odometry_history_index = 0;
  std::deque<QueryChunk> chunks;
  std::size_t keyframes = 0;
  std::size_t bytes = 0;
  bool overflowed = false;
  bool promotion_attempted = false;
  QueryReleaseKind latched_release_kind = QueryReleaseKind::kNone;

  bool active() const {
    return scope != QueryScope::kNone && generation > 0;
  }

  void clear() {
    scope = QueryScope::kNone;
    generation = 0;
    anchor_node_id = 0;
    anchor_odometry_history_index = 0;
    chunks.clear();
    keyframes = 0;
    bytes = 0;
    overflowed = false;
    promotion_attempted = false;
    latched_release_kind = QueryReleaseKind::kNone;
  }

  const QueryView *last_view() const {
    return chunks.empty() || chunks.back().views.empty()
               ? nullptr
               : &chunks.back().views.back();
  }
};

#pragma pack(push, 1)
struct LegacyPersistedQueryHeader {
  char magic[8];
  std::uint8_t version;
  std::uint8_t scope;
  std::uint8_t outcome;
  std::uint8_t reserved;
  std::uint64_t generation;
  std::uint32_t height_count;
};

enum class QueryPromotionKind : std::uint8_t {
  kNone = 0,
  kNegative = 1,
  kPositive = 2,
  kBridgeOnly = 3,
};

struct PersistedQueryHeader {
  char magic[8];
  std::uint8_t version;
  std::uint8_t scope;
  std::uint8_t outcome;
  std::uint8_t kind;
  std::uint64_t generation;
  std::uint32_t height_count;
  std::int32_t anchor_id;
  std::int32_t candidate_id;
};

struct PersistedHeightPoint {
  float x;
  float y;
  std::uint8_t band;
  std::uint8_t reserved[3];
};

struct PersistedHeightHeader {
  char magic[8];
  std::uint8_t version;
  std::uint8_t reserved[3];
  std::uint32_t height_count;
  std::uint32_t checksum;
};

struct DurableQueryTerminalRecord {
  char magic[8];
  std::uint8_t version;
  std::uint8_t scope;
  std::uint8_t outcome;
  std::uint8_t reserved;
  std::uint64_t generation;
  std::int32_t anchor_id;
  std::uint32_t checksum;
};
#pragma pack(pop)

static_assert(sizeof(LegacyPersistedQueryHeader) == 24);
static_assert(sizeof(PersistedQueryHeader) == 32);
static_assert(sizeof(PersistedHeightPoint) == 12);
static_assert(sizeof(PersistedHeightHeader) == 20);
static_assert(sizeof(DurableQueryTerminalRecord) == 28);

struct QueryCellEvidence {
  std::uint16_t ground_views = 0;
  std::uint16_t obstacle_views = 0;
  std::uint16_t empty_views = 0;
  std::uint8_t height_bands = 0;
  float first_obstacle_view_x = 0.0F;
  float first_obstacle_view_y = 0.0F;
  float first_obstacle_view_yaw = 0.0F;
  bool intrinsic_polarity_conflict = false;
};

struct NoveltyResumeObstacleEvidence {
  float first_view_x = 0.0F;
  float first_view_y = 0.0F;
  float first_view_yaw = 0.0F;
  bool confirmed = false;
};

struct QueryAggregate {
  cv::Mat ground;
  cv::Mat obstacles;
  cv::Mat empty;
  std::vector<HeightPoint> height_points;
  std::size_t duplicate_cells = 0;
  std::size_t conflict_cells = 0;
  std::size_t unsupported_cells = 0;
  std::size_t accepted_free_cells = 0;
  std::size_t accepted_obstacle_cells = 0;
  std::size_t safe_supports = 0;
  std::size_t unsafe_supports = 0;
  std::size_t unsafe_components = 0;
  std::size_t uncertain_veto_components = 0;
  std::size_t uncertain_veto_cells = 0;
  std::size_t uncertainty_domain_cells = 0;
  std::size_t uncertainty_radius_buckets = 0;
  std::size_t uncertainty_domain_work_cells = 0;
  double promotion_uncertainty_m = 0.0;
  double safe_promotion_uncertainty_m = 0.0;
  bool uncertainty_domain_exceeded = false;
  bool uncertainty_exceeded = false;
  bool valid = false;
  bool all_known = false;
};

constexpr std::array<char, 8> kPersistedQueryMagic{
    'B', 'R', 'Q', 'S', 'U', 'B', '1', '\0'};
constexpr std::array<char, 8> kPersistedHeightMagic{
    'B', 'R', 'H', 'G', 'H', 'T', '1', '\0'};
constexpr std::array<char, 8> kQueryTerminalLedgerMagic{
    'B', 'R', 'Q', 'T', 'E', 'R', 'M', '\0'};
constexpr std::size_t kMaxDurableQueryTerminals = 65536U;
constexpr std::size_t kMaxPersistedHeightPoints = 1U << 20U;

bool matrix_has_magic(
    const cv::Mat &encoded, const std::array<char, 8> &magic) {
  return encoded.type() == CV_8UC1 && encoded.isContinuous() &&
         encoded.total() >= magic.size() &&
         std::memcmp(encoded.data, magic.data(), magic.size()) == 0;
}

bool persisted_query_magic(const cv::Mat &encoded) {
  return matrix_has_magic(encoded, kPersistedQueryMagic);
}

bool persisted_height_magic(const cv::Mat &encoded) {
  return matrix_has_magic(encoded, kPersistedHeightMagic);
}

std::uint32_t persisted_height_checksum(
    const PersistedHeightHeader &header,
    const PersistedHeightPoint *points) {
  std::uint32_t value = 2166136261U;
  const auto update = [&value](const void *source, std::size_t size) {
    const auto *bytes = static_cast<const std::uint8_t *>(source);
    for (std::size_t index = 0; index < size; ++index) {
      value ^= bytes[index];
      value *= 16777619U;
    }
  };
  update(&header, offsetof(PersistedHeightHeader, checksum));
  if (header.height_count > 0U && points != nullptr) {
    update(points, static_cast<std::size_t>(header.height_count) *
                       sizeof(PersistedHeightPoint));
  }
  return value;
}

std::uint32_t query_terminal_checksum(
    const DurableQueryTerminalRecord &record) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(&record);
  std::uint32_t value = 2166136261U;
  for (std::size_t index = 0;
       index < offsetof(DurableQueryTerminalRecord, checksum); ++index) {
    value ^= bytes[index];
    value *= 16777619U;
  }
  return value;
}

br::QueryScope protocol_query_scope(QueryScope scope) {
  return scope == QueryScope::kRecovery
             ? br::QueryScope::kRecovery
             : scope == QueryScope::kNormal ? br::QueryScope::kNormal
                                             : br::QueryScope::kNone;
}

cv::Mat encode_persisted_query(
    QueryScope scope, std::uint64_t generation, br::QueryOutcome outcome,
    QueryPromotionKind kind, int anchor_id, int candidate_id,
    const std::vector<HeightPoint> &height_points) {
  const bool query_node =
      kind == QueryPromotionKind::kNegative ||
      kind == QueryPromotionKind::kPositive;
  const bool valid_query_identity =
      query_node && anchor_id > 0 &&
      ((kind == QueryPromotionKind::kPositive && candidate_id > 0) ||
       (kind == QueryPromotionKind::kNegative && candidate_id == 0));
  const bool valid_bridge_identity =
      kind == QueryPromotionKind::kBridgeOnly && anchor_id > 0 &&
      candidate_id > 0;
  if (scope == QueryScope::kNone || generation == 0U ||
      (outcome != br::QueryOutcome::kHolding &&
       outcome != br::QueryOutcome::kCommitted &&
       outcome != br::QueryOutcome::kBridgeOnlyDiscarded) ||
      ((outcome == br::QueryOutcome::kBridgeOnlyDiscarded)
           ? !valid_bridge_identity
           : !valid_query_identity) ||
      (outcome != br::QueryOutcome::kCommitted && !height_points.empty()) ||
      height_points.size() > std::numeric_limits<std::uint32_t>::max()) {
    return cv::Mat();
  }
  const std::size_t bytes = sizeof(PersistedQueryHeader) +
                            height_points.size() * sizeof(PersistedHeightPoint);
  if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return cv::Mat();
  }
  cv::Mat encoded(static_cast<int>(bytes), 1, CV_8UC1, cv::Scalar(0));
  PersistedQueryHeader header{};
  std::memcpy(header.magic, kPersistedQueryMagic.data(),
              kPersistedQueryMagic.size());
  header.version = 3U;
  header.scope = static_cast<std::uint8_t>(scope);
  header.outcome = static_cast<std::uint8_t>(outcome);
  header.kind = static_cast<std::uint8_t>(kind);
  header.generation = generation;
  header.height_count = static_cast<std::uint32_t>(height_points.size());
  header.anchor_id = anchor_id;
  header.candidate_id = candidate_id;
  std::memcpy(encoded.data, &header, sizeof(header));
  auto *destination = reinterpret_cast<PersistedHeightPoint *>(
      encoded.data + sizeof(PersistedQueryHeader));
  for (std::size_t index = 0; index < height_points.size(); ++index) {
    destination[index].x = height_points[index].x;
    destination[index].y = height_points[index].y;
    destination[index].band = height_points[index].band;
  }
  return encoded;
}

cv::Mat encode_persisted_height(
    const std::vector<HeightPoint> &height_points) {
  if (height_points.empty() ||
      height_points.size() > kMaxPersistedHeightPoints) {
    return cv::Mat();
  }
  const std::size_t bytes = sizeof(PersistedHeightHeader) +
                            height_points.size() * sizeof(PersistedHeightPoint);
  if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return cv::Mat();
  }
  cv::Mat encoded(static_cast<int>(bytes), 1, CV_8UC1, cv::Scalar(0));
  PersistedHeightHeader header{};
  std::memcpy(header.magic, kPersistedHeightMagic.data(),
              kPersistedHeightMagic.size());
  header.version = 1U;
  header.height_count =
      static_cast<std::uint32_t>(height_points.size());
  auto *destination = reinterpret_cast<PersistedHeightPoint *>(
      encoded.data + sizeof(PersistedHeightHeader));
  for (std::size_t index = 0; index < height_points.size(); ++index) {
    if (!std::isfinite(height_points[index].x) ||
        !std::isfinite(height_points[index].y) ||
        height_points[index].band >= kWallBandCount) {
      return cv::Mat();
    }
    destination[index].x = height_points[index].x;
    destination[index].y = height_points[index].y;
    destination[index].band = height_points[index].band;
  }
  header.checksum = persisted_height_checksum(header, destination);
  std::memcpy(encoded.data, &header, sizeof(header));
  return encoded;
}

bool decode_persisted_height_data(
    const cv::Mat &encoded, std::vector<HeightPoint> *height_points) {
  if (!persisted_height_magic(encoded) || height_points == nullptr ||
      encoded.total() < sizeof(PersistedHeightHeader)) {
    return false;
  }
  PersistedHeightHeader header{};
  std::memcpy(&header, encoded.data, sizeof(header));
  if (header.version != 1U || header.reserved[0] != 0U ||
      header.reserved[1] != 0U || header.reserved[2] != 0U ||
      header.height_count == 0U ||
      header.height_count > kMaxPersistedHeightPoints) {
    return false;
  }
  const std::size_t expected =
      sizeof(PersistedHeightHeader) +
      static_cast<std::size_t>(header.height_count) *
          sizeof(PersistedHeightPoint);
  if (expected != encoded.total()) {
    return false;
  }
  const auto *source = reinterpret_cast<const PersistedHeightPoint *>(
      encoded.data + sizeof(PersistedHeightHeader));
  if (header.checksum != persisted_height_checksum(header, source)) {
    return false;
  }
  std::vector<HeightPoint> decoded;
  decoded.reserve(header.height_count);
  for (std::size_t index = 0; index < header.height_count; ++index) {
    if (!std::isfinite(source[index].x) ||
        !std::isfinite(source[index].y) ||
        source[index].band >= kWallBandCount ||
        source[index].reserved[0] != 0U ||
        source[index].reserved[1] != 0U ||
        source[index].reserved[2] != 0U) {
      return false;
    }
    decoded.push_back(
        {source[index].x, source[index].y, source[index].band});
  }
  *height_points = std::move(decoded);
  return true;
}

bool decode_persisted_query_data(
    const cv::Mat &encoded, QueryScope *scope,
    std::uint64_t *generation, br::QueryOutcome *outcome,
    std::vector<HeightPoint> *height_points,
    QueryPromotionKind *kind = nullptr, int *anchor_id = nullptr,
    int *candidate_id = nullptr) {
  if (encoded.type() != CV_8UC1 || !encoded.isContinuous() ||
      encoded.total() < sizeof(LegacyPersistedQueryHeader) || scope == nullptr ||
      generation == nullptr || outcome == nullptr || height_points == nullptr) {
    return false;
  }
  LegacyPersistedQueryHeader legacy{};
  std::memcpy(&legacy, encoded.data, sizeof(legacy));
  if (std::memcmp(legacy.magic, kPersistedQueryMagic.data(),
                  kPersistedQueryMagic.size()) != 0 ||
      (legacy.version != 1U && legacy.version != 2U &&
       legacy.version != 3U)) {
    return false;
  }
  std::size_t header_size = sizeof(LegacyPersistedQueryHeader);
  QueryPromotionKind decoded_kind = QueryPromotionKind::kNone;
  int decoded_anchor_id = 0;
  int decoded_candidate_id = 0;
  if (legacy.version == 3U) {
    if (encoded.total() < sizeof(PersistedQueryHeader)) {
      return false;
    }
    PersistedQueryHeader current{};
    std::memcpy(&current, encoded.data, sizeof(current));
    decoded_kind = static_cast<QueryPromotionKind>(current.kind);
    decoded_anchor_id = current.anchor_id;
    decoded_candidate_id = current.candidate_id;
    header_size = sizeof(PersistedQueryHeader);
  }
  const std::size_t expected = header_size +
      static_cast<std::size_t>(legacy.height_count) *
          sizeof(PersistedHeightPoint);
  const bool query_node = decoded_kind == QueryPromotionKind::kNegative ||
                          decoded_kind == QueryPromotionKind::kPositive;
  const bool valid_query_identity =
      query_node && decoded_anchor_id > 0 &&
      ((decoded_kind == QueryPromotionKind::kPositive &&
        decoded_candidate_id > 0) ||
       (decoded_kind == QueryPromotionKind::kNegative &&
        decoded_candidate_id == 0));
  const bool valid_bridge_identity =
      decoded_kind == QueryPromotionKind::kBridgeOnly &&
      decoded_anchor_id > 0 && decoded_candidate_id > 0;
  if (expected != encoded.total() || legacy.generation == 0U ||
      (legacy.scope != static_cast<std::uint8_t>(QueryScope::kRecovery) &&
       legacy.scope != static_cast<std::uint8_t>(QueryScope::kNormal)) ||
      (legacy.version == 1U &&
       legacy.outcome !=
           static_cast<std::uint8_t>(br::QueryOutcome::kCommitted)) ||
      (legacy.version >= 2U &&
       legacy.outcome != static_cast<std::uint8_t>(br::QueryOutcome::kHolding) &&
       legacy.outcome != static_cast<std::uint8_t>(br::QueryOutcome::kCommitted) &&
       legacy.outcome != static_cast<std::uint8_t>(
                             br::QueryOutcome::kBridgeOnlyDiscarded)) ||
      (legacy.version == 3U &&
       ((legacy.outcome == static_cast<std::uint8_t>(
                               br::QueryOutcome::kBridgeOnlyDiscarded))
            ? !valid_bridge_identity
            : !valid_query_identity)) ||
      (legacy.outcome != static_cast<std::uint8_t>(
                             br::QueryOutcome::kCommitted) &&
       legacy.height_count != 0U)) {
    return false;
  }
  std::vector<HeightPoint> decoded;
  decoded.reserve(legacy.height_count);
  const auto *source = reinterpret_cast<const PersistedHeightPoint *>(
      encoded.data + header_size);
  for (std::size_t index = 0; index < legacy.height_count; ++index) {
    if (!std::isfinite(source[index].x) ||
        !std::isfinite(source[index].y) ||
        source[index].band >= kWallBandCount) {
      return false;
    }
    decoded.push_back(
        {source[index].x, source[index].y, source[index].band});
  }
  *scope = static_cast<QueryScope>(legacy.scope);
  *generation = legacy.generation;
  *outcome = static_cast<br::QueryOutcome>(legacy.outcome);
  *height_points = std::move(decoded);
  if (kind != nullptr) {
    *kind = decoded_kind;
  }
  if (anchor_id != nullptr) {
    *anchor_id = decoded_anchor_id;
  }
  if (candidate_id != nullptr) {
    *candidate_id = decoded_candidate_id;
  }
  return true;
}

bool decode_persisted_query(
    const rtabmap::SensorData &data, QueryScope *scope,
    std::uint64_t *generation, br::QueryOutcome *outcome,
    std::vector<HeightPoint> *height_points,
    QueryPromotionKind *kind = nullptr, int *anchor_id = nullptr,
    int *candidate_id = nullptr) {
  cv::Mat encoded;
  data.uncompressDataConst(nullptr, nullptr, nullptr, &encoded);
  return decode_persisted_query_data(
      encoded, scope, generation, outcome, height_points, kind, anchor_id,
      candidate_id);
}

std::array<float, 3> transform_point(const br::FrameMeta &meta, float x, float y, float z) {
  const double *t = meta.camera_to_base;
  return {
      static_cast<float>(t[0] * x + t[1] * y + t[2] * z + t[3]),
      static_cast<float>(t[4] * x + t[5] * y + t[6] * z + t[7]),
      static_cast<float>(t[8] * x + t[9] * y + t[10] * z + t[11]),
  };
}

void validate_transform(const br::FrameMeta &meta) {
  for (double value : meta.camera_to_base) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("camera transform contains a non-finite value");
    }
  }
  const double *r = meta.camera_to_base;
  for (int row = 0; row < 3; ++row) {
    double norm = 0.0;
    for (int column = 0; column < 3; ++column) {
      norm += r[row * 4 + column] * r[row * 4 + column];
    }
    if (std::abs(norm - 1.0) > 1e-3) {
      throw std::invalid_argument("camera transform rotation is not orthonormal");
    }
  }
  const double determinant =
      r[0] * (r[5] * r[10] - r[6] * r[9]) -
      r[1] * (r[4] * r[10] - r[6] * r[8]) +
      r[2] * (r[4] * r[9] - r[5] * r[8]);
  if (std::abs(determinant - 1.0) > 1e-3) {
    throw std::invalid_argument("camera transform rotation must be proper");
  }
}

bool structural_observation_usable(const FrameInput &frame) {
  const double *transform = frame.meta.camera_to_base;
  // RTAB optical +Z 轴是相机视线，这里只读当前帧的机体相对外参。
  const float horizontal_view = static_cast<float>(
      std::hypot(transform[2], transform[6]));
  const float camera_height_m = static_cast<float>(transform[11]);
  const float planar_offset_m = static_cast<float>(
      std::hypot(transform[3], transform[7]));
  return mapping_view_is_usable(horizontal_view, camera_height_m,
                                planar_offset_m);
}

std::uint64_t height_cell_key(int x, int y, std::uint8_t band) {
  constexpr std::uint64_t kCoordinateMask = (1ULL << 30U) - 1ULL;
  const std::uint64_t ux = static_cast<std::uint32_t>(x) & kCoordinateMask;
  const std::uint64_t uy = static_cast<std::uint32_t>(y) & kCoordinateMask;
  return (static_cast<std::uint64_t>(band) << 60U) | (ux << 30U) | uy;
}

std::uint8_t obstacle_height_band(float height_m) {
  const float normalized_height =
      (height_m - kObstacleMinM) / (kObstacleMaxM - kObstacleMinM);
  return static_cast<std::uint8_t>(std::min(
      static_cast<int>(kWallBandCount) - 1,
      std::max(0, static_cast<int>(normalized_height * kWallBandCount))));
}

std::size_t self_zone_cell_index(int cell_x, int cell_y) {
  if (cell_x < kSelfGridMinX || cell_x > kSelfGridMaxX ||
      cell_y < kSelfGridMinY || cell_y > kSelfGridMaxY) {
    return kSelfGridCells;
  }
  return static_cast<std::size_t>(cell_y - kSelfGridMinY) *
             kSelfGridWidth +
         static_cast<std::size_t>(cell_x - kSelfGridMinX);
}

void filter_depth(FrameInput &frame) {
  const int width = frame.depth.cols;
  const int height = frame.depth.rows;
  cv::Mat source = frame.depth.clone();
  cv::Mat edge = cv::Mat::zeros(height, width, CV_8U);
  for (int row = 0; row < height; ++row) {
    const float *values = source.ptr<float>(row);
    for (int column = 0; column < width; ++column) {
      const float value = values[column];
      if (!(std::isfinite(value) && value > 0.0F)) {
        edge.at<std::uint8_t>(row, column) = 1;
        continue;
      }
      if (column > 0) {
        const float neighbor = values[column - 1];
        if (neighbor > 0.0F && std::abs(value - neighbor) > kDepthEdgeJumpM) {
          edge.at<std::uint8_t>(row, column) = 1;
          edge.at<std::uint8_t>(row, column - 1) = 1;
        }
      }
      if (row > 0) {
        const float neighbor = source.at<float>(row - 1, column);
        if (neighbor > 0.0F && std::abs(value - neighbor) > kDepthEdgeJumpM) {
          edge.at<std::uint8_t>(row, column) = 1;
          edge.at<std::uint8_t>(row - 1, column) = 1;
        }
      }
    }
  }
  cv::dilate(edge, edge, cv::Mat::ones(3, 3, CV_8U));

  // The whole-body envelope suppresses arm returns, but it is much larger
  // than the R1Pro base and can contain a real wall while the chassis still
  // has clearance. Recover only grounded, vertically continuous structure
  // from that ambiguous zone. Articulated links normally occupy isolated
  // height bands and remain filtered.
  std::array<std::uint8_t, kSelfGridCells> self_zone_bands{};
  std::array<std::array<cv::Vec2f, kWallBandCount>, kSelfGridCells>
      self_zone_representatives{};
  for (int row = 0; row < height; row += kSelfSupportRowStride) {
    const float *values = source.ptr<float>(row);
    for (int column = 0; column < width; column += kDepthStride) {
      const float depth = values[column];
      if (edge.at<std::uint8_t>(row, column) ||
          !(std::isfinite(depth) && depth > 0.0F)) {
        continue;
      }
      const float optical_x =
          static_cast<float>((column - frame.meta.cx) / frame.meta.fx) * depth;
      const float optical_y =
          static_cast<float>((row - frame.meta.cy) / frame.meta.fy) * depth;
      const auto base = transform_point(frame.meta, optical_x, optical_y, depth);
      const float radial = std::hypot(base[0], base[1]);
      if (radial < kMinRangeM || radial > kMaxRangeM ||
          base[0] > kSelfForwardM || std::abs(base[1]) > kSelfHalfWidthM ||
          base[2] < kObstacleMinM || base[2] > kObstacleMaxM) {
        continue;
      }
      const int cell_x = static_cast<int>(std::floor(base[0] / kGridCellM));
      const int cell_y = static_cast<int>(std::floor(base[1] / kGridCellM));
      const std::size_t support_index =
          self_zone_cell_index(cell_x, cell_y);
      if (support_index < self_zone_bands.size()) {
        const std::uint8_t band = obstacle_height_band(base[2]);
        const std::uint8_t bit = static_cast<std::uint8_t>(1U << band);
        if ((self_zone_bands[support_index] & bit) == 0U) {
          self_zone_representatives[support_index][band] =
              cv::Vec2f(base[0], base[1]);
        }
        self_zone_bands[support_index] |= bit;
      }
    }
  }
  const std::uint8_t grounded_mask = static_cast<std::uint8_t>(
      (1U << kWallMinGroundedRun) - 1U);
  const std::uint8_t elevated_mask = static_cast<std::uint8_t>(
      ((1U << kWallBandCount) - 1U) & ~grounded_mask);
  for (std::size_t index = 0; index < self_zone_bands.size(); ++index) {
    const std::uint8_t bands = self_zone_bands[index];
    if ((bands & grounded_mask) != grounded_mask ||
        (bands & elevated_mask) == 0U) {
      continue;
    }
    for (std::uint8_t band = 0; band < kWallBandCount; ++band) {
      if ((bands & static_cast<std::uint8_t>(1U << band)) == 0U) {
        continue;
      }
      const cv::Vec2f &representative =
          self_zone_representatives[index][band];
      frame.recovered_height_points.push_back(
          {representative[0], representative[1], band});
    }
  }

  for (int row = 0; row < height; ++row) {
    float *values = frame.depth.ptr<float>(row);
    for (int column = 0; column < width; ++column) {
      float &depth = values[column];
      if (edge.at<std::uint8_t>(row, column) || !(std::isfinite(depth) && depth > 0.0F)) {
        depth = 0.0F;
        continue;
      }
      const float optical_x = static_cast<float>((column - frame.meta.cx) / frame.meta.fx) * depth;
      const float optical_y = static_cast<float>((row - frame.meta.cy) / frame.meta.fy) * depth;
      const auto base = transform_point(frame.meta, optical_x, optical_y, depth);
      const float radial = std::hypot(base[0], base[1]);
      const bool self_zone =
          base[0] <= kSelfForwardM &&
          std::abs(base[1]) <= kSelfHalfWidthM && base[2] > kFloorMaxM;
      if (radial < kMinRangeM || radial > kMaxRangeM ||
          self_zone) {
        depth = 0.0F;
      }
    }
  }
}

void extract_geometry(FrameInput &frame) {
  std::vector<cv::Vec3f> registration_scan_points;
  std::unordered_set<std::uint64_t> occupied;
  for (int row = 0; row < frame.depth.rows; row += kDepthStride) {
    const float *values = frame.depth.ptr<float>(row);
    for (int column = 0; column < frame.depth.cols; column += kDepthStride) {
      const float depth = values[column];
      if (depth <= 0.0F) {
        continue;
      }
      const float optical_x = static_cast<float>((column - frame.meta.cx) / frame.meta.fx) * depth;
      const float optical_y = static_cast<float>((row - frame.meta.cy) / frame.meta.fy) * depth;
      const auto base = transform_point(frame.meta, optical_x, optical_y, depth);
      // Keep the scan in the robot-base frame.  The identity local transform
      // is intentional: this is the same convention used by RTAB-Map's
      // already-transformed cloud path and avoids applying the camera extrinsic
      // twice when LocalGridMaker reconstructs the occupancy cells.
      // Preserve the full observed floor-to-head geometry for registration.
      // A horizontal floor does not constrain planar yaw by itself, but it
      // keeps enough overlap through turns for the wall/furniture returns to
      // determine the observable axes. Occupancy classification remains in
      // the separate height-band path below.
      if (base[2] >= -kFloorMaxM && base[2] <= kObstacleMaxM) {
        registration_scan_points.emplace_back(base[0], base[1], base[2]);
      }
      if (base[2] < kObstacleMinM || base[2] > kObstacleMaxM) {
        continue;
      }
      const std::uint8_t band = obstacle_height_band(base[2]);
      const int cell_x = static_cast<int>(std::floor(base[0] / kGridCellM));
      const int cell_y = static_cast<int>(std::floor(base[1] / kGridCellM));
      if (occupied.insert(height_cell_key(cell_x, cell_y, band)).second) {
        frame.height_points.push_back({base[0], base[1], band});
        frame.occupancy_height_points.push_back({base[0], base[1], band});
      }
    }
  }
  // Nearby grounded structure is display occupancy evidence only. Never append
  // it to a LaserScan or the registration height set: those feed RTAB graph
  // constraints, novelty decisions and the read-only matcher, whose behavior
  // must remain identical to v47.
  for (const HeightPoint &point : frame.recovered_height_points) {
    const int cell_x = static_cast<int>(std::floor(point.x / kGridCellM));
    const int cell_y = static_cast<int>(std::floor(point.y / kGridCellM));
    if (!occupied.insert(height_cell_key(cell_x, cell_y, point.band)).second) {
      continue;
    }
    frame.occupancy_height_points.push_back(point);
  }
  if (!registration_scan_points.empty()) {
    cv::Mat registration_scan_data(
        1, static_cast<int>(registration_scan_points.size()), CV_32FC3,
        registration_scan_points.data());
    frame.odometry_scan = rtabmap::LaserScan(
        registration_scan_data.clone(),
        static_cast<int>(registration_scan_points.size()), kMaxRangeM,
        rtabmap::LaserScan::kXYZ, rtabmap::Transform::getIdentity());
    // Points are already expressed in the robot-base frame. Keep the
    // LaserScan local transform identity so RTAB-Map does not apply the
    // camera extrinsic a second time while building the local grid.
    frame.depth_scan = rtabmap::LaserScan(
        registration_scan_data.clone(),
        static_cast<int>(registration_scan_points.size()), kMaxRangeM,
        rtabmap::LaserScan::kXYZ, rtabmap::Transform::getIdentity());
  }
}

std::vector<HeightPoint> height_points_from_scan(
    const rtabmap::LaserScan &scan) {
  std::vector<HeightPoint> output;
  const cv::Mat &data = scan.data();
  if (scan.empty() || scan.is2d() || data.depth() != CV_32F ||
      data.channels() < 3) {
    return output;
  }
  const cv::Mat flattened = data.isContinuous() ? data.reshape(1, 1)
                                                 : data.clone().reshape(1, 1);
  const float *values = flattened.ptr<float>(0);
  const int channels = data.channels();
  const rtabmap::Transform local = scan.localTransform().isNull()
                                       ? rtabmap::Transform::getIdentity()
                                       : scan.localTransform();
  std::unordered_set<std::uint64_t> occupied;
  output.reserve(static_cast<std::size_t>(data.total()));
  for (std::size_t index = 0; index < data.total(); ++index) {
    const float scan_x = values[index * static_cast<std::size_t>(channels)];
    const float scan_y =
        values[index * static_cast<std::size_t>(channels) + 1U];
    const float scan_z =
        values[index * static_cast<std::size_t>(channels) + 2U];
    if (!std::isfinite(scan_x) || !std::isfinite(scan_y) ||
        !std::isfinite(scan_z)) {
      continue;
    }
    const float base_x = local.r11() * scan_x + local.r12() * scan_y +
                         local.r13() * scan_z + local.x();
    const float base_y = local.r21() * scan_x + local.r22() * scan_y +
                         local.r23() * scan_z + local.y();
    const float base_z = local.r31() * scan_x + local.r32() * scan_y +
                         local.r33() * scan_z + local.z();
    if (base_z < kObstacleMinM || base_z > kObstacleMaxM) {
      continue;
    }
    const std::uint8_t band = obstacle_height_band(base_z);
    const int cell_x = static_cast<int>(std::floor(base_x / kGridCellM));
    const int cell_y = static_cast<int>(std::floor(base_y / kGridCellM));
    if (occupied.insert(height_cell_key(cell_x, cell_y, band)).second) {
      output.push_back({base_x, base_y, band});
    }
  }
  return output;
}

std::vector<HeightPoint> recovered_height_points_from_persisted(
    const std::vector<HeightPoint> &registration,
    const std::vector<HeightPoint> &occupancy) {
  if (registration.empty() || occupancy.empty()) {
    return {};
  }
  std::unordered_set<std::uint64_t> registration_cells;
  registration_cells.reserve(registration.size());
  for (const HeightPoint &point : registration) {
    const int cell_x = static_cast<int>(std::floor(point.x / kGridCellM));
    const int cell_y = static_cast<int>(std::floor(point.y / kGridCellM));
    registration_cells.insert(height_cell_key(cell_x, cell_y, point.band));
  }
  std::vector<HeightPoint> recovered;
  recovered.reserve(occupancy.size());
  for (const HeightPoint &point : occupancy) {
    const int cell_x = static_cast<int>(std::floor(point.x / kGridCellM));
    const int cell_y = static_cast<int>(std::floor(point.y / kGridCellM));
    if (registration_cells.find(
            height_cell_key(cell_x, cell_y, point.band)) ==
        registration_cells.end()) {
      recovered.push_back(point);
    }
  }
  return recovered;
}

FrameInput decode_frame(const std::vector<std::uint8_t> &payload) {
  if (payload.size() < sizeof(br::FrameMeta)) {
    throw std::invalid_argument("frame metadata is truncated");
  }
  FrameInput frame;
  std::memcpy(&frame.meta, payload.data(), sizeof(frame.meta));
  const auto &meta = frame.meta;
  if (meta.width == 0 || meta.height == 0 || meta.width > kMaxImageDimension ||
      meta.height > kMaxImageDimension) {
    throw std::invalid_argument("invalid image dimensions");
  }
  if (!(std::isfinite(meta.stamp) && std::isfinite(meta.fx) && std::isfinite(meta.fy) &&
        std::isfinite(meta.cx) && std::isfinite(meta.cy)) ||
      meta.stamp < 0.0 || meta.fx <= 0.0 || meta.fy <= 0.0) {
    throw std::invalid_argument("invalid frame timestamp or intrinsics");
  }
  if (!(std::isfinite(meta.odom_dx) && std::isfinite(meta.odom_dy) &&
        std::isfinite(meta.odom_dyaw)) ||
      // This delta accumulates across rejected or deliberately sparse camera
      // frames.  A valid body-velocity integral can therefore exceed one
      // ordinary camera-frame step without relaxing ICP's 5 cm residual gate.
      std::hypot(meta.odom_dx, meta.odom_dy) > 10.0 || std::abs(meta.odom_dyaw) > 3.2) {
    throw std::invalid_argument("invalid odometry delta");
  }
  if (meta.external_loop_candidate_id < 0) {
    throw std::invalid_argument("invalid external loop candidate id");
  }
  if ((meta.frame_flags & ~kKnownFrameFlags) != 0U) {
    throw std::invalid_argument("frame contains unknown control flags");
  }
  if ((meta.frame_flags & kFrameRecoveryRelocalization) != 0U &&
      ((meta.frame_flags & kFrameRecoveryHold) == 0U ||
       meta.external_loop_candidate_id <= 0)) {
    throw std::invalid_argument(
        "recovery relocalization requires a held metric hypothesis");
  }
  if ((meta.frame_flags & kFrameVerifiedGraphBridge) != 0U &&
      (meta.external_loop_candidate_id <= 0 ||
       (meta.frame_flags & kFrameRecoveryRelocalization) != 0U)) {
    throw std::invalid_argument(
        "normal verified graph bridge requires an exclusive metric hypothesis");
  }
  if ((meta.frame_flags & kFrameNormalGlobalNoMode) != 0U &&
      (meta.frame_flags & kFrameNormalGlobalSearchPending) == 0U) {
    throw std::invalid_argument(
        "normal global no-mode requires a pending search");
  }
  if ((meta.frame_flags & kFrameRecoveryGlobalNoMode) != 0U &&
      (meta.frame_flags & kFrameRecoveryHold) == 0U) {
    throw std::invalid_argument(
        "recovery global no-mode requires a recovery hold");
  }
  if ((meta.frame_flags & kFrameRecoveryHold) != 0U &&
      (meta.frame_flags & kFrameNormalGlobalSearchPending) != 0U) {
    throw std::invalid_argument("recovery and normal query holds are exclusive");
  }
  if ((meta.frame_flags &
       (kFrameRecoveryHold | kFrameNormalGlobalSearchPending)) != 0U &&
      (meta.query_generation == 0U ||
       (meta.query_generation >> 16U) == 0U ||
       (meta.query_generation & 0xffffU) == 0U)) {
    throw std::invalid_argument(
        "query hold requires a nonce and positive transaction counter");
  }
  if (meta.external_loop_candidate_id > 0) {
    for (double value : meta.external_loop_candidate_to_current) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("external loop pose is not finite");
      }
    }
    for (double value : meta.external_loop_covariance) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("external loop covariance is not finite");
      }
    }
  }
  validate_transform(meta);
  const std::uint64_t pixels = static_cast<std::uint64_t>(meta.width) * meta.height;
  const std::uint64_t expected = sizeof(br::FrameMeta) + pixels * 3ULL + pixels * 4ULL;
  if (payload.size() != expected) {
    throw std::invalid_argument("frame byte count does not match dimensions");
  }
  const std::uint8_t *rgb_data = payload.data() + sizeof(br::FrameMeta);
  const std::uint8_t *depth_data = rgb_data + pixels * 3ULL;
  cv::Mat rgb_input(static_cast<int>(meta.height), static_cast<int>(meta.width), CV_8UC3,
                    const_cast<std::uint8_t *>(rgb_data));
  cv::cvtColor(rgb_input, frame.rgb, cv::COLOR_RGB2BGR);
  cv::Mat depth_input(static_cast<int>(meta.height), static_cast<int>(meta.width), CV_32FC1,
                      const_cast<std::uint8_t *>(depth_data));
  frame.depth = depth_input.clone();
  frame.raw_depth = frame.depth.clone();
  frame.camera_to_base = rtabmap::Transform(
      static_cast<float>(meta.camera_to_base[0]), static_cast<float>(meta.camera_to_base[1]),
      static_cast<float>(meta.camera_to_base[2]), static_cast<float>(meta.camera_to_base[3]),
      static_cast<float>(meta.camera_to_base[4]), static_cast<float>(meta.camera_to_base[5]),
      static_cast<float>(meta.camera_to_base[6]), static_cast<float>(meta.camera_to_base[7]),
      static_cast<float>(meta.camera_to_base[8]), static_cast<float>(meta.camera_to_base[9]),
      static_cast<float>(meta.camera_to_base[10]), static_cast<float>(meta.camera_to_base[11]));
  filter_depth(frame);
  extract_geometry(frame);
  return frame;
}

struct ProcessResult {
  br::ResponseMeta meta{};
  std::vector<std::int8_t> occupancy;
  std::vector<std::uint8_t> low;
  std::vector<std::uint8_t> high;
  std::vector<br::PoseRecord> poses;
};

struct ReadOnlyRgbdRegistrationResult {
  int candidate_id = 0;
  rtabmap::Transform candidate_to_current;
  cv::Mat planar_covariance;
  rtabmap::RegistrationInfo info;
};

struct CoverageSample {
  std::uint32_t mapping_frame = 0;
  std::size_t known_cells = 0;
  std::size_t boundary_cells = 0;
  std::size_t frontier_cells = 0;
  float travel_m = 0.0F;
  float rotation_rad = 0.0F;
  float observation_novelty_ratio = 1.0F;
  float observation_endpoint_novelty_ratio = 1.0F;
  float observation_ray_novelty_ratio = 1.0F;
  std::map<int, rtabmap::Transform> graph_poses;
};

struct ConvergenceEvidence {
  std::size_t known_cells = 0;
  std::size_t boundary_cells = 0;
  std::size_t frontier_cells = 0;
  std::size_t recent_visual_revisits = 0;
  // Unlike recent_visual_revisits (a diagnostic sliding window), this is the
  // cumulative number of deduplicated native loop events in this session.
  std::size_t accepted_loop_events = 0;
  std::size_t visual_loop_regions = 0;
  float max_loop_node_span_ratio = 0.0F;
  float max_loop_motion_span_ratio = 0.0F;
  float known_growth_ratio = std::numeric_limits<float>::infinity();
  float recent_known_growth_ratio = std::numeric_limits<float>::infinity();
  float observation_novelty_ratio = 1.0F;
  float observation_endpoint_novelty_ratio = 1.0F;
  float observation_ray_novelty_ratio = 1.0F;
  float frontier_ratio = std::numeric_limits<float>::infinity();
  float graph_window_translation_m = std::numeric_limits<float>::infinity();
  float graph_window_yaw_rad = std::numeric_limits<float>::infinity();
  std::size_t graph_common_nodes = 0;
  bool window_complete = false;
  bool recent_window_complete = false;
  bool graph_window_complete = false;
  bool ready = false;
};

struct GraphShapeChange {
  float translation_m = std::numeric_limits<float>::infinity();
  float yaw_rad = std::numeric_limits<float>::infinity();
  std::size_t common_nodes = 0;
  bool valid = false;
};

void verify_feature_backend(FeatureBackend feature_backend,
                            const std::string &python_detector_path,
                            const std::string &python_matcher_path) {
  if (feature_backend != FeatureBackend::kKorniaSift) {
    return;
  }
  if (!rtabmap::Feature2D::isAvailable(
          rtabmap::Feature2D::kFeaturePyDetector)) {
    throw std::runtime_error(
        "kornia-sift requires an RTAB-Map build with WITH_PYTHON=ON");
  }
  if (python_matcher_path.empty() ||
      ::access(python_matcher_path.c_str(), R_OK) != 0) {
    throw std::runtime_error(
        "kornia-sift requires a readable CUDA matcher script");
  }

  rtabmap::ParametersMap parameters;
  parameters[rtabmap::Parameters::kKpDetectorStrategy()] = "15";
  parameters[rtabmap::Parameters::kKpMaxFeatures()] = "256";
  parameters[rtabmap::Parameters::kPyDetectorPath()] = python_detector_path;
  parameters[rtabmap::Parameters::kPyDetectorCuda()] = "true";
  std::unique_ptr<rtabmap::Feature2D> detector(
      rtabmap::Feature2D::create(
          rtabmap::Feature2D::kFeaturePyDetector, parameters));
  if (!detector) {
    throw std::runtime_error("failed to create RTAB-Map PyDetector");
  }

  cv::Mat probe(192, 256, CV_8U);
  for (int row = 0; row < probe.rows; ++row) {
    auto *pixels = probe.ptr<std::uint8_t>(row);
    for (int column = 0; column < probe.cols; ++column) {
      const int checker = ((row / 12) + (column / 12)) % 2;
      pixels[column] = static_cast<std::uint8_t>(
          (row * 17 + column * 37 + checker * 113) & 0xff);
    }
  }
  std::vector<cv::KeyPoint> keypoints = detector->generateKeypoints(probe);
  cv::Mat descriptors = detector->generateDescriptors(probe, keypoints);
  if (keypoints.empty() || descriptors.empty() ||
      descriptors.rows != static_cast<int>(keypoints.size()) ||
      descriptors.type() != CV_32FC1 || descriptors.cols != 128) {
    throw std::runtime_error(
        "kornia-sift CUDA startup probe returned invalid features: keypoints=" +
        std::to_string(keypoints.size()) +
        " descriptor_rows=" + std::to_string(descriptors.rows) +
        " descriptor_cols=" + std::to_string(descriptors.cols) +
        " descriptor_type=" + std::to_string(descriptors.type()));
  }
}

class SlamWorker {
 public:
  SlamWorker(std::string database_path,
             WorkerProfile profile,
             FeatureBackend feature_backend,
             std::string python_detector_path,
             std::string python_matcher_path)
      : database_path_(std::move(database_path)),
        profile_(profile),
        feature_backend_(feature_backend),
        python_detector_path_(std::move(python_detector_path)),
        python_matcher_path_(std::move(python_matcher_path)) {
    if (!rtabmap::Optimizer::isAvailable(rtabmap::Optimizer::kTypeCeres)) {
      throw std::runtime_error(
          "RTAB-Map worker requires the compiled Ceres optimizer backend");
    }
    verify_feature_backend(
        feature_backend_, python_detector_path_, python_matcher_path_);
    if (database_path_.empty()) {
      database_path_ = "/tmp/behavior_rtabmap_worker_" +
                       std::to_string(static_cast<long long>(::getpid())) +
                       ".db";
      owns_database_ = true;
      std::remove(database_path_.c_str());
    }
    if (::access(database_path_.c_str(), F_OK) != 0) {
      std::remove(query_terminal_ledger_path().c_str());
    }
    reset();
  }

  ~SlamWorker() {
    if (slam_) {
      slam_->close(true);
    }
    if (owns_database_) {
      std::remove(database_path_.c_str());
      std::remove(query_terminal_ledger_path().c_str());
    }
  }

  void reset() {
    if (query_integrity_failed_ || query_buffer_.active()) {
      throw std::runtime_error(
          "RESET cannot abort an active or partially committed query transaction");
    }
    if (slam_) {
      slam_->close(false);
    }
    parameters_ = slam_parameters(
        profile_, feature_backend_, python_detector_path_, python_matcher_path_);
    rtabmap::ParametersMap odometry_parameters = parameters_;
    odometry_parameters[rtabmap::Parameters::kRegStrategy()] = "1";
    odometry_parameters[rtabmap::Parameters::kVisMinInliers()] =
        std::to_string(kVisualOdomMinInliers);
    odometry_parameters[rtabmap::Parameters::kIcpMaxTranslation()] = "0.05";
    odometry_parameters[rtabmap::Parameters::kIcpMaxRotation()] = "0.20";
    odometry_.reset(rtabmap::Odometry::create(odometry_parameters));
    rtabmap::ParametersMap visual_parameters = parameters_;
    visual_parameters[rtabmap::Parameters::kRegStrategy()] = "0";
    // The CPU profile keeps its established ORB adjacent-frame fallback. The
    // explicit CUDA profile must keep PyDetector here too; falling back to ORB
    // would silently move feature extraction back onto the host.
    if (feature_backend_ == FeatureBackend::kCpu) {
      visual_parameters[rtabmap::Parameters::kKpDetectorStrategy()] = "2";
      visual_parameters[rtabmap::Parameters::kVisFeatureType()] = "2";
      visual_parameters[rtabmap::Parameters::kKpMaxFeatures()] = "800";
    }
    visual_parameters[rtabmap::Parameters::kVisMinInliers()] =
        std::to_string(kVisualOdomMinInliers);
    visual_registration_.reset(
        rtabmap::Registration::create(visual_parameters));
    rtabmap::ParametersMap read_only_rgbd_parameters = parameters_;
    read_only_rgbd_parameters[rtabmap::Parameters::kRegStrategy()] = "2";
    read_only_rgbd_parameters[rtabmap::Parameters::kRegRepeatOnce()] =
        "false";
    read_only_rgbd_parameters[rtabmap::Parameters::kRegForce3DoF()] = "true";
    read_only_rgbd_parameters["Reg/ChildFallbackOnFailure"] = "false";
    read_only_rgbd_parameters_ = std::move(read_only_rgbd_parameters);
    // Construct the private Vis+ICP pipeline only when a probe is requested.
    // PyDetector may allocate a second CUDA model, so eager construction would
    // regress ordinary mapping even when recovery is never needed.
    read_only_rgbd_registration_.reset();
    rtabmap::ParametersMap map_registration_parameters = parameters_;
    map_registration_parameters[rtabmap::Parameters::kRegStrategy()] = "1";
    map_registration_parameters[rtabmap::Parameters::kRegRepeatOnce()] =
        "false";
    map_registration_parameters[rtabmap::Parameters::kRegForce3DoF()] =
        "true";
    // The reference scan is already expressed in the optimized map frame, so
    // radial filtering around its coordinate origin would discard valid map
    // points. Spatial and long-baseline filtering are done explicitly when
    // the reference is assembled below.
    map_registration_parameters[rtabmap::Parameters::kIcpRangeMin()] = "0";
    map_registration_parameters[rtabmap::Parameters::kIcpRangeMax()] = "0";
    map_registration_parameters[
        rtabmap::Parameters::kIcpMaxCorrespondenceDistance()] = "0.25";
    map_registration_parameters[rtabmap::Parameters::kIcpCorrespondenceRatio()] =
        "0.30";
    map_registration_parameters[rtabmap::Parameters::kIcpMaxTranslation()] =
        "0.50";
    map_registration_parameters[rtabmap::Parameters::kIcpMaxRotation()] =
        "0.261799388";
    map_registration_parameters[
        rtabmap::Parameters::kIcpPointToPlaneLowComplexityStrategy()] = "1";
    map_registration_.reset(
        rtabmap::Registration::create(map_registration_parameters));
    if (!map_registration_) {
      throw std::runtime_error("failed to create read-only map registration");
    }
    slam_ = std::make_unique<rtabmap::Rtabmap>();
    slam_->init(parameters_, database_path_, false);
    query_grid_maker_ =
        std::make_unique<rtabmap::LocalGridMaker>(parameters_);
    local_grids_.clear();
    grid_ = std::make_unique<rtabmap::OccupancyGrid>(&local_grids_, parameters_);
    height_points_.clear();
    occupancy_height_points_.clear();
    recovered_height_points_.clear();
    traversed_free_cells_.clear();
    node_motion_progress_m_.clear();
    node_odometry_history_index_.clear();
    odometry_history_.clear();
    raw_qvel_history_.clear();
    poses_.clear();
    display_poses_.clear();
    force_global_grid_rebuild_ = false;
    global_graph_refresh_count_ = 0;
    current_pose_ = rtabmap::Transform::getIdentity();
    native_current_pose_ = rtabmap::Transform::getIdentity();
    fused_odom_pose_ = rtabmap::Transform::getIdentity();
    previous_visual_data_ = rtabmap::SensorData();
    previous_rgb_.release();
    previous_depth_.release();
    previous_camera_to_base_.setNull();
    loop_count_ = 0;
    last_inliers_ = 0;
    last_features_ = 0;
    last_ref_node_id_ = 0;
    last_mapping_node_id_ = 0;
    mode_ = MappingMode::kMapping;
    soft_mapping_state_ = SoftMappingState::kBuilding;
    localized_this_frame_ = false;
    visual_localized_this_frame_ = false;
    geometric_localized_this_frame_ = false;
    read_only_match_this_frame_ = false;
    recovery_hold_this_frame_ = false;
    recovery_graph_bridge_count_ = 0;
    verified_graph_bridge_count_ = 0;
    verified_bridge_anchor_id_this_frame_ = 0;
    normal_global_probe_complete_ = false;
    normal_global_search_pending_ = false;
    recovery_interruption_active_ = false;
    recovery_unobserved_translation_m_ = 0.0F;
    recovery_unobserved_yaw_rad_ = 0.0F;
    read_only_revisit_active_ = false;
    read_only_revisit_identity_verified_ = false;
    read_only_revisit_candidate_id_ = 0;
    read_only_revisit_start_travel_m_ = 0.0F;
    read_only_revisit_unknown_streak_ = 0;
    read_only_revisit_no_observation_streak_ = 0;
    native_novelty_resume_guard_updates_ = 0;
    native_novelty_resume_reconciliation_pending_ = false;
    native_novelty_resume_reconciliation_viewpoints_remaining_ = 0;
    native_novelty_resume_last_progress_pose_.setNull();
    native_novelty_resume_anchor_map_pose_.setNull();
    native_novelty_resume_anchor_history_index_ = 0;
    native_novelty_resume_obstacle_evidence_.clear();
    native_novelty_resume_frame_alignment_safe_ = false;
    native_novelty_resume_frame_alignment_supports_ = 0;
    native_novelty_resume_frame_alignment_max_bound_m_ = 0.0;
    native_novelty_resume_snapshot_map_.release();
    native_novelty_resume_snapshot_x_min_ = 0.0F;
    native_novelty_resume_snapshot_y_min_ = 0.0F;
    native_novelty_resume_protected_free_map_.release();
    native_novelty_resume_protected_free_x_min_ = 0.0F;
    native_novelty_resume_protected_free_y_min_ = 0.0F;
    soft_novelty_start_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    soft_novelty_anchor_pose_.setNull();
    soft_novelty_last_view_pose_.setNull();
    soft_novelty_max_translation_m_ = 0.0F;
    soft_novelty_distinct_viewpoints_ = 0;
    read_only_revisit_count_ = 0;
    read_only_candidate_active_ = false;
    read_only_candidate_start_motion_m_ = 0.0F;
    read_only_candidate_unknown_streak_ = 0;
    read_only_candidate_no_observation_streak_ = 0;
    native_revisit_hold_pending_ = false;
    native_revisit_hold_candidate_id_ = 0;
    read_only_scan_match_failures_ = 0;
    read_only_scan_match_next_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    read_only_scan_match_proposal_.clear();
    mapping_frames_ = 0;
    mapping_travel_m_ = 0.0F;
    mapping_rotation_rad_ = 0.0F;
    session_motion_progress_m_ = 0.0F;
    usable_observation_streak_ = 0;
    freeze_candidate_start_frame_ = 0;
    freeze_candidate_motion_progress_m_ = 0.0F;
    freeze_candidate_known_cells_ = 0;
    freeze_candidate_revisit_count_ = 0;
    freeze_candidate_max_novelty_ratio_ = 1.0F;
    localization_frames_ = 0;
    localization_observations_since_metric_ = 0;
    localization_last_metric_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    last_visual_revisit_frame_ = 0;
    coverage_history_.clear();
    recent_visual_revisit_frames_.clear();
    seen_loop_constraints_.clear();
    accepted_loop_pairs_.clear();
    visual_loop_regions_.clear();
    convergence_ = ConvergenceEvidence{};
    last_observation_novelty_ratio_ = 1.0F;
    last_observation_endpoint_novelty_ratio_ = 1.0F;
    last_observation_ray_novelty_ratio_ = 1.0F;
    idle_static_streak_ = 0;
    idle_hold_active_ = false;
    idle_accumulated_translation_m_ = 0.0F;
    idle_accumulated_yaw_rad_ = 0.0F;
    last_slam_stamp_s_ = -std::numeric_limits<double>::infinity();
    pending_slam_covariance_.release();
    query_buffer_.clear();
    completed_recovery_queries_.clear();
    completed_normal_queries_.clear();
    recovery_query_high_water_.clear();
    normal_query_high_water_.clear();
    durable_query_terminal_count_ = 0U;
    query_buffered_keyframes_ = 0;
    query_buffer_overflows_ = 0;
    query_invalid_frame_discards_ = 0;
    query_integrity_failures_ = 0;
    query_promotions_ = 0;
    query_all_known_discards_ = 0;
    query_promotion_rejections_ = 0;
    query_process_failures_ = 0;
    query_outcome_this_frame_ = br::QueryOutcome::kNone;
    query_scope_this_frame_ = QueryScope::kNone;
    query_generation_this_frame_ = 0;
    query_footprint_ignored_ids_.clear();
    cached_map_.release();
    cached_x_min_ = 0.0F;
    cached_y_min_ = 0.0F;
    cached_low_.clear();
    cached_high_.clear();
    frozen_map_.release();
    frozen_x_min_ = 0.0F;
    frozen_y_min_ = 0.0F;
    frozen_low_.clear();
    frozen_high_.clear();
    restore_persisted_state();
  }

  void accumulate_slam_covariance(const cv::Mat &covariance) {
    // RTAB-Map 每次 process() 需要的是自上一个图节点以来的里程计不确定度。
    // 当前主链每帧做里程计、每秒才建一个图节点，因此沿用 RTAB-Map
    // Reprocess 的帧率无关做法：窗口内各自由度取最大方差，而不是只拿窗口
    // 最后一帧的协方差。只增大对角线会保持原协方差的半正定性。
    if (pending_slam_covariance_.empty()) {
      pending_slam_covariance_ = covariance.clone();
      return;
    }
    for (int axis = 0; axis < 6; ++axis) {
      const double candidate = covariance.at<double>(axis, axis);
      if (std::isfinite(candidate) && candidate > 0.0) {
        pending_slam_covariance_.at<double>(axis, axis) = std::max(
            pending_slam_covariance_.at<double>(axis, axis), candidate);
      }
    }
  }

  bool external_loop_odometry_prior(
      int candidate_id, rtabmap::Transform *expected_candidate_to_current,
      cv::Mat *covariance) const {
    const auto node = node_odometry_history_index_.find(candidate_id);
    if (node == node_odometry_history_index_.end() ||
        node->second > odometry_history_.size()) {
      return false;
    }
    rtabmap::Transform candidate_to_current =
        rtabmap::Transform::getIdentity();
    cv::Matx33d forward_covariance = cv::Matx33d::zeros();
    for (std::size_t index = node->second;
         index < odometry_history_.size(); ++index) {
      propagate_planar_odometry(odometry_history_[index],
                                &candidate_to_current,
                                &forward_covariance);
    }
    *expected_candidate_to_current = candidate_to_current.to3DoF();
    *covariance = cv::Mat(forward_covariance).clone();
    return !expected_candidate_to_current->isNull() &&
           cv::checkRange(*covariance, true, nullptr);
  }

  static bool external_loop_metric_measurement(
      const br::FrameMeta &meta,
      rtabmap::Transform *measured_candidate_to_current,
      cv::Mat *covariance) {
    if (meta.external_loop_candidate_id <= 0) {
      return false;
    }
    *measured_candidate_to_current = rtabmap::Transform(
        static_cast<float>(meta.external_loop_candidate_to_current[0]),
        static_cast<float>(meta.external_loop_candidate_to_current[1]),
        static_cast<float>(meta.external_loop_candidate_to_current[2])).to3DoF();
    *covariance = cv::Mat(3, 3, CV_64FC1,
                         const_cast<double *>(meta.external_loop_covariance)).clone();
    if (measured_candidate_to_current->isNull() ||
        !cv::checkRange(*covariance, true, nullptr)) {
      return false;
    }
    const cv::Mat asymmetry = *covariance - covariance->t();
    if (cv::norm(asymmetry, cv::NORM_INF) > 1e-9 ||
        covariance->at<double>(0, 0) <= 0.0 ||
        covariance->at<double>(1, 1) <= 0.0 ||
        covariance->at<double>(2, 2) <= 0.0) {
      return false;
    }
    cv::Mat eigenvalues;
    if (!cv::eigen(*covariance, eigenvalues) || eigenvalues.rows != 3) {
      return false;
    }
    return eigenvalues.at<double>(2, 0) >= -1e-12;
  }

  static bool finite_planar_transform(const rtabmap::Transform &pose) {
    return !pose.isNull() && std::isfinite(pose.x()) &&
           std::isfinite(pose.y()) && std::isfinite(pose.theta());
  }

  static double wrap_angle(double angle) {
    constexpr double kPi = 3.14159265358979323846;
    while (angle >= kPi) {
      angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
      angle += 2.0 * kPi;
    }
    return angle;
  }

  static bool query_obstacle_views_are_independent(
      float reference_x, float reference_y, float reference_yaw,
      const rtabmap::Transform &candidate) {
    const double translation = std::hypot(
        static_cast<double>(candidate.x() - reference_x),
        static_cast<double>(candidate.y() - reference_y));
    const double yaw = std::abs(wrap_angle(
        static_cast<double>(candidate.theta() - reference_yaw)));
    return translation >= static_cast<double>(kQueryKeyframeTranslationM) ||
           yaw >= static_cast<double>(kQueryKeyframeYawRad);
  }

  static bool novelty_resume_obstacle_views_are_independent(
      float reference_x, float reference_y, float reference_yaw,
      const rtabmap::Transform &candidate) {
    const double translation = std::hypot(
        static_cast<double>(candidate.x() - reference_x),
        static_cast<double>(candidate.y() - reference_y));
    const double yaw = std::abs(wrap_angle(
        static_cast<double>(candidate.theta() - reference_yaw)));
    return translation >= static_cast<double>(
                              kNativeNoveltyResumeObstacleIndependentTranslationM) ||
           yaw >= static_cast<double>(
                      kNativeNoveltyResumeObstacleIndependentYawRad);
  }

  static cv::Vec3d planar_vector(const rtabmap::Transform &pose) {
    return cv::Vec3d(pose.x(), pose.y(), pose.theta());
  }

  static rtabmap::Transform planar_transform(const cv::Vec3d &value) {
    return rtabmap::Transform(
        static_cast<float>(value[0]), static_cast<float>(value[1]),
        static_cast<float>(wrap_angle(value[2]))).to3DoF();
  }

  static cv::Vec3d marginalized_bridge_vector(
      const cv::Vec3d &anchor_to_current,
      const cv::Vec3d &candidate_to_current) {
    const rtabmap::Transform bridge =
        (planar_transform(anchor_to_current) *
         planar_transform(candidate_to_current).inverse()).to3DoF();
    return planar_vector(bridge);
  }

  static cv::Mat numerical_planar_jacobian(
      const cv::Vec3d &anchor_to_current,
      const cv::Vec3d &candidate_to_current,
      bool perturb_anchor) {
    cv::Mat jacobian = cv::Mat::zeros(3, 3, CV_64FC1);
    constexpr double kEpsilon = 1e-5;
    for (int axis = 0; axis < 3; ++axis) {
      cv::Vec3d anchor_plus = anchor_to_current;
      cv::Vec3d anchor_minus = anchor_to_current;
      cv::Vec3d candidate_plus = candidate_to_current;
      cv::Vec3d candidate_minus = candidate_to_current;
      if (perturb_anchor) {
        anchor_plus[axis] += kEpsilon;
        anchor_minus[axis] -= kEpsilon;
      } else {
        candidate_plus[axis] += kEpsilon;
        candidate_minus[axis] -= kEpsilon;
      }
      const cv::Vec3d plus = marginalized_bridge_vector(
          anchor_plus, candidate_plus);
      const cv::Vec3d minus = marginalized_bridge_vector(
          anchor_minus, candidate_minus);
      jacobian.at<double>(0, axis) =
          (plus[0] - minus[0]) / (2.0 * kEpsilon);
      jacobian.at<double>(1, axis) =
          (plus[1] - minus[1]) / (2.0 * kEpsilon);
      jacobian.at<double>(2, axis) =
          wrap_angle(plus[2] - minus[2]) / (2.0 * kEpsilon);
    }
    return jacobian;
  }

  static cv::Mat sanitized_planar_covariance(const cv::Mat &input) {
    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64FC1);
    if (input.rows == 3 && input.cols == 3 && input.type() == CV_64FC1 &&
        cv::checkRange(input, true, nullptr)) {
      covariance = input.clone();
    }
    covariance = (covariance + covariance.t()) * 0.5;
    covariance.at<double>(0, 0) = std::max(
        covariance.at<double>(0, 0), kObservedTranslationVarianceFloor);
    covariance.at<double>(1, 1) = std::max(
        covariance.at<double>(1, 1), kObservedTranslationVarianceFloor);
    covariance.at<double>(2, 2) = std::max(
        covariance.at<double>(2, 2), kObservedYawVarianceFloor);
    covariance += cv::Mat::eye(3, 3, CV_64FC1) * 1e-12;
    return covariance;
  }

  static cv::Mat marginalized_bridge_covariance(
      const rtabmap::Transform &anchor_to_current,
      const cv::Mat &anchor_covariance,
      const rtabmap::Transform &candidate_to_current,
      const cv::Mat &candidate_covariance) {
    const cv::Vec3d anchor = planar_vector(anchor_to_current);
    const cv::Vec3d candidate = planar_vector(candidate_to_current);
    const cv::Mat anchor_jacobian =
        numerical_planar_jacobian(anchor, candidate, true);
    const cv::Mat candidate_jacobian =
        numerical_planar_jacobian(anchor, candidate, false);
    cv::Mat covariance =
        anchor_jacobian * sanitized_planar_covariance(anchor_covariance) *
            anchor_jacobian.t() +
        candidate_jacobian * sanitized_planar_covariance(candidate_covariance) *
            candidate_jacobian.t();
    return sanitized_planar_covariance(covariance);
  }

  static cv::Mat planar_information(const cv::Mat &planar_covariance) {
    cv::Mat inverse;
    const cv::Mat covariance =
        sanitized_planar_covariance(planar_covariance);
    if (!cv::invert(covariance, inverse, cv::DECOMP_SVD) ||
        !cv::checkRange(inverse, true, nullptr)) {
      return cv::Mat();
    }
    cv::Mat information = cv::Mat::eye(6, 6, CV_64FC1) * 1e-6;
    constexpr int kPlanarAxes[3] = {0, 1, 5};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        information.at<double>(kPlanarAxes[row], kPlanarAxes[column]) =
            inverse.at<double>(row, column);
      }
    }
    return information;
  }

  // Compare the independent RGB-D metric transform with the complete body
  // velocity chain.  A diagonal floor prevents a very confident but sparse
  // feature match from becoming an unbounded teleport; SVD keeps the test
  // well-defined when a corridor makes one planar axis degenerate.
  static bool metric_innovation_squared(
      const rtabmap::Transform &expected,
      const cv::Mat &expected_covariance,
      const rtabmap::Transform &measured,
      const cv::Mat &measured_covariance,
      double *value_out) {
    if (!finite_planar_transform(expected) ||
        !finite_planar_transform(measured) || value_out == nullptr) {
      return false;
    }
    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64FC1);
    const auto add_covariance = [&covariance](const cv::Mat &source) {
      if (source.rows != 3 || source.cols != 3 ||
          source.type() != CV_64FC1 ||
          !cv::checkRange(source, true, nullptr)) {
        return;
      }
      covariance += source;
    };
    add_covariance(expected_covariance);
    add_covariance(measured_covariance);
    // A missing prior is still a valid candidate when the correction caps
    // below pass.  These floors are only used for the statistical gate.
    covariance.at<double>(0, 0) = std::max(covariance.at<double>(0, 0), 0.01);
    covariance.at<double>(1, 1) = std::max(covariance.at<double>(1, 1), 0.01);
    covariance.at<double>(2, 2) = std::max(covariance.at<double>(2, 2),
                                             0.01 * 0.01);
    covariance = (covariance + covariance.t()) * 0.5;
    cv::Mat inverse;
    if (!cv::invert(covariance, inverse, cv::DECOMP_SVD) ||
        !cv::checkRange(inverse, true, nullptr)) {
      return false;
    }
    const rtabmap::Transform residual =
        (expected.inverse() * measured).to3DoF();
    cv::Mat vector = (cv::Mat_<double>(3, 1) << residual.x(), residual.y(),
                      wrap_angle(residual.theta()));
    const cv::Mat quadratic = vector.t() * inverse * vector;
    const double value = quadratic.at<double>(0, 0);
    if (!std::isfinite(value) || value < 0.0) {
      return false;
    }
    *value_out = value;
    return true;
  }

  bool map_pose_for_node(int node_id, rtabmap::Transform *pose) const {
    if (pose == nullptr || node_id <= 0) {
      return false;
    }
    const auto found = poses_.find(node_id);
    if (found != poses_.end() && finite_planar_transform(found->second)) {
      *pose = found->second.to3DoF();
      return true;
    }
    if (slam_) {
      const rtabmap::Transform native_pose = slam_->getPose(node_id);
      if (finite_planar_transform(native_pose)) {
        *pose = native_pose.to3DoF();
        return true;
      }
    }
    return false;
  }

  std::vector<int> read_only_rgbd_candidates(int seed_candidate_id) const {
    std::vector<int> output;
    if (!slam_ || seed_candidate_id <= 0) {
      return output;
    }
    rtabmap::Transform seed_pose;
    if (!map_pose_for_node(seed_candidate_id, &seed_pose)) {
      return output;
    }
    const rtabmap::Signature seed = slam_->getSignatureCopy(
        seed_candidate_id, false, false, false, false, false, false);
    if (seed.id() <= 0) {
      return output;
    }
    output.push_back(seed_candidate_id);
    std::vector<int> neighbors;
    for (const auto &entry : seed.getLinks()) {
      const rtabmap::Link &link = entry.second;
      if ((link.type() != rtabmap::Link::kNeighbor &&
           link.type() != rtabmap::Link::kNeighborMerged) ||
          entry.first <= 0 || entry.first == seed_candidate_id) {
        continue;
      }
      rtabmap::Transform neighbor_pose;
      if (map_pose_for_node(entry.first, &neighbor_pose)) {
        neighbors.push_back(entry.first);
      }
    }
    std::sort(neighbors.begin(), neighbors.end(),
              [seed_candidate_id](int left, int right) {
                const int left_gap = std::abs(left - seed_candidate_id);
                const int right_gap = std::abs(right - seed_candidate_id);
                return left_gap != right_gap ? left_gap < right_gap
                                             : left < right;
              });
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
    for (int neighbor_id : neighbors) {
      if (output.size() >= kReadOnlyRgbdMaxCandidates) {
        break;
      }
      output.push_back(neighbor_id);
    }
    return output;
  }

  bool resolve_read_only_rgbd_registration(
      int seed_candidate_id, const rtabmap::SensorData &current_data,
      const rtabmap::Transform &seed_to_current_guess,
      ReadOnlyRgbdRegistrationResult *result) {
    if (result == nullptr) {
      return false;
    }
    *result = ReadOnlyRgbdRegistrationResult();
    if (!slam_ || seed_candidate_id <= 0 || !current_data.isValid() ||
        current_data.imageRaw().empty() || current_data.depthRaw().empty() ||
        current_data.laserScanRaw().isEmpty() ||
        (!seed_to_current_guess.isNull() &&
         !finite_planar_transform(seed_to_current_guess))) {
      return false;
    }
    if (!read_only_rgbd_registration_) {
      // Registration implementations retain feature-matcher scratch state.
      // The probe therefore owns a separate object instead of perturbing the
      // adjacent-frame odometry pipeline, but pays its setup cost only once.
      read_only_rgbd_registration_.reset(
          rtabmap::Registration::create(read_only_rgbd_parameters_));
      if (!read_only_rgbd_registration_) {
        return false;
      }
    }

    rtabmap::Transform seed_map_pose;
    if (!map_pose_for_node(seed_candidate_id, &seed_map_pose)) {
      return false;
    }
    bool found = false;
    for (int candidate_id : read_only_rgbd_candidates(seed_candidate_id)) {
      rtabmap::Transform candidate_map_pose;
      if (!map_pose_for_node(candidate_id, &candidate_map_pose)) {
        continue;
      }
      // getSignatureCopy() performs at most a database read and returns a value
      // object. Decompression and Registration's feature/ICP scratch data are
      // confined to this local copy; Memory, STM/WM and LocalGridCache are not
      // touched by the probe.
      rtabmap::Signature candidate = slam_->getSignatureCopy(
          candidate_id, true, true, false, false, true, false);
      if (candidate.id() <= 0) {
        continue;
      }
      cv::Mat image_buffer;
      cv::Mat depth_buffer;
      rtabmap::LaserScan scan_buffer;
      candidate.sensorData().uncompressData(
          &image_buffer, &depth_buffer, &scan_buffer);
      if (candidate.sensorData().imageRaw().empty() ||
          candidate.sensorData().depthRaw().empty() ||
          candidate.sensorData().laserScanRaw().isEmpty()) {
        continue;
      }

      rtabmap::Transform candidate_guess;
      if (!seed_to_current_guess.isNull()) {
        candidate_guess =
            (candidate_map_pose.inverse() * seed_map_pose *
             seed_to_current_guess).to3DoF();
      }
      rtabmap::Signature current(current_data);
      rtabmap::RegistrationInfo info;
      const rtabmap::Transform measured =
          read_only_rgbd_registration_->computeTransformation(
              candidate, current, candidate_guess, &info);
      if (!finite_planar_transform(measured) ||
          measured.getNorm() > kReadOnlyRevisitMaxCandidateDistanceM ||
          info.inliers < kVisualLoopMinInliers ||
          !std::isfinite(info.inliersDistribution) ||
          info.inliersDistribution < kReadOnlyRgbdMinInlierDistribution ||
          info.icpCorrespondences <= 0 ||
          !std::isfinite(info.icpInliersRatio) ||
          info.icpInliersRatio < 0.12F || info.covariance.rows != 6 ||
          info.covariance.cols != 6 ||
          info.covariance.type() != CV_64FC1 ||
          !cv::checkRange(info.covariance, true, nullptr)) {
        continue;
      }

      const bool better =
          !found || info.inliers > result->info.inliers ||
          (info.inliers == result->info.inliers &&
           info.icpInliersRatio > result->info.icpInliersRatio) ||
          (info.inliers == result->info.inliers &&
           info.icpInliersRatio == result->info.icpInliersRatio &&
           info.inliersDistribution > result->info.inliersDistribution);
      if (!better) {
        continue;
      }
      cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64FC1);
      constexpr int kPlanarAxes[3] = {0, 1, 5};
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          covariance.at<double>(row, column) =
              info.covariance.at<double>(kPlanarAxes[row],
                                         kPlanarAxes[column]);
        }
      }
      covariance = (covariance + covariance.t()) * 0.5;
      covariance.at<double>(0, 0) = std::max(
          covariance.at<double>(0, 0), kObservedTranslationVarianceFloor);
      covariance.at<double>(1, 1) = std::max(
          covariance.at<double>(1, 1), kObservedTranslationVarianceFloor);
      covariance.at<double>(2, 2) = std::max(
          covariance.at<double>(2, 2), kObservedYawVarianceFloor);
      if (!cv::checkRange(covariance, true, nullptr)) {
        continue;
      }
      result->candidate_id = candidate_id;
      result->candidate_to_current = measured.to3DoF();
      result->planar_covariance = covariance;
      result->info = info.copyWithoutData();
      found = true;
    }
    return found;
  }

  bool resolve_read_only_revisit(
      const FrameInput &frame,
      const rtabmap::Transform &map_correction,
      const rtabmap::Transform &predicted_map_pose,
      const rtabmap::Transform &measured_candidate_to_current,
      const cv::Mat &measurement_covariance,
      bool recovery_relocalization,
      rtabmap::Transform *target_map_pose,
      double *innovation_squared,
      double *correction_translation,
      double *correction_yaw) const {
    if (frame.meta.external_loop_candidate_id <= 0 ||
        poses_.size() < kReadOnlyRevisitMinMapNodes ||
        !finite_planar_transform(predicted_map_pose) ||
        !finite_planar_transform(measured_candidate_to_current)) {
      return false;
    }
    rtabmap::Transform candidate_map_pose;
    if (!map_pose_for_node(frame.meta.external_loop_candidate_id,
                           &candidate_map_pose)) {
      return false;
    }
    // The candidate itself must be in the sensor's metric basin. This rejects
    // a perceptual alias before it can suppress a genuinely new room.
    if (measured_candidate_to_current.getNorm() >
        kReadOnlyRevisitMaxCandidateDistanceM) {
      return false;
    }
    rtabmap::Transform expected_candidate_to_current;
    cv::Mat expected_covariance;
    double innovation = 0.0;
    if (external_loop_odometry_prior(
            frame.meta.external_loop_candidate_id,
            &expected_candidate_to_current, &expected_covariance)) {
      if (recovery_relocalization) {
        const double translation_sigma = std::max(
            0.25, static_cast<double>(recovery_unobserved_translation_m_));
        const double yaw_sigma = std::max(
            0.261799388,
            static_cast<double>(recovery_unobserved_yaw_rad_));
        expected_covariance.at<double>(0, 0) +=
            translation_sigma * translation_sigma;
        expected_covariance.at<double>(1, 1) +=
            translation_sigma * translation_sigma;
        expected_covariance.at<double>(2, 2) += yaw_sigma * yaw_sigma;
      }
      if (!metric_innovation_squared(
              expected_candidate_to_current, expected_covariance,
              measured_candidate_to_current, measurement_covariance,
              &innovation) ||
          innovation > kReadOnlyRevisitMaxInnovationChiSquare) {
        return false;
      }
    }
    const rtabmap::Transform target =
        (candidate_map_pose * measured_candidate_to_current).to3DoF();
    if (!finite_planar_transform(target)) {
      return false;
    }
    const rtabmap::Transform correction =
        (predicted_map_pose.inverse() * target).to3DoF();
    const double translation = std::hypot(correction.x(), correction.y());
    const double yaw = std::abs(wrap_angle(correction.theta()));
    const double translation_limit =
        recovery_relocalization
            ? std::min(
                  static_cast<double>(kRecoveryRelocalizationMaxCorrectionM),
                  std::max(
                      static_cast<double>(kReadOnlyRevisitMaxCorrectionM),
                      static_cast<double>(recovery_unobserved_translation_m_) +
                          static_cast<double>(kReadOnlyRevisitMaxCorrectionM)))
            : static_cast<double>(kReadOnlyRevisitMaxCorrectionM);
    const double yaw_limit =
        recovery_relocalization
            ? std::min(
                  static_cast<double>(kRecoveryRelocalizationMaxCorrectionYawRad),
                  std::max(
                      static_cast<double>(kReadOnlyRevisitMaxCorrectionYawRad),
                      static_cast<double>(recovery_unobserved_yaw_rad_) +
                          static_cast<double>(kReadOnlyRevisitMaxCorrectionYawRad)))
            : static_cast<double>(kReadOnlyRevisitMaxCorrectionYawRad);
    if (!std::isfinite(translation) || !std::isfinite(yaw) ||
        translation > translation_limit || yaw > yaw_limit) {
      return false;
    }
    // ``map_correction`` is intentionally part of the call contract: the
    // target is a map pose, while the worker's fused pose is map-correction
    // inverse odometry. Referencing it here catches accidental frame swaps at
    // compile-time call sites and documents the transform convention.
    (void)map_correction;
    *target_map_pose = target;
    if (innovation_squared != nullptr) {
      *innovation_squared = innovation;
    }
    if (correction_translation != nullptr) {
      *correction_translation = translation;
    }
    if (correction_yaw != nullptr) {
      *correction_yaw = yaw;
    }
    return true;
  }

  static float height_band_center(std::uint8_t band) {
    const float clamped_band = static_cast<float>(std::min<std::uint8_t>(
        band, static_cast<std::uint8_t>(kWallBandCount - 1U)));
    return kObstacleMinM +
           (clamped_band + 0.5F) *
               (kObstacleMaxM - kObstacleMinM) /
               static_cast<float>(kWallBandCount);
  }

  bool resolve_read_only_scan_match(
      const FrameInput &frame,
      const rtabmap::Transform &predicted_map_pose,
      float frame_motion_progress,
      rtabmap::Transform *target_map_pose,
      cv::Mat *planar_covariance,
      rtabmap::RegistrationInfo *registration_info,
      float *accepted_novelty,
      std::size_t *reference_node_count) const {
    if (!map_registration_ || target_map_pose == nullptr ||
        planar_covariance == nullptr || registration_info == nullptr ||
        !finite_planar_transform(predicted_map_pose) ||
        frame.height_points.size() <
            static_cast<std::size_t>(kReadOnlyScanMatchMinCorrespondences)) {
      return false;
    }

    std::vector<cv::Vec3f> current_points;
    current_points.reserve(frame.height_points.size());
    for (const HeightPoint &point : frame.height_points) {
      current_points.emplace_back(
          point.x, point.y, height_band_center(point.band));
    }

    std::vector<cv::Vec3f> reference_points;
    reference_points.reserve(8192);
    std::unordered_set<std::uint64_t> reference_cells;
    reference_cells.reserve(16384);
    std::size_t contributing_nodes = 0;
    for (const auto &entry : height_points_) {
      const auto progress = node_motion_progress_m_.find(entry.first);
      const auto pose_iter = display_poses_.find(entry.first);
      if (progress == node_motion_progress_m_.end() ||
          pose_iter == display_poses_.end() ||
          !finite_planar_transform(pose_iter->second) ||
          frame_motion_progress - progress->second <
              kReadOnlyScanMatchReferenceMotionGapM) {
        continue;
      }
      const rtabmap::Transform &pose = pose_iter->second;
      if (std::hypot(pose.x() - predicted_map_pose.x(),
                     pose.y() - predicted_map_pose.y()) >
          kReadOnlyScanMatchReferenceNodeRadiusM) {
        continue;
      }
      bool node_contributed = false;
      for (const HeightPoint &point : entry.second) {
        const float world_x =
            pose.r11() * point.x + pose.r12() * point.y + pose.x();
        const float world_y =
            pose.r21() * point.x + pose.r22() * point.y + pose.y();
        if (std::hypot(world_x - predicted_map_pose.x(),
                       world_y - predicted_map_pose.y()) >
            kReadOnlyScanMatchReferenceRadiusM) {
          continue;
        }
        const int cell_x =
            static_cast<int>(std::floor(world_x / kGridCellM));
        const int cell_y =
            static_cast<int>(std::floor(world_y / kGridCellM));
        if (!reference_cells.insert(
                height_cell_key(cell_x, cell_y, point.band)).second) {
          continue;
        }
        reference_points.emplace_back(
            (static_cast<float>(cell_x) + 0.5F) * kGridCellM,
            (static_cast<float>(cell_y) + 0.5F) * kGridCellM,
            height_band_center(point.band));
        node_contributed = true;
      }
      if (node_contributed) {
        ++contributing_nodes;
      }
    }
    if (reference_node_count != nullptr) {
      *reference_node_count = contributing_nodes;
    }
    if (contributing_nodes < kReadOnlyScanMatchMinReferenceNodes ||
        reference_points.size() <
            2U * static_cast<std::size_t>(
                     kReadOnlyScanMatchMinCorrespondences)) {
      return false;
    }

    cv::Mat reference_matrix(
        1, static_cast<int>(reference_points.size()), CV_32FC3,
        reference_points.data());
    cv::Mat current_matrix(
        1, static_cast<int>(current_points.size()), CV_32FC3,
        current_points.data());
    rtabmap::SensorData reference_data;
    reference_data.setLaserScan(rtabmap::LaserScan(
        reference_matrix.clone(), static_cast<int>(reference_points.size()),
        0.0F, rtabmap::LaserScan::kXYZ,
        rtabmap::Transform::getIdentity()));
    rtabmap::SensorData current_data;
    current_data.setLaserScan(rtabmap::LaserScan(
        current_matrix.clone(), static_cast<int>(current_points.size()),
        kMaxRangeM, rtabmap::LaserScan::kXYZ,
        rtabmap::Transform::getIdentity()));

    rtabmap::RegistrationInfo info;
    const rtabmap::Transform raw_target =
        map_registration_->computeTransformation(
            reference_data, current_data, predicted_map_pose, &info);
    *registration_info = info.copyWithoutData();
    if (!finite_planar_transform(raw_target) ||
        info.icpCorrespondences < kReadOnlyScanMatchMinCorrespondences ||
        !std::isfinite(info.icpInliersRatio) ||
        info.icpInliersRatio < kReadOnlyScanMatchMinInlierRatio ||
        !std::isfinite(info.icpStructuralComplexity) ||
        info.icpStructuralComplexity <= 0.0F ||
        info.covariance.rows != 6 || info.covariance.cols != 6 ||
        info.covariance.type() != CV_64FC1 ||
        !cv::checkRange(info.covariance, true, nullptr)) {
      return false;
    }
    const rtabmap::Transform target = raw_target.to3DoF();
    const rtabmap::Transform correction =
        (predicted_map_pose.inverse() * target).to3DoF();
    const float correction_translation =
        std::hypot(correction.x(), correction.y());
    const float correction_yaw =
        static_cast<float>(std::abs(wrap_angle(correction.theta())));
    if (!std::isfinite(correction_translation) ||
        !std::isfinite(correction_yaw) ||
        correction_translation > kReadOnlyScanMatchMaxCorrectionM ||
        correction_yaw > read_only_scan_match_max_correction_yaw_rad()) {
      return false;
    }

    float endpoint_novelty = 1.0F;
    float ray_novelty = 1.0F;
    const float target_novelty = observation_novelty_ratio(
        frame, target, &endpoint_novelty, &ray_novelty);
    if (!std::isfinite(target_novelty) ||
        target_novelty > kReadOnlyScanMatchMaxAcceptedNoveltyRatio) {
      return false;
    }

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64FC1);
    constexpr int kPlanarAxes[3] = {0, 1, 5};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        covariance.at<double>(row, column) =
            info.covariance.at<double>(kPlanarAxes[row],
                                       kPlanarAxes[column]);
      }
    }
    covariance = (covariance + covariance.t()) * 0.5;
    covariance.at<double>(0, 0) = std::max(
        covariance.at<double>(0, 0), kObservedTranslationVarianceFloor);
    covariance.at<double>(1, 1) = std::max(
        covariance.at<double>(1, 1), kObservedTranslationVarianceFloor);
    covariance.at<double>(2, 2) = std::max(
        covariance.at<double>(2, 2), kObservedYawVarianceFloor);
    if (!cv::checkRange(covariance, true, nullptr)) {
      return false;
    }
    *target_map_pose = target;
    *planar_covariance = covariance;
    if (accepted_novelty != nullptr) {
      *accepted_novelty = target_novelty;
    }
    return true;
  }

  void clear_read_only_scan_match_proposal() {
    read_only_scan_match_proposal_.clear();
  }

  bool confirm_read_only_scan_match_proposal(
      std::uint64_t frame_id,
      const rtabmap::Transform &target_map_pose,
      const cv::Mat &planar_covariance,
      rtabmap::Transform *confirmed_target_map_pose,
      cv::Mat *confirmed_planar_covariance) {
    if (!finite_planar_transform(target_map_pose) ||
        !finite_planar_transform(fused_odom_pose_) ||
        confirmed_target_map_pose == nullptr ||
        confirmed_planar_covariance == nullptr) {
      return false;
    }

    auto remember_current = [&]() {
      read_only_scan_match_proposal_.target_map_pose =
          target_map_pose.to3DoF();
      read_only_scan_match_proposal_.fused_odom_pose =
          fused_odom_pose_.to3DoF();
      read_only_scan_match_proposal_.planar_covariance =
          planar_covariance.clone();
      read_only_scan_match_proposal_.frame_id = frame_id;
    };
    if (!read_only_scan_match_proposal_.active()) {
      remember_current();
      return false;
    }

    const rtabmap::Transform fused_baseline =
        (read_only_scan_match_proposal_.fused_odom_pose.inverse() *
         fused_odom_pose_).to3DoF();
    const float baseline_translation =
        std::hypot(fused_baseline.x(), fused_baseline.y());
    const float baseline_yaw =
        static_cast<float>(std::abs(wrap_angle(fused_baseline.theta())));
    if (baseline_translation <
            kReadOnlyScanMatchIndependentTranslationM &&
        baseline_yaw < kReadOnlyScanMatchIndependentYawRad) {
      return false;
    }

    const rtabmap::Transform transported_target =
        (read_only_scan_match_proposal_.target_map_pose *
         fused_baseline).to3DoF();
    const rtabmap::Transform disagreement =
        (transported_target.inverse() * target_map_pose).to3DoF();
    const float disagreement_translation =
        std::hypot(disagreement.x(), disagreement.y());
    const float disagreement_yaw =
        static_cast<float>(std::abs(wrap_angle(disagreement.theta())));
    const bool consistent =
        finite_planar_transform(transported_target) &&
        finite_planar_transform(disagreement) &&
        std::isfinite(disagreement_translation) &&
        std::isfinite(disagreement_yaw) &&
        disagreement_translation <= kFreezeGraphCorrectionM &&
        disagreement_yaw <= kReadOnlyScanMatchMaxDisagreementYawRad;
    if (trace_enabled()) {
      std::cerr << "read_only_scan_match_consensus frame=" << frame_id
                << " previous_frame="
                << read_only_scan_match_proposal_.frame_id
                << " baseline_m=" << baseline_translation
                << " baseline_yaw=" << baseline_yaw
                << " disagreement_m=" << disagreement_translation
                << " disagreement_yaw=" << disagreement_yaw
                << " accepted=" << (consistent ? 1 : 0) << "\n";
    }
    if (!consistent) {
      // The old proposal is no longer a trustworthy temporal reference. Keep
      // the current measurement only as the first half of a new consensus.
      remember_current();
      return false;
    }

    *confirmed_target_map_pose = target_map_pose.to3DoF();
    *confirmed_planar_covariance = planar_covariance.clone();
    clear_read_only_scan_match_proposal();
    return true;
  }

  void apply_read_only_anchor(
      const rtabmap::Transform &fused_pose_before,
      const rtabmap::Transform &map_correction,
      const rtabmap::Transform &target_map_pose,
      const cv::Mat &measurement_covariance,
      int candidate_id) {
    const rtabmap::Transform anchored_fused =
        (map_correction.inverse() * target_map_pose).to3DoF();
    if (!finite_planar_transform(anchored_fused)) {
      return;
    }
    // An absolute RGB-D anchor, graph bridge or confirmed scan consensus
    // invalidates every proposal expressed from the previous fused chain.
    clear_read_only_scan_match_proposal();
    const auto measured_variance = [&](int planar_axis,
                                       double fallback) {
      const double value =
          query_covariance_diagonal(measurement_covariance, planar_axis);
      return std::isfinite(value) ? value : fallback;
    };
    const rtabmap::Transform effective_increment =
        (fused_pose_before.inverse() * anchored_fused).to3DoF();
    fused_odom_pose_ = anchored_fused;
    if (!odometry_history_.empty()) {
      odometry_history_.back().increment = effective_increment;
      cv::Matx33d &history_covariance = odometry_history_.back().covariance;
      const double x_variance =
          std::max(measured_variance(0, 0.01), 0.000025);
      const double y_variance =
          std::max(measured_variance(1, 0.01), 0.000025);
      const double yaw_variance =
          std::max(measured_variance(2, 0.01 * 0.01), 0.0000761524227);
      history_covariance = cv::Matx33d(
          x_variance, 0.0, 0.0,
          0.0, y_variance, 0.0,
          0.0, 0.0, yaw_variance);
      if (!pending_slam_covariance_.empty() &&
          pending_slam_covariance_.rows == 6 &&
          pending_slam_covariance_.cols == 6) {
        pending_slam_covariance_.at<double>(0, 0) = std::max(
            pending_slam_covariance_.at<double>(0, 0), x_variance);
        pending_slam_covariance_.at<double>(1, 1) = std::max(
            pending_slam_covariance_.at<double>(1, 1), y_variance);
        pending_slam_covariance_.at<double>(5, 5) = std::max(
            pending_slam_covariance_.at<double>(5, 5), yaw_variance);
      }
    }
    native_current_pose_ = target_map_pose.to3DoF();
    current_pose_ = native_current_pose_;
    last_ref_node_id_ = candidate_id;
  }

  int recovery_bridge_anchor(int candidate_id) const {
    if (!slam_ || candidate_id <= 0 ||
        !finite_planar_transform(slam_->getPose(candidate_id))) {
      return 0;
    }
    const rtabmap::Signature candidate = slam_->getSignatureCopy(
        candidate_id, false, false, false, false, false, false);
    if (candidate.id() <= 0) {
      return 0;
    }
    std::vector<std::pair<std::size_t, int>> candidates;
    candidates.reserve(node_odometry_history_index_.size());
    for (const auto &entry : node_odometry_history_index_) {
      candidates.push_back({entry.second, entry.first});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &left, const auto &right) {
                return left.first != right.first
                           ? left.first > right.first
                           : left.second > right.second;
              });
    for (const auto &entry : candidates) {
      const int anchor_id = entry.second;
      if (anchor_id <= 0 || anchor_id == candidate_id ||
          candidate.getLinks().find(anchor_id) !=
              candidate.getLinks().end() ||
          !finite_planar_transform(slam_->getPose(anchor_id))) {
        continue;
      }
      rtabmap::Transform anchor_to_current;
      cv::Mat anchor_covariance;
      if (external_loop_odometry_prior(
              anchor_id, &anchor_to_current, &anchor_covariance)) {
        return anchor_id;
      }
    }
    return 0;
  }

  bool commit_verified_graph_bridge(
      std::uint64_t frame_id, int candidate_id,
      const rtabmap::Transform &candidate_to_current,
      const cv::Mat &candidate_covariance,
      const rtabmap::Transform &fused_pose_before,
      const char *evidence_source, bool recovery_semantics,
      QueryScope query_scope, std::uint64_t query_generation) {
    if (!slam_ || mode_ != MappingMode::kMapping || candidate_id <= 0 ||
        !finite_planar_transform(candidate_to_current) ||
        !finite_planar_transform(fused_pose_before)) {
      return false;
    }

    const int anchor_id = recovery_bridge_anchor(candidate_id);
    rtabmap::Transform anchor_to_current;
    cv::Mat anchor_covariance;
    if (anchor_id <= 0 ||
        !external_loop_odometry_prior(
            anchor_id, &anchor_to_current, &anchor_covariance)) {
      if (trace_enabled()) {
        std::cerr << "verified_bridge_rejected frame=" << frame_id
                  << " candidate=" << candidate_id
                  << " reason=no_unlinked_graph_anchor"
                  << " source=" << evidence_source << "\n";
      }
      return false;
    }

    // Eliminate the transient current observation Q exactly in SE(2):
    // A->C = (A->Q) * inverse(C->Q). Covariance is propagated through that
    // composition with first-order Jacobians under the independent odometry
    // and RGB-D measurement model. Q never enters Memory, LocalGridCache or
    // the occupancy raster.
    const rtabmap::Transform anchor_to_candidate =
        (anchor_to_current * candidate_to_current.inverse()).to3DoF();
    const cv::Mat bridge_covariance = marginalized_bridge_covariance(
        anchor_to_current, anchor_covariance,
        candidate_to_current, candidate_covariance);
    const cv::Mat information = planar_information(bridge_covariance);
    if (!finite_planar_transform(anchor_to_candidate) ||
        information.empty()) {
      return false;
    }

    const bool query_bridge =
        query_scope != QueryScope::kNone && query_generation != 0U;
    const cv::Mat bridge_marker =
        query_bridge
            ? rtabmap::compressData2(encode_persisted_query(
                  query_scope, query_generation,
                  br::QueryOutcome::kBridgeOnlyDiscarded,
                  QueryPromotionKind::kBridgeOnly, anchor_id, candidate_id, {}))
            : cv::Mat();
    const rtabmap::Link bridge(
        anchor_id, candidate_id, rtabmap::Link::kGlobalClosure,
        anchor_to_candidate, information, bridge_marker);
    bool committed = false;
    if (query_bridge) {
      // Materialize the already committed resident graph before adding any
      // transaction marker or provisional link. The checkpoint contains no
      // query artifact and gives persistQueryLink() durable A/C endpoints.
      query_integrity_failed_ = true;
      slam_->checkpointCommittedGraph();
    }
    try {
      committed = query_bridge ? slam_->addQueryBridgeLink(bridge)
                               : slam_->addLink(bridge);
    } catch (...) {
      if (query_bridge) {
        query_integrity_failed_ = true;
      }
      throw;
    }
    if (!committed) {
      if (query_bridge) {
        // addQueryBridgeLink() uses the no-weight query-link path. A rejected
        // optimization removes the provisional bidirectional edge before any
        // optimized pose, constraint, map correction, cache or database state
        // is published. Keep every accepted QuerySubmap view and veto under
        // the frozen transaction. This positive metric measurement is
        // frame-local, so a later qualifying keyframe must bring a fresh
        // measurement before another attempt.
        query_integrity_failed_ = false;
        if (query_buffer_.active() &&
            query_buffer_.scope == query_scope &&
            query_buffer_.generation == query_generation) {
          query_buffer_.latched_release_kind = QueryReleaseKind::kNone;
          query_buffer_.promotion_attempted = true;
        }
      }
      if (trace_enabled()) {
        std::cerr << "verified_bridge_rejected frame=" << frame_id
                  << " anchor=" << anchor_id
                  << " candidate=" << candidate_id
                  << " source=" << evidence_source
                  << " reason=graph_consistency"
                  << " transform_x=" << anchor_to_candidate.x()
                  << " transform_y=" << anchor_to_candidate.y()
                  << " transform_yaw=" << anchor_to_candidate.theta()
                  << "\n";
      }
      return false;
    }
    verified_bridge_anchor_id_this_frame_ = anchor_id;
    try {

    const int left = std::min(anchor_id, candidate_id);
    const int right = std::max(anchor_id, candidate_id);
    seen_loop_constraints_.insert(
        {left, right, static_cast<int>(rtabmap::Link::kGlobalClosure)});
    if (loop_count_ < std::numeric_limits<std::uint32_t>::max()) {
      ++loop_count_;
    }
    const auto anchor_motion = node_motion_progress_m_.find(anchor_id);
    const auto candidate_motion = node_motion_progress_m_.find(candidate_id);
    if (anchor_motion != node_motion_progress_m_.end() &&
        candidate_motion != node_motion_progress_m_.end() &&
        loop_has_independent_motion(
            std::abs(anchor_motion->second - candidate_motion->second))) {
      const auto candidate_pose = poses_.find(candidate_id);
      note_visual_revisit(
          anchor_id, candidate_id,
          candidate_pose != poses_.end() ? candidate_pose->second
                                         : slam_->getPose(candidate_id));
    }

    const rtabmap::Transform optimized_anchor = slam_->getPose(anchor_id);
    if (!finite_planar_transform(optimized_anchor)) {
      throw std::runtime_error(
          "RTAB-Map accepted a recovery bridge without an optimized anchor");
    }

    // Q was deliberately kept out of Memory. Back-substitute its map pose
    // from the optimized historical endpoint instead of continuing with the
    // stale pre-closure mapCorrection*fusedOdom estimate. Rebase the current
    // fused chain and its last history step only after addLink() has accepted
    // the permanent A->C factor, so every rejected recovery is a no-op.
    // An old candidate loaded from LTM is not necessarily retained in
    // Rtabmap::getPose(), whose table is only the local optimized graph. Read
    // it from the complete graph while refreshing historical display poses.
    rtabmap::Transform optimized_candidate;
    if (!refresh_global_graph_poses(
            anchor_id, optimized_anchor, candidate_id,
            &optimized_candidate)) {
      throw std::runtime_error(
          "RTAB-Map accepted a recovery bridge without a global candidate pose");
    }
    const rtabmap::Transform optimized_current =
        (optimized_candidate * candidate_to_current).to3DoF();
    if (!finite_planar_transform(optimized_current)) {
      throw std::runtime_error(
          "RTAB-Map recovery bridge produced an invalid back-substituted pose");
    }
    rebuild_display_poses();
    grid_->clear();
    grid_->update(display_poses_);
    force_global_grid_rebuild_ = false;
    refresh_mapping_cache();

    const rtabmap::Transform correction =
        slam_->getMapCorrection().isNull()
            ? rtabmap::Transform::getIdentity()
            : slam_->getMapCorrection().to3DoF();
    apply_read_only_anchor(
        fused_pose_before, correction, optimized_current,
        candidate_covariance, candidate_id);
    localized_this_frame_ = true;
    visual_localized_this_frame_ = true;
    geometric_localized_this_frame_ = true;
    read_only_match_this_frame_ = true;
    read_only_revisit_active_ = true;
    soft_mapping_state_ = SoftMappingState::kKnownLocalizing;
    read_only_revisit_identity_verified_ = true;
    read_only_revisit_candidate_id_ = candidate_id;
    read_only_revisit_start_travel_m_ = session_motion_progress_m_;
    reset_soft_novelty_evidence();
    read_only_revisit_no_observation_streak_ = 0;
    read_only_scan_match_failures_ = 0;
    read_only_scan_match_next_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    if (recovery_semantics) {
      recovery_interruption_active_ = false;
      recovery_unobserved_translation_m_ = 0.0F;
      recovery_unobserved_yaw_rad_ = 0.0F;
    }
    normal_global_probe_complete_ = false;
    if (read_only_revisit_count_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++read_only_revisit_count_;
    }
    std::uint32_t &bridge_count = recovery_semantics
                                      ? recovery_graph_bridge_count_
                                      : verified_graph_bridge_count_;
    if (bridge_count < std::numeric_limits<std::uint32_t>::max()) {
      ++bridge_count;
    }
    if (trace_enabled()) {
      std::cerr << (recovery_semantics
                        ? "recovery_bridge_committed frame="
                        : "verified_bridge_committed frame=") << frame_id
                << " anchor=" << anchor_id
                << " candidate=" << candidate_id
                << " source=" << evidence_source
                << " transform_x=" << anchor_to_candidate.x()
                << " transform_y=" << anchor_to_candidate.y()
                << " transform_yaw=" << anchor_to_candidate.theta()
                << " pose_x=" << current_pose_.x()
                << " pose_y=" << current_pose_.y()
                << " pose_yaw=" << current_pose_.theta()
                << " bridges=" << bridge_count << "\n";
    }
    if (query_bridge) {
      // Persist the terminal bridge marker only after every posterior graph,
      // raster and pose-rebase check above has succeeded.
      slam_->persistQueryLink(bridge);
      query_integrity_failed_ = false;
    }
    return true;
    } catch (...) {
      if (query_bridge) {
        query_integrity_failed_ = true;
      }
      throw;
    }
  }

  ProcessResult process(FrameInput &frame) {
    localized_this_frame_ = false;
    visual_localized_this_frame_ = false;
    geometric_localized_this_frame_ = false;
    read_only_match_this_frame_ = false;
    recovery_hold_this_frame_ = false;
    query_outcome_this_frame_ = br::QueryOutcome::kNone;
    query_scope_this_frame_ = QueryScope::kNone;
    query_generation_this_frame_ = 0;
    verified_bridge_anchor_id_this_frame_ = 0;
    last_ref_node_id_ = 0;
    if (native_revisit_hold_pending_) {
      const int candidate_id = native_revisit_hold_candidate_id_;
      native_revisit_hold_pending_ = false;
      native_revisit_hold_candidate_id_ = 0;
      enter_native_revisit_hold(frame.meta.frame_id, candidate_id);
    }
    rtabmap::CameraModel camera(
        frame.meta.fx, frame.meta.fy, frame.meta.cx, frame.meta.cy, frame.camera_to_base,
        0.0, cv::Size(static_cast<int>(frame.meta.width), static_cast<int>(frame.meta.height)));
    const cv::Mat &registration_depth =
        frame.raw_depth.empty() ? frame.depth : frame.raw_depth;
    // RGB-D 特征和长期几何重访使用原始合规米制深度；LaserScan 只供
    // Grid/Sensor=0 生成占据栅格。独立的相邻帧 ICP 使用过滤后的几何副本，
    // 这样深度边缘不会成为 odometry 的伪对应，同时不会削弱长期外观检索。
    rtabmap::SensorData data(frame.rgb, registration_depth, camera,
                             static_cast<int>(frame.meta.frame_id),
                             frame.meta.stamp);
    data.setLaserScan(frame.depth_scan);
    const rtabmap::Transform guess(static_cast<float>(frame.meta.odom_dx),
                                   static_cast<float>(frame.meta.odom_dy),
                                   static_cast<float>(frame.meta.odom_dyaw));
    const rtabmap::Transform qvel_increment = guess.to3DoF();
    const bool observation_usable =
        structural_observation_usable(frame) &&
        (frame.meta.frame_flags & kFrameStructuralObservationUsable) != 0U;
    const bool recovery_hold_requested =
        (frame.meta.frame_flags & kFrameRecoveryHold) != 0U;
    const bool recovery_metric_requested =
        (frame.meta.frame_flags & kFrameRecoveryRelocalization) != 0U;
    const bool verified_bridge_requested =
        (frame.meta.frame_flags & kFrameVerifiedGraphBridge) != 0U;
    const bool recovery_global_no_mode_requested =
        (frame.meta.frame_flags & kFrameRecoveryGlobalNoMode) != 0U;
    const std::uint64_t query_generation = frame.meta.query_generation;
    const bool normal_global_no_mode_requested =
        (frame.meta.frame_flags & kFrameNormalGlobalNoMode) != 0U;
    normal_global_search_pending_ =
        (frame.meta.frame_flags & kFrameNormalGlobalSearchPending) != 0U;
    const QueryScope query_scope =
        recovery_hold_requested
            ? QueryScope::kRecovery
            : normal_global_search_pending_ ? QueryScope::kNormal
                                            : QueryScope::kNone;
    query_scope_this_frame_ = query_scope;
    query_generation_this_frame_ =
        query_scope == QueryScope::kNone ? 0U : query_generation;
    if (query_integrity_failed_) {
      throw std::runtime_error(
          "query promotion previously crossed a partial-commit boundary");
    }
    br::QueryOutcome replayed_query_outcome = br::QueryOutcome::kNone;
    const bool query_terminal_replay =
        query_terminal_outcome(
            query_scope, query_generation, &replayed_query_outcome);
    const bool stale_query_generation =
        query_scope != QueryScope::kNone && !query_terminal_replay &&
        query_generation_is_stale(query_scope, query_generation);
    const bool conflicting_active_query =
        query_buffer_.active() && query_scope != QueryScope::kNone &&
        (query_buffer_.scope != query_scope ||
         query_buffer_.generation != query_generation);
    if (query_scope != QueryScope::kNone) {
      query_outcome_this_frame_ = query_terminal_replay
                                      ? replayed_query_outcome
                                      : br::QueryOutcome::kHolding;
    }
    const bool orphan_query_hold =
        query_scope == QueryScope::kNone && query_buffer_.active();
    if (orphan_query_hold) {
      query_scope_this_frame_ = query_buffer_.scope;
      query_generation_this_frame_ = query_buffer_.generation;
      query_outcome_this_frame_ = br::QueryOutcome::kHolding;
    }
    normal_global_probe_complete_ =
        normal_global_search_pending_ && !recovery_hold_requested &&
        normal_global_no_mode_requested;
    // A missing hold bit is not a commit/abort certificate. Preserve any
    // active transaction so a transient client retry cannot silently reopen
    // ordinary mapping or discard the provisional evidence.
    if (normal_global_probe_complete_ &&
        soft_mapping_state_ != SoftMappingState::kBuilding) {
      if (trace_enabled()) {
        std::cerr << "normal_global_no_mode frame=" << frame.meta.frame_id
                  << " soft_state="
                  << static_cast<int>(soft_mapping_state_) << "\n";
      }
    }
    if (!observation_usable) {
      if (!recovery_interruption_active_) {
        recovery_unobserved_translation_m_ = 0.0F;
        recovery_unobserved_yaw_rad_ = 0.0F;
      }
      recovery_interruption_active_ = true;
      recovery_unobserved_translation_m_ +=
          std::hypot(qvel_increment.x(), qvel_increment.y());
      recovery_unobserved_yaw_rad_ += std::abs(qvel_increment.theta());
    }
    if (!observation_usable) {
      usable_observation_streak_ = 0U;
    } else if (usable_observation_streak_ <
               std::numeric_limits<std::uint32_t>::max()) {
      ++usable_observation_streak_;
    }
    // Keep the live mapping instance incremental throughout recovery. In
    // RTAB-Map, changing Mem/IncrementalMemory from true to false increments
    // the map id and moves STM to WM; switching the same instance back would
    // therefore make the first post-recovery mapping node a disconnected
    // component. The client performs transient retrieval and metric RGB-D
    // verification while recovery_hold keeps this instance immutable.
    const bool predicted_translation_motion =
        std::hypot(qvel_increment.x(), qvel_increment.y()) >
        kFusionMinPredictedTranslationM;
    const bool predicted_yaw_motion =
        std::abs(qvel_increment.theta()) > kFusionMinPredictedYawRad;
    const bool predicted_motion =
        predicted_translation_motion || predicted_yaw_motion;
    const bool timestamp_restarted =
        std::isfinite(last_slam_stamp_s_) &&
        frame.meta.stamp + 1e-9 < last_slam_stamp_s_;
    if (timestamp_restarted) {
      if (query_buffer_.active()) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "timestamp rewound during an unterminated query transaction");
      }
      pending_slam_covariance_.release();
    }
    const bool slam_update_due =
        !std::isfinite(last_slam_stamp_s_) || timestamp_restarted ||
        frame.meta.stamp + 1e-9 >=
            last_slam_stamp_s_ + kSlamUpdateIntervalS;
    // Odometry may attach transient scan features to SensorData. Keep the
    // backend input pristine so persistent place recognition extracts SIFT
    // descriptors from the original RGB-D frame.
    rtabmap::SensorData odometry_data(
        frame.rgb, frame.depth, camera, static_cast<int>(frame.meta.frame_id),
        frame.meta.stamp);
    odometry_data.setLaserScan(frame.odometry_scan);
    const rtabmap::Transform odom_before = odometry_->getPose();
    rtabmap::OdometryInfo odom_info;
    rtabmap::Transform odom_pose;
    bool recovered_from_qvel = !observation_usable;
    if (!observation_usable) {
      // Manipulation can move the head far below/in front of the chassis and
      // keep it there for thousands of frames.  Those views contain the arm,
      // held object and table, not a stable adjacent-frame odometry reference.
      // Preserve the compliant body-frame prediction and explicitly break the
      // RGB-D chain so the first upright frame cannot register against it.
      odom_pose = (odom_before * qvel_increment).to3DoF();
      odometry_->reset(odom_pose);
      forget_observation_reference();
    } else {
      odom_pose =
          odometry_->process(odometry_data, qvel_increment, &odom_info);
    }
    if (observation_usable && odom_pose.isNull() &&
        !frame.odometry_scan.empty()) {
      const rtabmap::Transform recovery_pose =
          (odom_before * qvel_increment).to3DoF();
      odometry_->reset(recovery_pose);
      rtabmap::OdometryInfo recovery_info;
      odom_pose = odometry_->process(
          odometry_data, rtabmap::Transform(), &recovery_info);
      if (!odom_pose.isNull()) {
        odom_info = recovery_info;
        recovered_from_qvel = true;
      }
    }
    if (odom_pose.isNull()) {
      last_inliers_ = static_cast<std::uint32_t>(
          std::max(0, odom_info.reg.inliers));
      last_features_ =
          static_cast<std::uint32_t>(std::max(0, odom_info.features));
      remember_observation(frame, data);
      return snapshot(frame.meta.frame_id, false, false, false);
    }

    rtabmap::Transform measured_increment =
        (odom_before.inverse() * odom_pose).to3DoF();
    const bool rejected_odometry_jump =
        !plausible_measured_increment(qvel_increment, measured_increment);
    if (rejected_odometry_jump) {
      const rtabmap::Transform expected_pose =
          (odom_before * qvel_increment).to3DoF();
      odometry_->reset(expected_pose);
      rtabmap::OdometryInfo anchored_info;
      const rtabmap::Transform anchored_pose = odometry_->process(
          odometry_data, rtabmap::Transform(), &anchored_info);
      if (!anchored_pose.isNull()) {
        odom_pose = anchored_pose;
        odom_info = anchored_info;
      } else {
        odom_pose = expected_pose;
        odometry_->reset(expected_pose);
      }
      measured_increment = (odom_before.inverse() * odom_pose).to3DoF();
    }

    const int pose_correspondences = odom_info.reg.icpCorrespondences;
    const bool icp_candidate_valid =
        !recovered_from_qvel && !rejected_odometry_jump;
    const bool icp_translation_supported =
        icp_candidate_valid &&
        pose_correspondences >= kIcpMinTranslationCorrespondences;
    const bool icp_yaw_supported =
        icp_candidate_valid &&
        pose_correspondences >= kIcpMinTranslationCorrespondences;
    const bool unsupported_registration =
        frame.meta.frame_id > 1U && icp_candidate_valid &&
        !icp_translation_supported && !icp_yaw_supported;
    if (unsupported_registration) {
      const rtabmap::Transform expected_pose =
          (odom_before * qvel_increment).to3DoF();
      odometry_->reset(expected_pose);
      rtabmap::OdometryInfo anchored_info;
      const rtabmap::Transform anchored_pose = odometry_->process(
          odometry_data, rtabmap::Transform(), &anchored_info);
      if (!anchored_pose.isNull()) {
        odom_pose = anchored_pose;
        odom_info = anchored_info;
      } else {
        odom_pose = expected_pose;
        odometry_->reset(expected_pose);
      }
      measured_increment = qvel_increment;
    }

    // Fuse each planar degree of freedom only when the current RGB-D geometry
    // observes it. A single wall constrains yaw but not translation along the
    // wall, so one scalar "ICP succeeded" decision is insufficient and bends
    // long corridors. qvel remains the body-frame prediction on weak axes.
    const bool icp_translation_observable =
        icp_translation_supported &&
        odom_info.reg.icpStructuralComplexity >=
            kIcpMinStructuralComplexity;
    // During recovery and a verified read-only continuation, commanded qvel
    // cannot distinguish sub-centimetre contact slip from real motion along a
    // weak point-to-plane axis. Reuse the existing adjacent RGB-D registration
    // only in those quarantined states; ordinary incremental mapping keeps its
    // current fusion policy.
    const bool weak_axis_visual_probe =
        (recovery_hold_requested || read_only_revisit_active_) &&
        !icp_translation_observable;
    const bool static_observation = is_static_observation(frame);
    // Keep a reversible idle state in the native worker.  We deliberately
    // classify the frame before any RTAB-Map call: static RGB-D samples still
    // update the lightweight reference used for wake-up, but they do not enter
    // Working Memory or grow the database.  Accumulated qvel catches slow
    // motion that is below one-frame thresholds.
    const float qvel_translation_m =
        std::hypot(qvel_increment.x(), qvel_increment.y());
    const float qvel_yaw_rad = std::abs(qvel_increment.theta());
    bool idle_motion_detected = false;
    if (static_observation && observation_usable) {
      idle_accumulated_translation_m_ += qvel_translation_m;
      idle_accumulated_yaw_rad_ += qvel_yaw_rad;
      idle_motion_detected =
          qvel_translation_m > kIdlePerFrameWakeTranslationM ||
          qvel_yaw_rad > kIdlePerFrameWakeYawRad ||
          idle_accumulated_translation_m_ >=
              kIdleAccumulatedWakeTranslationM ||
          idle_accumulated_yaw_rad_ >= kIdleAccumulatedWakeYawRad;
      if (idle_motion_detected) {
        if (idle_hold_active_ && trace_enabled()) {
          std::cerr << "idle_wake frame=" << frame.meta.frame_id
                    << " reason=qvel translation_m="
                    << idle_accumulated_translation_m_ << " yaw_rad="
                    << idle_accumulated_yaw_rad_ << "\n";
        }
        idle_hold_active_ = false;
        idle_static_streak_ = 0;
        idle_accumulated_translation_m_ = 0.0F;
        idle_accumulated_yaw_rad_ = 0.0F;
      } else {
        if (idle_static_streak_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++idle_static_streak_;
        }
        if (!idle_hold_active_ &&
            idle_static_streak_ >= kIdleEnterStaticFrames) {
          idle_hold_active_ = true;
          if (trace_enabled()) {
            std::cerr << "idle_enter frame=" << frame.meta.frame_id
                      << " static_frames=" << idle_static_streak_ << "\n";
          }
        }
      }
    } else {
      if (idle_hold_active_ && trace_enabled()) {
        std::cerr << "idle_wake frame=" << frame.meta.frame_id
                  << " reason=visual_or_camera_change\n";
      }
      idle_hold_active_ = false;
      idle_static_streak_ = 0;
      idle_accumulated_translation_m_ = 0.0F;
      idle_accumulated_yaw_rad_ = 0.0F;
    }
    rtabmap::RegistrationInfo visual_info;
    rtabmap::Transform visual_increment;
    if (slam_update_due && !static_observation &&
        ((!icp_translation_supported || !icp_yaw_supported) ||
         weak_axis_visual_probe) &&
        previous_visual_data_.isValid() &&
        visual_registration_) {
      visual_increment = visual_registration_->computeTransformation(
          previous_visual_data_, data, qvel_increment, &visual_info);
      if (!visual_increment.isNull()) {
        visual_increment = visual_increment.to3DoF();
      }
    }
    const bool visual_odom_observable =
        !visual_increment.isNull() &&
        visual_info.inliers >= kVisualOdomMinInliers &&
        visual_info.inliersRatio >= kVisualOdomMinInlierRatio &&
        visual_info.inliersDistribution >= kVisualOdomMinDistribution &&
        plausible_measured_increment(qvel_increment, visual_increment);
    const rtabmap::Transform icp_innovation =
        (qvel_increment.inverse() * measured_increment).to3DoF();
    const rtabmap::Transform visual_innovation =
        visual_odom_observable
            ? (qvel_increment.inverse() * visual_increment).to3DoF()
            : rtabmap::Transform::getIdentity();
    const bool icp_translation_innovation =
        std::hypot(icp_innovation.x(), icp_innovation.y()) >
        kFusionTranslationInnovationM;
    const bool icp_yaw_innovation =
        std::abs(icp_innovation.theta()) > kFusionYawInnovationRad;
    const bool visual_translation_innovation =
        std::hypot(visual_innovation.x(), visual_innovation.y()) >
        kFusionTranslationInnovationM;
    const bool visual_yaw_innovation =
        std::abs(visual_innovation.theta()) > kFusionYawInnovationRad;
    const bool observed_motion_candidate =
        predicted_motion ||
        (icp_translation_observable && icp_translation_innovation) ||
        (icp_yaw_supported && icp_yaw_innovation) ||
        (visual_odom_observable &&
         (visual_translation_innovation || visual_yaw_innovation));
    const bool predicted_but_observed_static =
        static_observation && observed_motion_candidate;
    // In a quarantined recovery/known-continuation interval qvel is only a
    // search prior on a weak translation axis. The visual increment updates
    // transient odometry; map writes remain governed by the existing state
    // machine and QuerySubmap transaction.
    const bool weak_axis_visual_translation =
        weak_axis_visual_probe && visual_odom_observable;
    const bool use_visual_translation =
        predicted_translation_motion && visual_odom_observable &&
        !icp_translation_observable &&
        (visual_translation_innovation || weak_axis_visual_translation);
    const bool quarantined_odometry =
        recovery_hold_requested || read_only_revisit_active_;
    const bool use_icp_translation =
        predicted_translation_motion && !use_visual_translation &&
        icp_translation_observable &&
        (icp_translation_innovation || quarantined_odometry);
    // A long wall observes yaw even when translation along it is degenerate.
    // It may therefore correct a genuine disagreement (including a blocked
    // turn), but never inject its sub-degree numerical bias on every frame.
    const bool use_icp_yaw =
        predicted_yaw_motion && icp_yaw_supported &&
        (icp_yaw_innovation || quarantined_odometry);
    const bool use_visual_yaw =
        predicted_yaw_motion && !icp_yaw_supported &&
        visual_odom_observable && visual_yaw_innovation;
    const bool keep_qvel_translation =
        !use_visual_translation && !use_icp_translation;
    const bool keep_qvel_yaw = !use_icp_yaw && !use_visual_yaw;
    rtabmap::Transform fused_increment(
        use_visual_translation
            ? visual_increment.x()
            : use_icp_translation ? measured_increment.x()
                                  : qvel_increment.x(),
        use_visual_translation
            ? visual_increment.y()
            : use_icp_translation ? measured_increment.y()
                                  : qvel_increment.y(),
        use_icp_yaw ? measured_increment.theta()
                    : use_visual_yaw ? visual_increment.theta()
                                     : qvel_increment.theta());
    if (predicted_but_observed_static) {
      fused_increment = rtabmap::Transform::getIdentity();
    }
    last_inliers_ = static_cast<std::uint32_t>(std::max(
        std::max(0, odom_info.reg.inliers),
        std::max(0, visual_info.inliers)));
    last_features_ = static_cast<std::uint32_t>(std::max(
        std::max(0, odom_info.features),
        std::max(0, visual_info.matches)));
    // Keep the pre-frame fused pose so a confirmed revisit can replace this
    // increment with the metric anchor while preserving the odometry chain
    // for all subsequent frames.
    const rtabmap::Transform fused_odom_before = fused_odom_pose_;
    fused_odom_pose_ = (fused_odom_pose_ * fused_increment).to3DoF();
    // Query views stay in the continuous pre-anchor odometry frame. A verified
    // bridge may rebase fused_odom_pose_ below; this copy is Q in the original
    // local chain and is therefore the endpoint used to aggregate held views.
    const rtabmap::Transform query_endpoint_odom_pose = fused_odom_pose_;
    const int registration_depth_pixels =
        registration_depth.empty()
            ? 0
            : cv::countNonZero(registration_depth > 0.0F);
    const bool registration_observation =
        !frame.rgb.empty() && !registration_depth.empty() &&
        registration_depth_pixels >= kRegistrationMinDepthPixels;
    // The graph neighbor edge must describe the source selected on each axis.
    // Never assign an ICP-small variance to an axis that retained qvel: doing
    // so makes a weak corridor direction artificially rigid and prevents later
    // absolute constraints from correcting it.
    cv::Mat covariance = use_visual_translation || use_visual_yaw
                             ? visual_info.covariance
                             : odom_info.reg.covariance;
    if (covariance.empty() || covariance.rows != 6 ||
        covariance.cols != 6 || covariance.type() != CV_64FC1) {
      covariance = cv::Mat::eye(6, 6, CV_64FC1);
      covariance.at<double>(0, 0) = covariance.at<double>(1, 1) = 0.01;
      covariance.at<double>(5, 5) = 0.01;
    } else {
      covariance = covariance.clone();
    }
    if (keep_qvel_translation) {
      covariance.at<double>(0, 0) = std::max(
          covariance.at<double>(0, 0), kQvelFallbackTranslationVariance);
      covariance.at<double>(1, 1) = std::max(
          covariance.at<double>(1, 1), kQvelFallbackTranslationVariance);
    }
    if (keep_qvel_yaw) {
      covariance.at<double>(5, 5) = std::max(
          covariance.at<double>(5, 5), kQvelFallbackYawVariance);
    }
    if (trace_enabled()) {
      std::cerr << "rgbd_odometry frame=" << frame.meta.frame_id
                << " qvel_x=" << qvel_increment.x()
                << " qvel_y=" << qvel_increment.y()
                << " qvel_yaw=" << qvel_increment.theta()
                << " measured_x=" << measured_increment.x()
                << " measured_y=" << measured_increment.y()
                << " measured_yaw=" << measured_increment.theta()
                << " fused_x=" << fused_increment.x()
                << " fused_y=" << fused_increment.y()
                << " fused_yaw=" << fused_increment.theta()
                << " recovered=" << (recovered_from_qvel ? 1 : 0)
                << " jump_rejected=" << (rejected_odometry_jump ? 1 : 0)
                << " source="
                << (predicted_but_observed_static
                        ? "static"
                        : use_visual_translation || use_visual_yaw
                              ? "visual"
                              : use_icp_translation || use_icp_yaw ? "icp"
                                                                  : "qvel")
                << " static=" << (static_observation ? 1 : 0)
                << " icp_complexity="
                << odom_info.reg.icpStructuralComplexity
                << " icp_correspondences="
                << odom_info.reg.icpCorrespondences
                << " icp_translation_supported="
                << (icp_translation_supported ? 1 : 0)
                << " icp_yaw_supported=" << (icp_yaw_supported ? 1 : 0)
                << " icp_inliers_ratio=" << odom_info.reg.icpInliersRatio
                << " icp_translation_observable="
                << (icp_translation_observable ? 1 : 0)
                << " use_icp_xy=" << (use_icp_translation ? 1 : 0)
                << " use_icp_yaw=" << (use_icp_yaw ? 1 : 0)
                << " use_visual_xy="
                << (use_visual_translation ? 1 : 0)
                << " use_visual_yaw=" << (use_visual_yaw ? 1 : 0)
                << " keep_qvel_xy=" << (keep_qvel_translation ? 1 : 0)
                << " keep_qvel_yaw=" << (keep_qvel_yaw ? 1 : 0)
                << " visual_evidence="
                << (visual_odom_observable ? 1 : 0)
                << " covariance_x=" << covariance.at<double>(0, 0)
                << " covariance_y=" << covariance.at<double>(1, 1)
                << " covariance_yaw=" << covariance.at<double>(5, 5)
                << " visual_inliers=" << visual_info.inliers
                << " visual_ratio=" << visual_info.inliersRatio
                << " visual_distribution="
                << visual_info.inliersDistribution
                << " innovation_xy="
                << std::hypot(icp_innovation.x(), icp_innovation.y())
                << " innovation_yaw=" << icp_innovation.theta()
                << " pose_x=" << fused_odom_pose_.x()
                << " pose_y=" << fused_odom_pose_.y()
                << " pose_yaw=" << fused_odom_pose_.theta()
                << " structural_observation="
                << (observation_usable ? 1 : 0)
                << " registration_depth_pixels=" << registration_depth_pixels
                << "\n";
    }
    const bool loop_prior_translation_observed =
        static_observation || use_visual_translation ||
        (icp_translation_supported && icp_translation_observable);
    const bool loop_prior_yaw_observed =
        static_observation || use_visual_yaw || icp_yaw_supported;
    odometry_history_.push_back(PlanarOdometryStep{
        fused_increment.to3DoF(),
        loop_prior_step_covariance(
            covariance, odom_info.reg.covariance, visual_info.covariance,
            static_observation, loop_prior_translation_observed,
            loop_prior_yaw_observed, use_visual_translation,
            use_visual_yaw)});
    // Keep the graph prior statistically independent from RGB-D. Query
    // submaps use the fused chain internally for local rigidity, while the
    // canonical A-Q factor is formed only from compliant base_qvel increments.
    raw_qvel_history_.push_back(PlanarOdometryStep{
        qvel_increment.to3DoF(),
        raw_qvel_step_covariance(qvel_increment)});
    UASSERT(raw_qvel_history_.size() == odometry_history_.size());
    // A verified bridge rebases the live fused chain and rewrites the final
    // history step below. Query evidence, however, is expressed in the
    // pre-bridge odometry frame. Preserve this step so every V->Q covariance
    // and the sole A->Q edge use the original compliant odometry evidence.
    const PlanarOdometryStep query_endpoint_history_step =
        odometry_history_.back();
    accumulate_slam_covariance(covariance);
    // Both mapping and localization accept only a chassis-centred structural
    // view.  A frozen grid is immutable, but feeding manipulation imagery to
    // localization would still corrupt transient visual/odometry memory.
    const bool process_observation =
        registration_observation && observation_usable;

    // RTAB-Map's public external-hypothesis API selects a loop candidate, but
    // it still inserts the current frame into Working Memory before validating
    // that candidate.  That is correct for ordinary graph mapping, but it is
    // precisely the operation that produced the second wall corner in the
    // capture.  Resolve the metric hypothesis before process() and make the
    // observation transient when it is a confirmed revisit.
    const bool was_mapping = mode_ == MappingMode::kMapping;
    recovery_hold_this_frame_ =
        observation_usable && recovery_hold_requested &&
        recovery_interruption_active_;
    const bool recovery_relocalization =
        recovery_hold_this_frame_ && recovery_metric_requested;
    if (observation_usable && !recovery_hold_requested &&
        recovery_interruption_active_) {
      recovery_interruption_active_ = false;
      recovery_unobserved_translation_m_ = 0.0F;
      recovery_unobserved_yaw_rad_ = 0.0F;
    }
    const rtabmap::Transform map_correction_before =
        slam_->getMapCorrection().isNull()
            ? rtabmap::Transform::getIdentity()
            : slam_->getMapCorrection().to3DoF();
    const rtabmap::Transform predicted_map_pose =
        (map_correction_before * fused_odom_pose_).to3DoF();
    session_motion_progress_m_ +=
        std::hypot(fused_increment.x(), fused_increment.y()) +
        kFreezeRotationEquivalentRadiusM * std::abs(fused_increment.theta());
    const float frame_motion_progress = session_motion_progress_m_;

    rtabmap::Transform external_measured_candidate_to_current;
    cv::Mat external_measurement_covariance;
    const bool external_metric_valid =
        external_loop_metric_measurement(
            frame.meta, &external_measured_candidate_to_current,
            &external_measurement_covariance);
    // Keep the odometry disagreement diagnostic alongside the pre-process
    // decision.  It is useful when a candidate is rejected before RTAB-Map
    // sees the frame and preserves the same metric convention as the native
    // external-link gate.
    if (external_metric_valid && trace_enabled()) {
      rtabmap::Transform qvel_candidate_to_current;
      cv::Mat qvel_covariance;
      if (external_loop_odometry_prior(
              frame.meta.external_loop_candidate_id,
              &qvel_candidate_to_current, &qvel_covariance)) {
        const rtabmap::Transform delta =
            (qvel_candidate_to_current.inverse() *
             external_measured_candidate_to_current).to3DoF();
        std::cerr << "external_loop_prior frame=" << frame.meta.frame_id
                  << " candidate=" << frame.meta.external_loop_candidate_id
                  << " qvel_disagreement_m="
                  << std::hypot(delta.x(), delta.y())
                  << " qvel_disagreement_yaw=" << delta.theta() << "\n";
      }
    }
    bool read_only_revisit = false;
    bool read_only_revisit_loop = false;
    bool verified_bridge_committed_this_frame = false;
    bool verified_bridge_candidate_this_frame = false;
    int staged_bridge_candidate_id = 0;
    rtabmap::Transform staged_candidate_to_current;
    cv::Mat staged_candidate_covariance;
    rtabmap::Transform staged_query_map_pose;
    rtabmap::Transform staged_fused_pose_before;
    bool staged_recovery_semantics = false;
    const bool native_novelty_query_resume_allowed =
        normal_global_search_pending_ && !normal_global_probe_complete_ &&
        query_scope == QueryScope::kNormal && query_generation > 0U &&
        !query_terminal_replay && !stale_query_generation &&
        !conflicting_active_query && !external_metric_valid &&
        query_outcome_this_frame_ == br::QueryOutcome::kHolding &&
        query_buffer_.active() &&
        query_buffer_.scope == QueryScope::kNormal &&
        query_buffer_.generation == query_generation &&
        query_buffer_.anchor_node_id > 0 && !query_buffer_.overflowed;
    bool native_novelty_query_resume_this_frame = false;

    if (!query_terminal_replay && !stale_query_generation &&
        !conflicting_active_query &&
        (was_mapping || recovery_relocalization) && slam_update_due &&
        process_observation &&
        read_only_revisit_can_start(
            frame.meta.external_loop_candidate_id > 0,
            external_metric_valid)) {
      rtabmap::Transform target_map_pose;
      double innovation_squared = 0.0;
      double correction_translation = 0.0;
      double correction_yaw = 0.0;
      const bool resolved = resolve_read_only_revisit(
          frame, map_correction_before, predicted_map_pose,
          external_measured_candidate_to_current,
          external_measurement_covariance, recovery_relocalization,
          &target_map_pose,
          &innovation_squared, &correction_translation, &correction_yaw);
      if (resolved) {
        const bool permanent_bridge_requested =
            recovery_relocalization || verified_bridge_requested;
        if (permanent_bridge_requested &&
            query_scope != QueryScope::kNone && query_generation != 0U) {
          // Delay the graph mutation until the provisional aggregate is
          // classified. Unknown coverage is represented by an explicit Q
          // node with A-Q and C-Q edges; only an all-known/rejected aggregate
          // is eliminated into the old-old A-C bridge.
          verified_bridge_candidate_this_frame = true;
          staged_bridge_candidate_id =
              frame.meta.external_loop_candidate_id;
          staged_candidate_to_current =
              external_measured_candidate_to_current;
          staged_candidate_covariance =
              external_measurement_covariance.clone();
          staged_query_map_pose = target_map_pose.to3DoF();
          staged_fused_pose_before = fused_odom_before.to3DoF();
          staged_recovery_semantics = recovery_relocalization;
          read_only_revisit = true;
        } else {
          verified_bridge_committed_this_frame =
              permanent_bridge_requested &&
              commit_verified_graph_bridge(
                  frame.meta.frame_id,
                  frame.meta.external_loop_candidate_id,
                  external_measured_candidate_to_current,
                  external_measurement_covariance, fused_odom_before,
                  "external_rgbd", recovery_relocalization,
                  query_scope, query_generation);
        }
        if (verified_bridge_committed_this_frame) {
          read_only_revisit = true;
          read_only_revisit_loop = true;
        } else if (!permanent_bridge_requested) {
          read_only_candidate_active_ = false;
          read_only_candidate_start_motion_m_ = 0.0F;
          read_only_candidate_unknown_streak_ = 0;
          read_only_candidate_no_observation_streak_ = 0;
          apply_read_only_anchor(
              fused_odom_before, map_correction_before, target_map_pose,
              external_measurement_covariance,
              frame.meta.external_loop_candidate_id);
          read_only_revisit = true;
          read_only_revisit_loop = true;
          read_only_match_this_frame_ = true;
          localized_this_frame_ = true;
          visual_localized_this_frame_ = true;
          geometric_localized_this_frame_ = true;
          read_only_revisit_active_ = true;
          soft_mapping_state_ = SoftMappingState::kKnownLocalizing;
          read_only_revisit_identity_verified_ = true;
          read_only_revisit_candidate_id_ =
              frame.meta.external_loop_candidate_id;
          read_only_revisit_start_travel_m_ = frame_motion_progress;
          reset_soft_novelty_evidence();
          read_only_revisit_no_observation_streak_ = 0;
          read_only_scan_match_failures_ = 0;
          read_only_scan_match_next_stamp_s_ =
              frame.meta.stamp + kReadOnlyScanMatchRetryIntervalS;
          if (read_only_revisit_count_ <
              std::numeric_limits<std::uint32_t>::max()) {
            ++read_only_revisit_count_;
          }
        } else {
          // A rejected permanent factor is a transaction abort. Keep the
          // current soft/recovery hold and do not anchor, process, or rasterize
          // this observation.
          read_only_revisit = true;
          native_current_pose_ = predicted_map_pose;
          current_pose_ = native_current_pose_;
        }
        if (trace_enabled()) {
          std::cerr << "read_only_revisit_anchor frame="
                    << frame.meta.frame_id << " candidate="
                    << frame.meta.external_loop_candidate_id
                    << " innovation=" << innovation_squared
                    << " correction_m=" << correction_translation
                    << " correction_yaw=" << correction_yaw
                    << " recovery="
                    << (recovery_relocalization ? 1 : 0)
                    << " graph_bridge="
                    << (verified_bridge_committed_this_frame ? 1 : 0)
                    << " target_x=" << target_map_pose.x()
                    << " target_y=" << target_map_pose.y()
                    << " target_yaw=" << target_map_pose.theta() << "\n";
        }
      } else if (trace_enabled()) {
        std::cerr << "read_only_revisit_rejected frame="
                  << frame.meta.frame_id << " candidate="
                  << frame.meta.external_loop_candidate_id << "\n";
      }
    }

    // A mature graph has already observed this place at least once.  If the
    // chassis has not commanded any planar translation, do not let the
    // registration backend's small rotational jitter reopen ordinary mapping.
    // Enter the existing transient localization path so a historical scan can
    // still anchor the pose; only a later translational, multi-view novelty
    // decision may release the frame back to Building.
    const bool known_revisit_stationary =
        known_revisit_stationary_hold(
            was_mapping, loop_count_,
            predicted_translation_motion, query_scope != QueryScope::kNone,
            external_metric_valid) &&
        !recovery_hold_this_frame_ && !verified_bridge_requested;
    if (known_revisit_stationary && !read_only_revisit_active_ &&
        !read_only_candidate_active_) {
      soft_mapping_state_ = SoftMappingState::kUncertainHold;
      read_only_revisit_active_ = true;
      read_only_revisit_identity_verified_ = true;
      read_only_revisit_candidate_id_ = 0;
      read_only_revisit_start_travel_m_ = frame_motion_progress;
      read_only_revisit_no_observation_streak_ = 0;
      read_only_scan_match_failures_ = 0;
      reset_soft_novelty_evidence();
      read_only_scan_match_next_stamp_s_ =
          -std::numeric_limits<double>::infinity();
      if (trace_enabled()) {
        std::cerr << "known_revisit_stationary_hold frame="
                  << frame.meta.frame_id << " loops="
                  << accepted_loop_pairs_.size() << "\n";
      }
    }

    // Continue a transient localization window after its anchor. Scan-to-map
    // windows are periodically re-anchored against old metric geometry; qvel
    // only propagates between those measurements. Registration failures enter
    // a no-write hold. Mapping reopens only after sustained unknown structure
    // is also observed across a sensor-scale translational baseline.
    if (was_mapping && !read_only_revisit && read_only_revisit_active_) {
      float revisit_novelty = 0.0F;
      if (process_observation && !frame.height_points.empty()) {
        revisit_novelty = observation_novelty_ratio(
            frame, predicted_map_pose,
            &last_observation_endpoint_novelty_ratio_,
            &last_observation_ray_novelty_ratio_);
        last_observation_novelty_ratio_ = revisit_novelty;
        read_only_revisit_no_observation_streak_ = 0;
      } else {
        revisit_novelty = 0.0F;
        if (read_only_revisit_no_observation_streak_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++read_only_revisit_no_observation_streak_;
        }
      }
      bool scan_match_attempted = false;
      bool scan_match_observed = false;
      bool scan_match_committed = false;
      rtabmap::Transform novelty_evidence_pose = predicted_map_pose;
      rtabmap::RegistrationInfo scan_match_info;
      std::size_t scan_match_reference_nodes = 0;
      float scan_match_accepted_novelty = revisit_novelty;
      if (process_observation && !frame.height_points.empty() &&
          poses_.size() >= kReadOnlyScanMatchMinMapNodes &&
          read_only_revisit_identity_verified_ &&
          frame.meta.stamp >= read_only_scan_match_next_stamp_s_) {
        scan_match_attempted = true;
        rtabmap::Transform target_map_pose;
        cv::Mat scan_match_covariance;
        rtabmap::Transform confirmed_target_map_pose;
        cv::Mat confirmed_scan_match_covariance;
        scan_match_observed = resolve_read_only_scan_match(
            frame, predicted_map_pose, frame_motion_progress,
            &target_map_pose, &scan_match_covariance, &scan_match_info,
            &scan_match_accepted_novelty, &scan_match_reference_nodes);
        read_only_scan_match_next_stamp_s_ =
            frame.meta.stamp + kReadOnlyScanMatchRetryIntervalS;
        if (scan_match_observed) {
          novelty_evidence_pose = target_map_pose.to3DoF();
          // A valid geometric observation is not a localization failure just
          // because it is still waiting for an independent second view.
          read_only_scan_match_failures_ = 0;
          scan_match_committed = confirm_read_only_scan_match_proposal(
              frame.meta.frame_id, target_map_pose, scan_match_covariance,
              &confirmed_target_map_pose,
              &confirmed_scan_match_covariance);
          if (trace_enabled() && !scan_match_committed) {
            const rtabmap::Transform correction =
                (predicted_map_pose.inverse() * target_map_pose).to3DoF();
            std::cerr << "read_only_scan_match_proposal frame="
                      << frame.meta.frame_id
                      << " correspondences="
                      << scan_match_info.icpCorrespondences
                      << " ratio=" << scan_match_info.icpInliersRatio
                      << " complexity="
                      << scan_match_info.icpStructuralComplexity
                      << " correction_m="
                      << std::hypot(correction.x(), correction.y())
                      << " correction_yaw=" << correction.theta()
                      << " reference_nodes="
                      << scan_match_reference_nodes << "\n";
          }
        }
        if (scan_match_committed) {
          apply_read_only_anchor(
              fused_odom_before, map_correction_before,
              confirmed_target_map_pose,
              confirmed_scan_match_covariance, 0);
          read_only_revisit = true;
          read_only_match_this_frame_ = true;
          localized_this_frame_ = true;
          geometric_localized_this_frame_ = true;
          // The visual anchor established place identity. Once historical
          // geometry also agrees, subsequent refresh failures are meaningful
          // and use the scan-localization exit policy.
          read_only_revisit_candidate_id_ = 0;
          read_only_scan_match_failures_ = 0;
          // The distance cap is a safety bound since the last trustworthy
          // metric localization, not since soft localization first began.
          // Resetting it prevents a long known traversal from periodically
          // reopening mapping and drawing duplicate walls.
          read_only_revisit_start_travel_m_ = frame_motion_progress;
          soft_mapping_state_ = SoftMappingState::kKnownLocalizing;
          reset_soft_novelty_evidence();
          last_observation_novelty_ratio_ = scan_match_accepted_novelty;
          if (trace_enabled()) {
            const rtabmap::Transform correction =
                (predicted_map_pose.inverse() *
                 confirmed_target_map_pose).to3DoF();
            std::cerr << "read_only_scan_match_refresh frame="
                      << frame.meta.frame_id
                      << " correspondences="
                      << scan_match_info.icpCorrespondences
                      << " ratio=" << scan_match_info.icpInliersRatio
                      << " complexity="
                      << scan_match_info.icpStructuralComplexity
                      << " correction_m="
                      << std::hypot(correction.x(), correction.y())
                      << " correction_yaw=" << correction.theta()
                      << " novelty=" << scan_match_accepted_novelty
                      << " reference_nodes="
                      << scan_match_reference_nodes << "\n";
          }
        } else if (!scan_match_observed &&
                   read_only_scan_match_failures_ <
                       std::numeric_limits<std::uint32_t>::max()) {
          ++read_only_scan_match_failures_;
        }
      }
      if (process_observation && !frame.height_points.empty()) {
        // A drifted prediction can make familiar geometry appear novel in the
        // old raster. Prefer the novelty measured at the accepted historical
        // scan pose whenever one is available, and only then update the
        // multi-view resume evidence. Otherwise the very pose error that
        // triggered localization can authorize duplicate map writes.
        if (scan_match_observed) {
          revisit_novelty = scan_match_accepted_novelty;
          last_observation_novelty_ratio_ = revisit_novelty;
        }
        note_soft_novelty_evidence(
            revisit_novelty, novelty_evidence_pose, frame.meta.stamp,
            predicted_but_observed_static);
      }
      const float revisit_motion =
          frame_motion_progress - read_only_revisit_start_travel_m_;
      const bool within_motion =
          std::isfinite(revisit_motion) &&
          revisit_motion <= kReadOnlyRevisitMaxTranslationHoldM;
      const bool within_observation =
          read_only_revisit_no_observation_streak_ <=
              kReadOnlyRevisitNoObservationGrace &&
          read_only_scan_match_failures_ < kReadOnlyScanMatchMaxFailures;
      const bool novelty_resume =
          !known_revisit_stationary &&
          (!normal_global_search_pending_ || normal_global_probe_complete_ ||
           native_novelty_query_resume_allowed) &&
          soft_novelty_ready(frame.meta.stamp);
      if (novelty_resume) {
        native_novelty_query_resume_this_frame =
            native_novelty_query_resume_allowed;
        arm_novelty_resume_occupancy_reconciliation(
            frame.meta.frame_id, predicted_map_pose);
        if (trace_enabled()) {
          std::cerr << "mapping_soft_resumed frame=" << frame.meta.frame_id
                    << " reason=sustained_novelty"
                    << " novelty=" << revisit_novelty
                    << " observations=" << read_only_revisit_unknown_streak_
                    << " viewpoints=" << soft_novelty_distinct_viewpoints_
                    << " translation_span_m="
                    << soft_novelty_max_translation_m_
                    << " duration="
                    << frame.meta.stamp - soft_novelty_start_stamp_s_ << "\n";
        }
        soft_mapping_state_ = SoftMappingState::kBuilding;
        read_only_revisit_active_ = false;
        read_only_revisit_identity_verified_ = false;
        read_only_revisit_candidate_id_ = 0;
        read_only_revisit_start_travel_m_ = 0.0F;
        read_only_revisit_no_observation_streak_ = 0;
        read_only_scan_match_failures_ = 0;
        clear_read_only_scan_match_proposal();
        reset_soft_novelty_evidence();
        normal_global_probe_complete_ = false;
      } else if (!scan_match_committed) {
        if (!within_motion || !within_observation) {
          soft_mapping_state_ = SoftMappingState::kUncertainHold;
        }
        read_only_revisit = true;
        read_only_match_this_frame_ =
            soft_mapping_state_ != SoftMappingState::kUncertainHold;
        native_current_pose_ = predicted_map_pose;
        current_pose_ = native_current_pose_;
        last_ref_node_id_ = read_only_revisit_candidate_id_;
        if (trace_enabled()) {
          std::cerr << (soft_mapping_state_ ==
                                SoftMappingState::kUncertainHold
                            ? "read_only_uncertain_hold frame="
                            : "read_only_revisit_hold frame=")
                    << frame.meta.frame_id
                    << " candidate=" << read_only_revisit_candidate_id_
                    << " novelty=" << revisit_novelty
                    << " unknown_streak=" << read_only_revisit_unknown_streak_
                    << " no_observation_streak="
                    << read_only_revisit_no_observation_streak_
                    << " scan_attempted="
                    << (scan_match_attempted ? 1 : 0)
                    << " scan_failures="
                    << read_only_scan_match_failures_
                    << " motion=" << revisit_motion
                    << " novelty_translation_span_m="
                    << soft_novelty_max_translation_m_ << "\n";
        }
      }
    }

    // A local geometric overlap may be visible a few observations before the
    // asynchronous appearance matcher has enough temporal consensus to name
    // the old place. Stage those observations instead of irreversibly writing
    // them into the global map. The ICP transform is deliberately not applied:
    // repeated corridors can align metrically while still being the wrong
    // place. Unknown coverage or one sensor-scale motion window without an
    // appearance confirmation moves staging into an uncertain no-write hold.
    // Only sustained, multi-view novelty may reopen mapping.
    if (was_mapping && !read_only_revisit && !read_only_revisit_active_ &&
        read_only_candidate_active_) {
      float candidate_novelty = 0.0F;
      if (process_observation && !frame.height_points.empty()) {
        candidate_novelty = observation_novelty_ratio(
            frame, predicted_map_pose,
            &last_observation_endpoint_novelty_ratio_,
            &last_observation_ray_novelty_ratio_);
        last_observation_novelty_ratio_ = candidate_novelty;
        read_only_candidate_no_observation_streak_ = 0;
        note_soft_novelty_evidence(
            candidate_novelty, predicted_map_pose, frame.meta.stamp,
            predicted_but_observed_static);
        if (candidate_novelty > kReadOnlyRevisitMaxNoveltyRatio) {
          if (read_only_candidate_unknown_streak_ <
              std::numeric_limits<std::uint32_t>::max()) {
            ++read_only_candidate_unknown_streak_;
          }
        } else {
          read_only_candidate_unknown_streak_ = 0;
        }
      } else if (read_only_candidate_no_observation_streak_ <
                 std::numeric_limits<std::uint32_t>::max()) {
        ++read_only_candidate_no_observation_streak_;
      }
      const float candidate_motion =
          frame_motion_progress - read_only_candidate_start_motion_m_;
      const bool keep_staging =
          std::isfinite(candidate_motion) &&
          read_only_candidate_can_hold(
              candidate_motion, read_only_candidate_unknown_streak_,
              read_only_candidate_no_observation_streak_);
      const bool novelty_resume =
          !known_revisit_stationary &&
          (!normal_global_search_pending_ || normal_global_probe_complete_ ||
           native_novelty_query_resume_allowed) &&
          soft_novelty_ready(frame.meta.stamp);
      if (novelty_resume) {
        native_novelty_query_resume_this_frame =
            native_novelty_query_resume_allowed;
        arm_novelty_resume_occupancy_reconciliation(
            frame.meta.frame_id, predicted_map_pose);
        if (trace_enabled()) {
          std::cerr << "mapping_soft_resumed frame=" << frame.meta.frame_id
                    << " reason=sustained_novelty_from_candidate"
                    << " novelty=" << candidate_novelty
                    << " observations=" << read_only_revisit_unknown_streak_
                    << " viewpoints=" << soft_novelty_distinct_viewpoints_
                    << " translation_span_m="
                    << soft_novelty_max_translation_m_
                    << " duration="
                    << frame.meta.stamp - soft_novelty_start_stamp_s_ << "\n";
        }
        soft_mapping_state_ = SoftMappingState::kBuilding;
        read_only_candidate_active_ = false;
        read_only_candidate_start_motion_m_ = 0.0F;
        read_only_candidate_unknown_streak_ = 0;
        read_only_candidate_no_observation_streak_ = 0;
        clear_read_only_scan_match_proposal();
        reset_soft_novelty_evidence();
        normal_global_probe_complete_ = false;
      } else if (keep_staging) {
        soft_mapping_state_ = SoftMappingState::kCandidateHold;
        read_only_revisit = true;
        read_only_match_this_frame_ = true;
        native_current_pose_ = predicted_map_pose;
        current_pose_ = native_current_pose_;
        if (trace_enabled()) {
          std::cerr << "read_only_candidate_hold frame="
                    << frame.meta.frame_id
                    << " novelty=" << candidate_novelty
                    << " unknown_streak="
                    << read_only_candidate_unknown_streak_
                    << " no_observation_streak="
                    << read_only_candidate_no_observation_streak_
                    << " motion=" << candidate_motion
                    << " novelty_translation_span_m="
                    << soft_novelty_max_translation_m_ << "\n";
        }
      } else {
        if (trace_enabled()) {
          std::cerr << "read_only_candidate_exit frame="
                    << frame.meta.frame_id
                    << " novelty=" << candidate_novelty
                    << " unknown_streak="
                    << read_only_candidate_unknown_streak_
                    << " no_observation_streak="
                    << read_only_candidate_no_observation_streak_
                    << " motion=" << candidate_motion << "\n";
        }
        read_only_candidate_active_ = false;
        read_only_candidate_start_motion_m_ = 0.0F;
        read_only_candidate_unknown_streak_ = 0;
        read_only_candidate_no_observation_streak_ = 0;
        read_only_revisit_active_ = true;
        read_only_revisit_identity_verified_ = false;
        read_only_revisit_candidate_id_ = 0;
        read_only_revisit_start_travel_m_ = frame_motion_progress;
        read_only_scan_match_failures_ = 0;
        soft_mapping_state_ = SoftMappingState::kUncertainHold;
        read_only_revisit = true;
        read_only_match_this_frame_ = false;
        native_current_pose_ = predicted_map_pose;
        current_pose_ = native_current_pose_;
      }
    }

    if (native_novelty_query_resume_this_frame) {
      const int terminal_anchor_id = query_buffer_.anchor_node_id;
      const auto anchor = poses_.find(terminal_anchor_id);
      if (query_buffer_.scope != QueryScope::kNormal ||
          query_buffer_.generation != query_generation ||
          terminal_anchor_id <= 0 || anchor == poses_.end() ||
          !finite_planar_transform(anchor->second)) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "native novelty query terminal has an invalid anchor");
      }
      // Sustained multi-view native novelty is a terminal proof that the
      // provisional normal query should be discarded, not promoted. Make the
      // frozen anchor and graph durable before the graphless ACK, then release
      // the current frame to the ordinary mapping path without adding Q or A-C.
      query_integrity_failed_ = true;
      slam_->checkpointCommittedGraph();
      persist_graphless_query_terminal(
          QueryScope::kNormal, query_generation, terminal_anchor_id,
          br::QueryOutcome::kNoveltyResumed);
      test_only_exit_after_novelty_resumed_ledger();
      mark_query_generation_completed(
          QueryScope::kNormal, query_generation,
          br::QueryOutcome::kNoveltyResumed);
      query_buffer_.clear();
      query_outcome_this_frame_ = br::QueryOutcome::kNoveltyResumed;
      normal_global_search_pending_ = false;
      normal_global_probe_complete_ = false;
      native_novelty_resume_guard_updates_ = std::max(
          native_novelty_resume_guard_updates_,
          kNativeNoveltyResumeGuardUpdates);
      query_integrity_failed_ = false;
      if (trace_enabled()) {
        std::cerr << "query_native_novelty_resumed frame="
                  << frame.meta.frame_id << " generation="
                  << query_generation << " anchor=" << terminal_anchor_id
                  << "\n";
      }
    }

    if (was_mapping &&
        !native_novelty_query_resume_this_frame &&
        !native_novelty_resume_reconciliation_pending_ &&
        soft_mapping_state_ == SoftMappingState::kBuilding &&
        !read_only_revisit && !read_only_revisit_active_ &&
        !read_only_candidate_active_ && slam_update_due &&
        process_observation && !frame.height_points.empty() &&
        !cached_map_.empty() &&
        poses_.size() >= kReadOnlyScanMatchMinMapNodes &&
        accepted_loop_pairs_.size() >=
            kReadOnlyScanMatchMinLoopEvidence &&
        icp_translation_observable &&
        frame.meta.stamp >= read_only_scan_match_next_stamp_s_) {
      const float novelty = observation_novelty_ratio(
          frame, predicted_map_pose,
          &last_observation_endpoint_novelty_ratio_,
          &last_observation_ray_novelty_ratio_);
      last_observation_novelty_ratio_ = novelty;
      if (novelty <= kReadOnlyCandidatePrefilterNoveltyRatio) {
        rtabmap::Transform staged_target_pose;
        cv::Mat staged_covariance;
        rtabmap::RegistrationInfo staged_info;
        float accepted_novelty = novelty;
        std::size_t reference_nodes = 0;
        const bool geometry_overlaps = resolve_read_only_scan_match(
            frame, predicted_map_pose, frame_motion_progress,
            &staged_target_pose, &staged_covariance, &staged_info,
            &accepted_novelty, &reference_nodes);
        read_only_scan_match_next_stamp_s_ =
            frame.meta.stamp + kReadOnlyScanMatchRetryIntervalS;
        if (geometry_overlaps) {
          soft_mapping_state_ = SoftMappingState::kCandidateHold;
          read_only_candidate_active_ = true;
          read_only_candidate_start_motion_m_ = frame_motion_progress;
          read_only_candidate_unknown_streak_ = 0;
          read_only_candidate_no_observation_streak_ = 0;
          reset_soft_novelty_evidence();
          normal_global_probe_complete_ = false;
          read_only_revisit = true;
          read_only_match_this_frame_ = true;
          native_current_pose_ = predicted_map_pose;
          current_pose_ = native_current_pose_;
          if (trace_enabled()) {
            const rtabmap::Transform correction =
                (predicted_map_pose.inverse() * staged_target_pose).to3DoF();
            std::cerr << "read_only_candidate_start frame="
                      << frame.meta.frame_id << " novelty=" << novelty
                      << " accepted_novelty=" << accepted_novelty
                      << " correspondences="
                      << staged_info.icpCorrespondences
                      << " ratio=" << staged_info.icpInliersRatio
                      << " complexity="
                      << staged_info.icpStructuralComplexity
                      << " proposed_correction_m="
                      << std::hypot(correction.x(), correction.y())
                      << " proposed_correction_yaw="
                      << correction.theta()
                      << " reference_nodes=" << reference_nodes << "\n";
          }
        } else if (trace_enabled()) {
          std::cerr << "read_only_candidate_rejected frame="
                    << frame.meta.frame_id << " novelty=" << novelty
                    << " correspondences=" << staged_info.icpCorrespondences
                    << " ratio=" << staged_info.icpInliersRatio
                    << " complexity="
                    << staged_info.icpStructuralComplexity
                    << " reference_nodes=" << reference_nodes
                    << " reason=" << staged_info.rejectedMsg << "\n";
        }
      }
    }

    // recovery_hold quarantines a recovered structural view until the
    // external appearance+RGB-D verifier either commits a graph bridge or
    // releases it as novel. The main RTAB-Map instance stays incremental and
    // receives no transient recovery signature.
    const bool recovery_mapping_hold =
        recovery_hold_this_frame_ && mode_ == MappingMode::kMapping;
    const bool recovery_negative_release =
        recovery_global_no_mode_requested && recovery_mapping_hold &&
        !external_metric_valid;
    const bool normal_negative_release =
        normal_global_no_mode_requested && normal_global_search_pending_ &&
        soft_mapping_state_ == SoftMappingState::kBuilding &&
        !read_only_revisit && !external_metric_valid;
    const bool matching_query_buffer =
        query_buffer_.active() && query_buffer_.scope == query_scope &&
        query_buffer_.generation == query_generation;
    // Any valid current RGB-D metric observation disproves an older no-mode
    // release, even if graph consistency does not turn it into a bridge. Never
    // let that stale negative certificate authorize mapping on a later frame.
    if (matching_query_buffer && external_metric_valid) {
      query_buffer_.latched_release_kind = QueryReleaseKind::kNone;
    }
    const QueryReleaseKind fresh_query_release_kind =
        verified_bridge_candidate_this_frame
            ? QueryReleaseKind::kPositiveMetric
            : (recovery_negative_release || normal_negative_release)
                  ? QueryReleaseKind::kNegativeNoMode
                  : QueryReleaseKind::kNone;
    const QueryReleaseKind query_release_kind =
        fresh_query_release_kind != QueryReleaseKind::kNone
            ? fresh_query_release_kind
            : matching_query_buffer
                  ? query_buffer_.latched_release_kind
                  : QueryReleaseKind::kNone;
    const bool query_release =
        query_release_kind != QueryReleaseKind::kNone;
    const bool positive_query_release =
        query_release_kind == QueryReleaseKind::kPositiveMetric;
    const bool negative_query_release =
        query_release_kind == QueryReleaseKind::kNegativeNoMode;
    const bool force_query_keyframe =
        fresh_query_release_kind != QueryReleaseKind::kNone ||
        (negative_query_release && matching_query_buffer &&
         !query_buffer_.promotion_attempted);
    const bool query_quarantine =
        recovery_mapping_hold || normal_global_search_pending_ ||
        orphan_query_hold;
    const bool query_frame_eligible =
        was_mapping && slam_update_due && process_observation;
    bool query_frame_usable_for_promotion = false;
    if (query_frame_eligible && query_scope != QueryScope::kNone &&
        !query_terminal_replay && !stale_query_generation &&
        (query_quarantine || query_release)) {
      query_frame_usable_for_promotion = collect_query_view(
          query_scope, query_generation, frame, query_endpoint_odom_pose,
          query_release_kind, force_query_keyframe);
    }

    bool query_promotion = false;
    bool suppress_query_release = query_release;
    QueryScope promoted_query_scope = QueryScope::kNone;
    std::uint64_t promoted_query_generation = 0;
    int promoted_query_anchor_id = 0;
    rtabmap::Transform promoted_query_map_pose;
    rtabmap::Transform promoted_query_odom_pose;
    rtabmap::Transform promoted_query_neighbor_transform;
    cv::Mat promoted_query_covariance;
    std::vector<HeightPoint> promoted_height_points;
    QueryPromotionKind promoted_query_kind = QueryPromotionKind::kNone;
    bool promoted_query_has_metric_bridge = false;
    int promoted_query_candidate_id = 0;
    rtabmap::Transform promoted_candidate_to_query;
    cv::Mat promoted_candidate_covariance;
    cv::Mat promoted_metric_information;
    rtabmap::Transform promoted_fused_pose_before;
    bool promoted_recovery_semantics = false;
    if ((positive_query_release || negative_query_release) &&
        !verified_bridge_committed_this_frame &&
        !query_terminal_replay && !stale_query_generation &&
        query_frame_eligible && query_frame_usable_for_promotion &&
        query_buffer_.active() &&
        query_buffer_.scope == query_scope &&
        query_buffer_.generation == query_generation &&
        !query_buffer_.overflowed &&
        !query_buffer_.promotion_attempted &&
        !query_generation_completed(query_scope, query_generation)) {
      const int promotion_anchor_id = query_buffer_.anchor_node_id;
      rtabmap::Transform promotion_edge_transform;
      cv::Mat promotion_edge_covariance;
      const rtabmap::Signature anchor_signature =
          promotion_anchor_id > 0
              ? slam_->getSignatureCopy(promotion_anchor_id, false, false,
                                        false, false, false, false)
              : rtabmap::Signature();
      const rtabmap::Signature current_map_signature =
          last_mapping_node_id_ > 0
              ? slam_->getSignatureCopy(last_mapping_node_id_, false, false,
                                        false, false, false, false)
              : rtabmap::Signature();
      const rtabmap::Transform anchor_map_pose =
          promotion_anchor_id > 0
              ? slam_->getPose(promotion_anchor_id).to3DoF()
              : rtabmap::Transform();
      const rtabmap::Transform anchor_odom_pose =
          anchor_signature.id() == promotion_anchor_id
              ? anchor_signature.getPose().to3DoF()
              : rtabmap::Transform();
      const bool promotion_context_valid =
          promotion_anchor_id > 0 && anchor_signature.id() > 0 &&
          current_map_signature.id() > 0 &&
          (!positive_query_release ||
           staged_bridge_candidate_id != promotion_anchor_id) &&
          anchor_signature.mapId() == current_map_signature.mapId() &&
          finite_planar_transform(anchor_map_pose) &&
          finite_planar_transform(anchor_odom_pose) &&
          query_anchor_edge_covariance(
              promotion_anchor_id, &promotion_edge_transform,
              &promotion_edge_covariance);
      if (!promotion_context_valid && !positive_query_release) {
        // The frozen anchor and raw-qvel history are immutable for this
        // transaction, so collecting another chunk cannot repair this state.
        // Fail the session explicitly instead of quarantining forever or
        // silently falling through to ordinary mapping.
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "negative query has no durable raw-qvel anchor context");
      }
      const rtabmap::Transform promotion_map_pose =
          positive_query_release
              ? staged_query_map_pose
              : promotion_context_valid
                    ? (anchor_map_pose * promotion_edge_transform).to3DoF()
                    : rtabmap::Transform();
      const cv::Mat promotion_covariance =
          positive_query_release
              ? staged_candidate_covariance.clone()
              : promotion_edge_covariance.clone();
      QueryAggregate aggregate;
      if (promotion_context_valid) {
        aggregate = build_query_aggregate(
            query_endpoint_odom_pose, promotion_map_pose,
            promotion_covariance,
            positive_query_release
                ? staged_candidate_to_current
                : promotion_edge_transform,
            positive_query_release,
            query_endpoint_history_step);
      }
      if (negative_query_release && aggregate.uncertainty_exceeded) {
        // The safety tube is measured from the immutable A anchor. Its
        // uncertainty only grows as more raw-qvel motion is integrated, so an
        // evidence reset cannot make a later attempt valid. Terminate this
        // client session rather than enter an unbounded HOLD loop.
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "negative query exceeded the safe raw-qvel uncertainty bound");
      }
      if (aggregate.valid && aggregate.all_known &&
          positive_query_release) {
        if (query_all_known_discards_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++query_all_known_discards_;
        }
        if (trace_enabled()) {
          std::cerr << "query_all_known_discard frame=" << frame.meta.frame_id
                    << " generation=" << query_generation << " scope="
                    << static_cast<int>(query_scope)
                    << " duplicate_cells=" << aggregate.duplicate_cells
                    << "\n";
        }
      } else if (aggregate.valid && aggregate.all_known) {
        // A negative certificate plus a wholly known aggregate has no graph or
        // raster work to commit. Persist this otherwise graph-less terminal
        // before releasing the hold so worker restart can replay the exact ACK.
        // The ledger references A by id, so A and the complete pre-query graph
        // must be durable before the first ledger byte is appended.
        query_integrity_failed_ = true;
        slam_->checkpointCommittedGraph();
        persist_graphless_query_terminal(
            query_scope, query_generation, promotion_anchor_id,
            br::QueryOutcome::kAllKnown);
        test_only_exit_after_all_known_ledger();
        mark_query_generation_completed(query_scope, query_generation,
                                        br::QueryOutcome::kAllKnown);
        query_buffer_.clear();
        query_outcome_this_frame_ = br::QueryOutcome::kAllKnown;
        query_integrity_failed_ = false;
        if (query_all_known_discards_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++query_all_known_discards_;
        }
      }
      if (aggregate.valid && !aggregate.all_known) {
        promoted_query_neighbor_transform =
            promotion_edge_transform.to3DoF();
        promoted_query_covariance = promotion_edge_covariance.clone();
        promoted_query_map_pose =
            (anchor_map_pose * promoted_query_neighbor_transform).to3DoF();
        promoted_query_odom_pose =
            (anchor_odom_pose * promoted_query_neighbor_transform).to3DoF();
        if (!finite_planar_transform(promoted_query_map_pose) ||
            !finite_planar_transform(promoted_query_odom_pose)) {
          aggregate.valid = false;
        }
      }
      if (aggregate.valid && !aggregate.all_known &&
          negative_query_release) {
        // A negative promotion has no later C-Q optimization. Its only graph
        // pose is optimized(A)*T_AQ(raw qvel), so unknown/conflict filtering
        // must use that exact pose rather than the fused prediction used to
        // search for an old place.
        aggregate = build_query_aggregate(
            query_endpoint_odom_pose, promoted_query_map_pose,
            promoted_query_covariance, promoted_query_neighbor_transform,
            false, query_endpoint_history_step);
      }
      if (aggregate.valid && !aggregate.all_known) {
        promoted_query_kind = positive_query_release
                                  ? QueryPromotionKind::kPositive
                                  : QueryPromotionKind::kNegative;
        data.setOccupancyGrid(
            aggregate.ground, aggregate.obstacles, aggregate.empty,
            kGridCellM, cv::Point3f(0.0F, 0.0F, 0.0F));
        // The Q node stays RAM-only until every graph, payload and raster
        // invariant below succeeds. A HOLDING marker is never a terminal ACK
        // and must never reach a cleanly reopened database.
        data.setUserData(encode_persisted_query(
            query_scope, query_generation, br::QueryOutcome::kHolding,
            promoted_query_kind, promotion_anchor_id,
            positive_query_release ? staged_bridge_candidate_id : 0,
            {}));
        query_promotion = true;
        suppress_query_release = false;
        promoted_query_scope = query_scope;
        promoted_query_generation = query_generation;
        promoted_query_anchor_id = promotion_anchor_id;
        promoted_height_points = std::move(aggregate.height_points);
        promoted_fused_pose_before = fused_odom_before.to3DoF();
        promoted_query_has_metric_bridge = positive_query_release;
        if (promoted_query_has_metric_bridge) {
          promoted_query_candidate_id = staged_bridge_candidate_id;
          promoted_candidate_to_query = staged_candidate_to_current.to3DoF();
          promoted_candidate_covariance =
              staged_candidate_covariance.clone();
          promoted_fused_pose_before = staged_fused_pose_before.to3DoF();
          promoted_recovery_semantics = staged_recovery_semantics;
        }
        if (trace_enabled()) {
          std::cerr << "query_promotion_ready frame=" << frame.meta.frame_id
                    << " generation=" << query_generation << " scope="
                    << static_cast<int>(query_scope)
                    << " positive_bridge="
                    << (positive_query_release ? 1 : 0)
                    << " free=" << aggregate.accepted_free_cells
                    << " obstacles=" << aggregate.accepted_obstacle_cells
                    << " duplicate=" << aggregate.duplicate_cells
                    << " conflicts=" << aggregate.conflict_cells
                    << " unsupported=" << aggregate.unsupported_cells
                    << " safe_supports=" << aggregate.safe_supports
                    << " unsafe_supports=" << aggregate.unsafe_supports
                    << " unsafe_components=" << aggregate.unsafe_components
                    << " uncertain_veto_components="
                    << aggregate.uncertain_veto_components
                    << " uncertain_veto_cells="
                    << aggregate.uncertain_veto_cells
                    << " promotion_uncertainty_m="
                    << aggregate.promotion_uncertainty_m
                    << " safe_promotion_uncertainty_m="
                    << aggregate.safe_promotion_uncertainty_m
                    << " uncertainty_domain_exceeded="
                    << (aggregate.uncertainty_domain_exceeded ? 1 : 0)
                    << " uncertainty_domain_cells="
                    << aggregate.uncertainty_domain_cells
                    << " uncertainty_radius_buckets="
                    << aggregate.uncertainty_radius_buckets
                    << " uncertainty_domain_work_cells="
                    << aggregate.uncertainty_domain_work_cells
                    << "\n";
        }
      } else if (query_outcome_this_frame_ == br::QueryOutcome::kHolding &&
                 (!aggregate.all_known ||
                  !positive_query_release)) {
        // This rejection happened before processQuerySignature(), so the
        // committed graph and raster are unchanged. Keep every earlier view,
        // conflict and uncertainty veto. Only appending a genuinely new valid
        // keyframe clears this latch and permits one more aggregate attempt.
        // Positive measurements remain frame-local; the A-C fallback below may
        // still terminate this exact frame, but no later frame may reuse it.
        query_buffer_.promotion_attempted = true;
        if (query_promotion_rejections_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++query_promotion_rejections_;
        }
        if (trace_enabled()) {
          std::cerr << "query_promotion_rejected frame="
                    << frame.meta.frame_id << " generation="
                    << query_generation << " scope="
                    << static_cast<int>(query_scope)
                    << " valid=" << (aggregate.valid ? 1 : 0)
                    << " all_known=" << (aggregate.all_known ? 1 : 0)
                    << " uncertainty_exceeded="
                    << (aggregate.uncertainty_exceeded ? 1 : 0)
                    << " promotion_uncertainty_m="
                    << aggregate.promotion_uncertainty_m
                    << " promotion_context_valid="
                    << (promotion_context_valid ? 1 : 0)
                    << " accepted_free=" << aggregate.accepted_free_cells
                    << " accepted_obstacle="
                    << aggregate.accepted_obstacle_cells
                    << " conflicts=" << aggregate.conflict_cells
                    << " duplicate=" << aggregate.duplicate_cells
                    << " unsupported=" << aggregate.unsupported_cells
                    << " safe_supports=" << aggregate.safe_supports
                    << " unsafe_supports=" << aggregate.unsafe_supports
                    << " unsafe_components=" << aggregate.unsafe_components
                    << " uncertain_veto_components="
                    << aggregate.uncertain_veto_components
                    << " uncertain_veto_cells="
                    << aggregate.uncertain_veto_cells
                    << " safe_promotion_uncertainty_m="
                    << aggregate.safe_promotion_uncertainty_m
                    << " uncertainty_domain_exceeded="
                    << (aggregate.uncertainty_domain_exceeded ? 1 : 0)
                    << " uncertainty_domain_cells="
                    << aggregate.uncertainty_domain_cells
                    << " uncertainty_radius_buckets="
                    << aggregate.uncertainty_radius_buckets
                    << " uncertainty_domain_work_cells="
                    << aggregate.uncertainty_domain_work_cells
                    << "\n";
        }
      }
    }
    if (verified_bridge_candidate_this_frame && !query_promotion &&
        query_frame_usable_for_promotion) {
      verified_bridge_committed_this_frame = commit_verified_graph_bridge(
          frame.meta.frame_id, staged_bridge_candidate_id,
          staged_candidate_to_current, staged_candidate_covariance,
          staged_fused_pose_before, "external_rgbd",
          staged_recovery_semantics, query_scope, query_generation);
      read_only_revisit = verified_bridge_committed_this_frame;
      read_only_revisit_loop = verified_bridge_committed_this_frame;
    }
    if (verified_bridge_committed_this_frame && query_release &&
        !query_promotion &&
        query_outcome_this_frame_ == br::QueryOutcome::kHolding) {
      // The A-C bridge and fused-pose rebase are already a complete, safe
      // transaction. If provisional coverage cannot pass its stricter gate,
      // discard it explicitly instead of deadlocking or weakening the gate.
      mark_query_generation_completed(
          query_scope, query_generation,
          br::QueryOutcome::kBridgeOnlyDiscarded);
      if (query_buffer_.scope == query_scope &&
          query_buffer_.generation == query_generation) {
        query_buffer_.clear();
      }
      query_outcome_this_frame_ = br::QueryOutcome::kBridgeOnlyDiscarded;
    }
    if (query_release && !query_promotion &&
        query_outcome_this_frame_ == br::QueryOutcome::kHolding &&
        query_scope == QueryScope::kNormal) {
      // A negative certificate is only a request to attempt promotion. Keep
      // the normal query read-only until the aggregate is committed (or proved
      // all-known); validation/resource/process failures are not a transition
      // back to ordinary mapping.
      soft_mapping_state_ = SoftMappingState::kUncertainHold;
      read_only_revisit_active_ = true;
    }
    const bool native_novelty_resume_guard =
        native_novelty_resume_guard_updates_ > 0U;
    const bool idle_skip =
        idle_hold_active_ && static_observation && !idle_motion_detected &&
        was_mapping && soft_mapping_state_ == SoftMappingState::kBuilding &&
        !read_only_revisit && !read_only_revisit_active_ &&
        !read_only_candidate_active_ && !query_buffer_.active() &&
        !recovery_mapping_hold && !query_promotion &&
        !external_metric_valid && !normal_global_search_pending_;
    const bool skip_slam =
        idle_skip || !slam_update_due || !process_observation ||
        (!query_promotion &&
          (read_only_revisit || recovery_mapping_hold ||
           suppress_query_release || orphan_query_hold ||
           normal_global_search_pending_ || conflicting_active_query ||
           query_terminal_replay || stale_query_generation ||
           native_novelty_resume_guard));
    if (skip_slam) {
      if (slam_update_due) {
        last_slam_stamp_s_ = frame.meta.stamp;
      }
      const rtabmap::Transform correction =
          slam_->getMapCorrection().isNull()
              ? rtabmap::Transform::getIdentity()
              : slam_->getMapCorrection().to3DoF();
      // A fresh anchor has already set the exact target pose. Continuation and
      // ordinary skipped frames use the current map correction and fused
      // qvel chain, which are in the same map frame.
      native_current_pose_ = (correction * fused_odom_pose_).to3DoF();
      current_pose_ = native_current_pose_;
      last_observation_novelty_ratio_ = observation_novelty_ratio(
          frame, native_current_pose_, &last_observation_endpoint_novelty_ratio_,
          &last_observation_ray_novelty_ratio_);
      note_candidate_novelty();
      if (was_mapping && !idle_skip &&
          soft_mapping_state_ == SoftMappingState::kBuilding &&
          !recovery_hold_this_frame_ && !native_novelty_resume_guard) {
        ++mapping_frames_;
        mapping_travel_m_ +=
            std::hypot(fused_increment.x(), fused_increment.y());
        mapping_rotation_rad_ += std::abs(fused_increment.theta());
        if (mapping_frames_ % kFreezeMetricPeriodFrames == 0) {
          convergence_ = update_convergence_evidence();
          if (trace_enabled()) {
            std::cerr << "mapping_convergence frame=" << frame.meta.frame_id
                      << " mapping_frames=" << mapping_frames_
                      << " nodes=" << poses_.size()
                      << " travel=" << mapping_travel_m_
                      << " known=" << convergence_.known_cells
                      << " boundary=" << convergence_.boundary_cells
                      << " frontier=" << convergence_.frontier_cells
                      << " frontier_ratio=" << convergence_.frontier_ratio
                      << " growth_ratio=" << convergence_.known_growth_ratio
                      << " visual_revisits="
                      << convergence_.recent_visual_revisits
                      << " accepted_loop_events="
                      << convergence_.accepted_loop_events
                      << " visual_loop_regions="
                      << convergence_.visual_loop_regions
                      << " max_loop_node_span_ratio="
                      << convergence_.max_loop_node_span_ratio
                      << " max_loop_motion_span_ratio="
                      << convergence_.max_loop_motion_span_ratio
                      << " graph_common_nodes="
                      << convergence_.graph_common_nodes
                      << " graph_window_translation="
                      << convergence_.graph_window_translation_m
                      << " graph_window_yaw="
                      << convergence_.graph_window_yaw_rad
                      << " loop_settle_frames="
                      << mapping_frames_ - last_visual_revisit_frame_
                      << " usable_streak=" << usable_observation_streak_
                      << " ready=" << (convergence_.ready ? 1 : 0)
                      << " read_only=" << (read_only_revisit ? 1 : 0)
                      << "\n";
          }
          consider_freeze_candidate(frame.meta.frame_id);
        }
      } else {
        note_localization_outcome(frame.meta.stamp, false);
      }
      if (native_novelty_resume_guard && slam_update_due &&
          process_observation && was_mapping && !query_promotion &&
          !read_only_revisit && !recovery_mapping_hold) {
        --native_novelty_resume_guard_updates_;
        if (trace_enabled()) {
          std::cerr << "native_novelty_resume_guard_consumed frame="
                    << frame.meta.frame_id << " remaining="
                    << native_novelty_resume_guard_updates_ << "\n";
        }
      }
      if (trace_enabled()) {
        std::cerr << "slam_skipped frame=" << frame.meta.frame_id
                  << " stamp=" << frame.meta.stamp
                  << " update_due=" << (slam_update_due ? 1 : 0)
                  << " structural_observation="
                  << (observation_usable ? 1 : 0)
                  << " registration_observation="
                  << (registration_observation ? 1 : 0)
                  << " mapping="
                  << (was_mapping ? 1 : 0)
                  << " read_only=" << (read_only_revisit ? 1 : 0)
                  << " recovery_hold="
                  << (recovery_hold_this_frame_ ? 1 : 0)
                  << " recovery_mapping_hold="
                  << (recovery_mapping_hold ? 1 : 0)
                  << " novelty_resume_guard="
                  << (native_novelty_resume_guard ? 1 : 0)
                  << " idle=" << (idle_skip ? 1 : 0) << "\n";
      }
      // A missing/unsuitable camera sample suppresses only this RTAB update.
      // qvel still propagates the pose, but an unsuitable frame must never
      // become the visual reference used after the head returns upright.
      if (observation_usable) {
        remember_observation(frame, data);
      } else {
        forget_observation_reference();
      }
      return snapshot(frame.meta.frame_id, true, false,
                      read_only_revisit_loop);
    }
    last_slam_stamp_s_ = frame.meta.stamp;
    // RTAB-Map 的标准 external-odometry 路径：qvel 传播 odom；原生
    // RGB-D 闭环或定位约束更新 map->odom。
    // In mature mapping, an external candidate that did not pass the
    // pre-process read-only gate is deliberately not forwarded to RTAB-Map:
    // forwarding it would still write a new signature before validation. New
    // areas and the early bootstrap phase retain ordinary native processing.
    const bool suppress_mature_external_candidate =
        was_mapping && external_metric_valid &&
        poses_.size() >= kReadOnlyRevisitMinMapNodes;
    const cv::Mat slam_covariance = pending_slam_covariance_.clone();
    if (!query_promotion) {
      pending_slam_covariance_.release();
    } else {
      // process() has no public rollback API. Mark this generation before the
      // only mutating call so an exception or partial upstream failure can
      // never replay the same aggregate. The covariance and buffer are cleared
      // only after a distinct new signature is observable.
      query_buffer_.promotion_attempted = true;
    }
    if (external_metric_valid && !suppress_mature_external_candidate &&
        !query_promotion) {
      slam_->setExternalLoopClosureHypothesis(
          frame.meta.external_loop_candidate_id,
          external_measured_candidate_to_current,
          external_measurement_covariance,
          kExternalLoopInnovationChiSquare);
      if (trace_enabled()) {
        std::cerr << "external_loop_metric frame=" << frame.meta.frame_id
                  << " candidate=" << frame.meta.external_loop_candidate_id
                  << " measured_x="
                  << external_measured_candidate_to_current.x()
                  << " measured_y="
                  << external_measured_candidate_to_current.y()
                  << " measured_yaw="
                  << external_measured_candidate_to_current.theta()
                  << "\n";
      }
    } else if (frame.meta.external_loop_candidate_id > 0 &&
               trace_enabled()) {
      std::cerr << "external_loop_metric_suppressed frame="
                << frame.meta.frame_id << " candidate="
                << frame.meta.external_loop_candidate_id
                << " mature_mapping="
                << (suppress_mature_external_candidate ? 1 : 0) << "\n";
    }
    if (query_promotion) {
      // Q and its HOLDING marker still exist only in local variables. Commit
      // the pre-query resident graph now, then treat every later exception as
      // fatal until the terminal Q transaction is durably acknowledged.
      query_integrity_failed_ = true;
      slam_->checkpointCommittedGraph();
    }
    const std::map<int, rtabmap::Transform> query_poses_before =
        query_promotion ? slam_->getLocalOptimizedPoses()
                        : std::map<int, rtabmap::Transform>();
    const std::multimap<int, rtabmap::Link> query_constraints_before =
        query_promotion ? slam_->getLocalConstraints()
                        : std::multimap<int, rtabmap::Link>();
    const std::map<int, rtabmap::Transform> query_odom_poses_before =
        query_promotion ? slam_->getStatistics().odomCachePoses()
                        : std::map<int, rtabmap::Transform>();
    const std::multimap<int, rtabmap::Link> query_odom_constraints_before =
        query_promotion ? slam_->getStatistics().odomCacheConstraints()
                        : std::multimap<int, rtabmap::Link>();
    cv::Mat query_known_map_before =
        query_promotion ? cached_map_.clone() : cv::Mat();
    float query_known_x_min_before = cached_x_min_;
    float query_known_y_min_before = cached_y_min_;
    const rtabmap::Transform query_correction_before =
        query_promotion
            ? (slam_->getMapCorrection().isNull()
                   ? rtabmap::Transform::getIdentity()
                   : slam_->getMapCorrection().to3DoF())
            : rtabmap::Transform();
    int query_signature_id = 0;
    bool upstream_processed = false;
    if (query_promotion) {
      // From this point until finalizeQuerySignature() returns, Q exists only
      // in RTAB-Map RAM. Any exception must terminate the process without
      // running close(), otherwise a destructor flush could persist a
      // partially verified node.
      query_integrity_failed_ = true;
      try {
        query_signature_id = slam_->processQuerySignature(
            data, promoted_query_odom_pose, promoted_query_covariance,
            promoted_query_anchor_id, promoted_query_neighbor_transform,
            promoted_query_map_pose);
      } catch (...) {
        // The dedicated upstream path has crossed its only mutating boundary.
        // Without a rollback API, any exception is fail-stop for this worker;
        // retrying could create a second Q node on top of a partial insertion.
        throw;
      }
      upstream_processed = query_signature_id > 0;
      if (upstream_processed && promoted_query_has_metric_bridge) {
        promoted_metric_information =
            planar_information(promoted_candidate_covariance);
        if (promoted_query_candidate_id <= 0 ||
            promoted_candidate_to_query.isNull() ||
            promoted_metric_information.empty()) {
          throw std::runtime_error(
              "positive query bridge has invalid metric evidence");
        }
        const rtabmap::Link metric_bridge(
            promoted_query_candidate_id, query_signature_id,
            rtabmap::Link::kGlobalClosure, promoted_candidate_to_query,
            promoted_metric_information);
        if (!slam_->addQueryLink(metric_bridge)) {
          throw std::runtime_error(
              "positive query C-Q graph consistency check failed");
        }
        rtabmap::Transform optimized_query =
            slam_->getPose(query_signature_id).to3DoF();
        if (!finite_planar_transform(optimized_query) ||
            !refresh_global_graph_poses(
                query_signature_id, optimized_query, query_signature_id,
                &optimized_query)) {
          throw std::runtime_error(
              "positive query bridge has no optimized global Q pose");
        }
        rebuild_display_poses();
        std::map<int, rtabmap::Transform> old_display_poses = display_poses_;
        old_display_poses.erase(query_signature_id);
        grid_->clear();
        grid_->update(old_display_poses);
        force_global_grid_rebuild_ = false;
        refresh_mapping_cache();
        query_known_map_before = cached_map_.clone();
        query_known_x_min_before = cached_x_min_;
        query_known_y_min_before = cached_y_min_;

        // C-Q optimization can move Q across global raster cells. The
        // provisional pre-filter above is only an admission check; repeat the
        // complete multi-view/global-lattice/old-known gate at the optimized
        // pose and replace the RAM-only payload before any local-grid cache or
        // database write sees it.
        QueryAggregate final_aggregate = build_query_aggregate(
            query_endpoint_odom_pose, optimized_query,
            promoted_candidate_covariance, promoted_candidate_to_query, true,
            query_endpoint_history_step);
        if (trace_enabled()) {
          std::cerr << "query_promotion_postopt frame=" << frame.meta.frame_id
                    << " generation=" << promoted_query_generation
                    << " valid=" << (final_aggregate.valid ? 1 : 0)
                    << " all_known=" << (final_aggregate.all_known ? 1 : 0)
                    << " free=" << final_aggregate.accepted_free_cells
                    << " obstacles="
                    << final_aggregate.accepted_obstacle_cells
                    << " conflicts=" << final_aggregate.conflict_cells
                    << " duplicate=" << final_aggregate.duplicate_cells
                    << " unsupported=" << final_aggregate.unsupported_cells
                    << " safe_supports=" << final_aggregate.safe_supports
                    << " unsafe_supports=" << final_aggregate.unsafe_supports
                    << " unsafe_components="
                    << final_aggregate.unsafe_components
                    << " uncertain_veto_components="
                    << final_aggregate.uncertain_veto_components
                    << " uncertain_veto_cells="
                    << final_aggregate.uncertain_veto_cells
                    << " promotion_uncertainty_m="
                    << final_aggregate.promotion_uncertainty_m
                    << " safe_promotion_uncertainty_m="
                    << final_aggregate.safe_promotion_uncertainty_m
                    << " uncertainty_domain_exceeded="
                    << (final_aggregate.uncertainty_domain_exceeded ? 1 : 0)
                    << " uncertainty_domain_cells="
                    << final_aggregate.uncertainty_domain_cells
                    << " uncertainty_radius_buckets="
                    << final_aggregate.uncertainty_radius_buckets
                    << " uncertainty_domain_work_cells="
                    << final_aggregate.uncertainty_domain_work_cells
                    << "\n";
        }
        if (!final_aggregate.valid || final_aggregate.all_known ||
            final_aggregate.conflict_cells != 0U) {
          throw std::runtime_error(
              "positive query failed post-optimization occupancy validation");
        }
        const cv::Mat holding_marker = encode_persisted_query(
            promoted_query_scope, promoted_query_generation,
            br::QueryOutcome::kHolding, promoted_query_kind,
            promoted_query_anchor_id, promoted_query_candidate_id, {});
        if (holding_marker.empty() ||
            !slam_->updateQuerySignaturePayload(
                query_signature_id, final_aggregate.ground,
                final_aggregate.obstacles, final_aggregate.empty,
                kGridCellM, cv::Point3f(0.0F, 0.0F, 0.0F),
                holding_marker)) {
          throw std::runtime_error(
              "positive query final occupancy payload replacement failed");
        }
        promoted_height_points = std::move(final_aggregate.height_points);

        const rtabmap::Transform correction =
            slam_->getMapCorrection().isNull()
                ? rtabmap::Transform::getIdentity()
                : slam_->getMapCorrection().to3DoF();
        apply_read_only_anchor(
            promoted_fused_pose_before, correction, optimized_query,
            promoted_candidate_covariance, promoted_query_candidate_id);
        localized_this_frame_ = true;
        visual_localized_this_frame_ = true;
        geometric_localized_this_frame_ = true;
        read_only_match_this_frame_ = true;
        read_only_revisit_active_ = true;
        soft_mapping_state_ = SoftMappingState::kKnownLocalizing;
        read_only_revisit_identity_verified_ = true;
        read_only_revisit_candidate_id_ = promoted_query_candidate_id;
        read_only_revisit_start_travel_m_ = session_motion_progress_m_;
        reset_soft_novelty_evidence();
        read_only_revisit_no_observation_streak_ = 0;
        read_only_scan_match_failures_ = 0;
        read_only_scan_match_next_stamp_s_ =
            -std::numeric_limits<double>::infinity();
        if (promoted_recovery_semantics) {
          recovery_interruption_active_ = false;
          recovery_unobserved_translation_m_ = 0.0F;
          recovery_unobserved_yaw_rad_ = 0.0F;
        }
        normal_global_probe_complete_ = false;
        if (read_only_revisit_count_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++read_only_revisit_count_;
        }
        std::uint32_t &bridge_count =
            promoted_recovery_semantics
                ? recovery_graph_bridge_count_
                : verified_graph_bridge_count_;
        if (bridge_count < std::numeric_limits<std::uint32_t>::max()) {
          ++bridge_count;
        }
        const int left =
            std::min(promoted_query_candidate_id, query_signature_id);
        const int right =
            std::max(promoted_query_candidate_id, query_signature_id);
        seen_loop_constraints_.insert(
            {left, right,
             static_cast<int>(rtabmap::Link::kGlobalClosure)});
        if (loop_count_ < std::numeric_limits<std::uint32_t>::max()) {
          ++loop_count_;
        }
        verified_bridge_committed_this_frame = true;
        read_only_revisit = true;
        read_only_revisit_loop = true;
      }
      if (upstream_processed && !promoted_query_has_metric_bridge) {
        const rtabmap::Transform optimized_query =
            slam_->getPose(query_signature_id).to3DoF();
        const rtabmap::Transform correction =
            slam_->getMapCorrection().isNull()
                ? rtabmap::Transform::getIdentity()
                : slam_->getMapCorrection().to3DoF();
        if (!finite_planar_transform(optimized_query) ||
            !finite_planar_transform(correction)) {
          throw std::runtime_error(
              "negative query has no consistent optimized pose");
        }
        // Continue the live fused chain from Q's canonical raw odometry pose.
        // The graph A-Q factor remains the separately accumulated base_qvel
        // measurement; this rebase only aligns subsequent online increments.
        apply_read_only_anchor(
            promoted_fused_pose_before, correction, optimized_query,
            promoted_query_covariance, query_signature_id);
      }
    } else {
      if (native_novelty_resume_reconciliation_pending_ ||
          !native_novelty_resume_protected_free_map_.empty() ||
          !native_novelty_resume_snapshot_map_.empty()) {
        apply_novelty_resume_occupancy_filter(
            data, frame.depth_scan, predicted_map_pose, frame.meta.frame_id);
      }
      if (!frame.occupancy_height_points.empty()) {
        const cv::Mat height_marker =
            encode_persisted_height(frame.occupancy_height_points);
        if (height_marker.empty()) {
          throw std::runtime_error(
              "ordinary mapping height payload is invalid or too large");
        }
        data.setUserData(height_marker);
      }
      upstream_processed =
          slam_->process(data, fused_odom_pose_, slam_covariance);
    }
    bool query_signature_committed = false;
    if (query_promotion && upstream_processed) {
      const auto &query_poses_after = slam_->getLocalOptimizedPoses();
      const auto &query_constraints_after = slam_->getLocalConstraints();
      const auto &query_odom_poses_after =
          slam_->getStatistics().odomCachePoses();
      const auto &query_odom_constraints_after =
          slam_->getStatistics().odomCacheConstraints();
      const rtabmap::Signature inserted = slam_->getSignatureCopy(
          query_signature_id, true, true, true, true, false, false);
      const rtabmap::Transform expected_query_transform =
          promoted_query_neighbor_transform.to3DoF();
      const cv::Mat expected_query_information =
          promoted_query_covariance.empty()
              ? cv::Mat()
              : promoted_query_covariance.inv();
      std::size_t incident_constraints = 0;
      bool exact_neighbor = false;
      bool exact_metric_bridge = false;
      std::multimap<int, rtabmap::Link> old_constraints_after;
      for (const auto &entry : query_constraints_after) {
        const rtabmap::Link &link = entry.second;
        if (link.from() == query_signature_id ||
            link.to() == query_signature_id) {
          ++incident_constraints;
          exact_neighbor = exact_neighbor || query_link_matches(
              link, promoted_query_anchor_id, query_signature_id,
              expected_query_transform, expected_query_information);
          exact_metric_bridge = exact_metric_bridge ||
              (promoted_query_has_metric_bridge &&
               query_link_matches(
                   link,
                   promoted_query_candidate_id, query_signature_id,
                   promoted_candidate_to_query,
                   promoted_metric_information,
                   rtabmap::Link::kGlobalClosure));
        } else {
          old_constraints_after.insert(entry);
        }
      }
      bool old_poses_valid =
          query_poses_after.size() == query_poses_before.size() + 1U;
      for (const auto &entry : query_poses_before) {
        const auto after = query_poses_after.find(entry.first);
        if (after == query_poses_after.end()) {
          old_poses_valid = false;
          break;
        }
        const rtabmap::Transform delta =
            (entry.second.inverse() * after->second).to3DoF();
        if (!finite_planar_transform(delta) ||
            (!promoted_query_has_metric_bridge &&
             (std::hypot(delta.x(), delta.y()) > 1.0e-5F ||
              std::abs(wrap_angle(delta.theta())) > 1.0e-5F))) {
          old_poses_valid = false;
          break;
        }
      }
      bool old_odom_poses_unchanged =
          query_odom_poses_after.size() ==
          query_odom_poses_before.size() + 1U;
      for (const auto &entry : query_odom_poses_before) {
        const auto after = query_odom_poses_after.find(entry.first);
        if (after == query_odom_poses_after.end()) {
          old_odom_poses_unchanged = false;
          break;
        }
        const rtabmap::Transform delta =
            (entry.second.inverse() * after->second).to3DoF();
        if (!finite_planar_transform(delta) ||
            std::hypot(delta.x(), delta.y()) > 1.0e-6F ||
            std::abs(wrap_angle(delta.theta())) > 1.0e-6F) {
          old_odom_poses_unchanged = false;
          break;
        }
      }
      std::size_t incident_odom_constraints = 0;
      bool exact_odom_neighbor = false;
      std::multimap<int, rtabmap::Link> old_odom_constraints_after;
      for (const auto &entry : query_odom_constraints_after) {
        const rtabmap::Link &link = entry.second;
        if (link.from() == query_signature_id ||
            link.to() == query_signature_id) {
          ++incident_odom_constraints;
          exact_odom_neighbor = exact_odom_neighbor || query_link_matches(
              link, promoted_query_anchor_id, query_signature_id,
              expected_query_transform, expected_query_information);
        } else {
          old_odom_constraints_after.insert(entry);
        }
      }
      const bool odom_cache_exact =
          old_odom_poses_unchanged &&
          query_odom_poses_after.find(query_signature_id) !=
              query_odom_poses_after.end() &&
          query_constraints_equal(query_odom_constraints_before,
                                  old_odom_constraints_after) &&
          incident_odom_constraints == 1U && exact_odom_neighbor;
      const auto inserted_pose = query_poses_after.find(query_signature_id);
      const rtabmap::Transform inserted_pose_error =
          inserted_pose == query_poses_after.end()
              ? rtabmap::Transform()
              : (promoted_query_map_pose.inverse() *
                 inserted_pose->second).to3DoF();
      const auto inserted_odom_pose =
          query_odom_poses_after.find(query_signature_id);
      const rtabmap::Transform signature_odom_error =
          inserted.id() != query_signature_id
              ? rtabmap::Transform()
              : (promoted_query_odom_pose.inverse() *
                 inserted.getPose().to3DoF()).to3DoF();
      const rtabmap::Transform cache_odom_error =
          inserted_odom_pose == query_odom_poses_after.end()
              ? rtabmap::Transform()
              : (promoted_query_odom_pose.inverse() *
                 inserted_odom_pose->second.to3DoF()).to3DoF();
      const rtabmap::Transform correction_after =
          slam_->getMapCorrection().isNull()
              ? rtabmap::Transform::getIdentity()
              : slam_->getMapCorrection().to3DoF();
      const rtabmap::Transform live_pose_error =
          inserted_pose == query_poses_after.end()
              ? rtabmap::Transform()
              : (inserted_pose->second.inverse() *
                 (correction_after * fused_odom_pose_).to3DoF()).to3DoF();
      QueryScope persisted_scope = QueryScope::kNone;
      std::uint64_t persisted_generation = 0;
      br::QueryOutcome persisted_outcome = br::QueryOutcome::kNone;
      std::vector<HeightPoint> persisted_height;
      QueryPromotionKind persisted_kind = QueryPromotionKind::kNone;
      int persisted_anchor_id = 0;
      int persisted_candidate_id = 0;
      cv::Mat persisted_image;
      cv::Mat persisted_depth;
      rtabmap::LaserScan persisted_scan;
      cv::Mat persisted_user_data;
      cv::Mat persisted_ground;
      cv::Mat persisted_obstacles;
      cv::Mat persisted_empty;
      inserted.sensorData().uncompressDataConst(
          &persisted_image, &persisted_depth, &persisted_scan,
          &persisted_user_data, &persisted_ground, &persisted_obstacles,
          &persisted_empty);
      const bool payload_valid = inserted.id() == query_signature_id &&
          inserted.sensorData().gridCellSize() > 0.0F &&
          !persisted_scan.empty() && !persisted_image.empty() &&
          !persisted_depth.empty() &&
          query_layer_is_valid(persisted_ground) &&
          query_layer_is_valid(persisted_obstacles) &&
          query_layer_is_valid(persisted_empty) &&
          decode_persisted_query(
              inserted.sensorData(), &persisted_scope,
              &persisted_generation, &persisted_outcome,
              &persisted_height, &persisted_kind, &persisted_anchor_id,
              &persisted_candidate_id) &&
          persisted_scope == promoted_query_scope &&
          persisted_generation == promoted_query_generation &&
          persisted_outcome == br::QueryOutcome::kHolding &&
          persisted_kind == promoted_query_kind &&
          persisted_anchor_id == promoted_query_anchor_id &&
          persisted_candidate_id == promoted_query_candidate_id &&
          persisted_height.empty();
      const bool expected_metric_topology =
          promoted_query_has_metric_bridge
              ? incident_constraints == 2U && exact_metric_bridge
              : incident_constraints == 1U;
      query_signature_committed =
          old_poses_valid &&
          odom_cache_exact &&
          query_constraints_after.size() ==
              query_constraints_before.size() +
                  (promoted_query_has_metric_bridge ? 2U : 1U) &&
          query_constraints_equal(query_constraints_before,
                                  old_constraints_after) &&
          expected_metric_topology && exact_neighbor &&
          finite_planar_transform(signature_odom_error) &&
          std::hypot(signature_odom_error.x(), signature_odom_error.y()) <=
              1.0e-6F &&
          std::abs(wrap_angle(signature_odom_error.theta())) <= 1.0e-6F &&
          finite_planar_transform(cache_odom_error) &&
          std::hypot(cache_odom_error.x(), cache_odom_error.y()) <= 1.0e-6F &&
          std::abs(wrap_angle(cache_odom_error.theta())) <= 1.0e-6F &&
          finite_planar_transform(live_pose_error) &&
          std::hypot(live_pose_error.x(), live_pose_error.y()) <= 1.0e-5F &&
          std::abs(wrap_angle(live_pose_error.theta())) <= 1.0e-5F &&
          finite_planar_transform(inserted_pose_error) &&
          (promoted_query_has_metric_bridge ||
           (std::hypot(inserted_pose_error.x(), inserted_pose_error.y()) <=
                1.0e-5F &&
            std::abs(wrap_angle(inserted_pose_error.theta())) <= 1.0e-5F)) &&
          payload_valid;
      if (!query_signature_committed) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "query signature crossed insertion boundary without exact A-Q invariant");
      }
    } else if (query_promotion) {
      const auto &query_poses_after = slam_->getLocalOptimizedPoses();
      const auto &query_constraints_after = slam_->getLocalConstraints();
      bool poses_unchanged =
          query_poses_after.size() == query_poses_before.size();
      for (const auto &entry : query_poses_before) {
        const auto after = query_poses_after.find(entry.first);
        if (after == query_poses_after.end()) {
          poses_unchanged = false;
          break;
        }
        const rtabmap::Transform delta =
            (entry.second.inverse() * after->second).to3DoF();
        if (!finite_planar_transform(delta) ||
            std::hypot(delta.x(), delta.y()) > 1.0e-6F ||
            std::abs(wrap_angle(delta.theta())) > 1.0e-6F) {
          poses_unchanged = false;
          break;
        }
      }
      const rtabmap::Transform correction_after =
          slam_->getMapCorrection().isNull()
              ? rtabmap::Transform::getIdentity()
              : slam_->getMapCorrection().to3DoF();
      const rtabmap::Transform correction_error =
          (query_correction_before.inverse() * correction_after).to3DoF();
      const bool clean_failure =
          poses_unchanged &&
          query_constraints_equal(query_constraints_before,
                                  query_constraints_after) &&
          finite_planar_transform(correction_error) &&
          std::hypot(correction_error.x(), correction_error.y()) <= 1.0e-6F &&
          std::abs(wrap_angle(correction_error.theta())) <= 1.0e-6F;
      if (!clean_failure) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "query signature failed after mutating the canonical graph prefix");
      }
      query_buffer_.promotion_attempted = false;
      if (verified_bridge_committed_this_frame) {
        mark_query_generation_completed(
            promoted_query_scope, promoted_query_generation,
            br::QueryOutcome::kBridgeOnlyDiscarded);
        query_buffer_.clear();
        query_outcome_this_frame_ =
            br::QueryOutcome::kBridgeOnlyDiscarded;
      }
    }
    const bool processed =
        query_promotion ? query_signature_committed : upstream_processed;
    if (query_promotion) {
      if (query_signature_committed) {
        // Register the per-node footprint exception before assembling its
        // pre-filtered local grid. The terminal ACK is deliberately delayed
        // until the raster-level immutable-old-map check below succeeds.
        query_footprint_ignored_ids_.insert(query_signature_id);
        grid_->setFootprintIgnoredIds(query_footprint_ignored_ids_);
      } else {
        if (query_outcome_this_frame_ == br::QueryOutcome::kHolding &&
            promoted_query_scope == QueryScope::kNormal) {
          soft_mapping_state_ = SoftMappingState::kUncertainHold;
          read_only_revisit_active_ = true;
        }
        if (query_process_failures_ <
            std::numeric_limits<std::uint32_t>::max()) {
          ++query_process_failures_;
        }
        if (trace_enabled()) {
          std::cerr << "query_process_failed frame=" << frame.meta.frame_id
                    << " generation=" << promoted_query_generation
                    << " scope=" << static_cast<int>(promoted_query_scope)
                    << " upstream_processed="
                    << (upstream_processed ? 1 : 0)
                    << " signature_id=" << query_signature_id
                    << "\n";
        }
      }
    }
    const rtabmap::Statistics &statistics = slam_->getStatistics();
    // process() can return true and keep the new id in refImageId() even when
    // RTAB-Map's displacement gate has already deleted that signature. Such a
    // node is saved with weight=-9 and is absent after reopen. Never publish
    // its transient pose/grid as permanent map state.
    const int accepted_signature_id = statistics.refImageId();
    rtabmap::Signature accepted_signature;
    if (was_mapping && processed && accepted_signature_id > 0) {
      accepted_signature = slam_->getSignatureCopy(
          accepted_signature_id, false, false, false, true,
          trace_enabled(), false);
    }
    const bool ordinary_signature_retained =
        query_promotion ||
        (accepted_signature_id > 0 &&
         accepted_signature.id() == accepted_signature_id &&
         accepted_signature.getWeight() >= 0 &&
         statistics.poses().find(accepted_signature_id) !=
             statistics.poses().end());
    const bool map_updated =
        was_mapping && processed && ordinary_signature_retained;
    last_ref_node_id_ =
        was_mapping && processed && !ordinary_signature_retained
            ? last_mapping_node_id_
            : accepted_signature_id;
    if (was_mapping && processed && !query_promotion &&
        !ordinary_signature_retained && trace_enabled()) {
      std::cerr << "native_signature_discarded frame=" << frame.meta.frame_id
                << " ref=" << accepted_signature_id
                << " signature_id=" << accepted_signature.id()
                << " weight=" << accepted_signature.getWeight()
                << " optimized_pose="
                << (statistics.poses().find(accepted_signature_id) !=
                            statistics.poses().end()
                        ? 1
                        : 0)
                << "\n";
    }
    const int closure_id = statistics.loopClosureId() > 0
                               ? statistics.loopClosureId()
                               : statistics.proximityDetectionId();
    const float visual_inliers = statistic_value(
        statistics, rtabmap::Statistics::kLoopVisual_inliers());
    const float visual_inlier_ratio = statistic_value(
        statistics, rtabmap::Statistics::kLoopVisual_inliers_ratio());
    const float visual_inlier_distribution = statistic_value(
        statistics, rtabmap::Statistics::kLoopVisual_inliers_distribution());
    const float visual_matches = statistic_value(
        statistics, rtabmap::Statistics::kLoopVisual_matches());
    const float proximity_visual = statistic_value(
        statistics,
        rtabmap::Statistics::kProximitySpace_detections_added_visually());
    const float proximity_icp_multi = statistic_value(
        statistics,
        rtabmap::Statistics::kProximitySpace_detections_added_icp_multi());
    const float proximity_icp_global = statistic_value(
        statistics,
        rtabmap::Statistics::kProximitySpace_detections_added_icp_global());
    const float highest_hypothesis_id = statistic_value(
        statistics, rtabmap::Statistics::kLoopHighest_hypothesis_id());
    const float highest_hypothesis_value = statistic_value(
        statistics, rtabmap::Statistics::kLoopHighest_hypothesis_value());
    const float accepted_hypothesis_id = statistic_value(
        statistics, rtabmap::Statistics::kLoopAccepted_hypothesis_id());
    const float rejected_hypothesis = statistic_value(
        statistics, rtabmap::Statistics::kLoopRejectedHypothesis());
    const float optimization_error_ratio = statistic_value(
        statistics, rtabmap::Statistics::kLoopOptimization_max_error_ratio());
    const float optimization_angle_error_ratio = statistic_value(
        statistics,
        rtabmap::Statistics::kLoopOptimization_max_ang_error_ratio());
    // 有效的全局闭环和空间近邻边都必须带原生米制几何约束。official
    // 映射用 depth-scan ICP 验证外观/空间候选；实验 profile 也可能使用
    // RGB-D 特征或 VisIcp，但 Link 的非空变换是共同的提交条件。
    // rehearsal 合并也使用 GlobalClosure 类型，但其变换为空，必须排除。
    // LocalSpaceClosure 有时只写进当前 Signature 的 links，而不会出现在
    // Statistics::constraints()；两条原生发布路径必须合并，否则闭环后的
    // 全局图刷新和冻结证据会漏掉这类合法重访。
    std::vector<std::pair<int, int>> new_metric_loop_pairs;
    bool accepted_metric_link = false;
    bool accepted_visual_link = false;
    int accepted_metric_peer = 0;
    const auto append_metric_link = [&](const rtabmap::Link &link) {
      if (!processed || !link.isValid() ||
          (link.type() != rtabmap::Link::kGlobalClosure &&
           link.type() != rtabmap::Link::kLocalSpaceClosure) ||
          link.from() <= 0 || link.to() <= 0 || link.from() == link.to() ||
          link.transform().isNull()) {
        return;
      }
      const int left = std::min(link.from(), link.to());
      const int right = std::max(link.from(), link.to());
      const std::array<int, 3> key{
          left, right, static_cast<int>(link.type())};
      if (was_mapping && seen_loop_constraints_.insert(key).second) {
        new_metric_loop_pairs.push_back({left, right});
      }
      if (link.from() == statistics.refImageId() ||
          link.to() == statistics.refImageId()) {
        accepted_metric_link = true;
        accepted_metric_peer = link.from() == statistics.refImageId()
                                   ? link.to()
                                   : link.from();
        accepted_visual_link = accepted_visual_link ||
                               statistic_value(
                                   statistics,
                                   rtabmap::Statistics::kLoopVisual_inliers()) >=
                                   15.0F;
      }
    };
    for (const auto &entry : statistics.constraints()) {
      append_metric_link(entry.second);
    }
    const auto signature_data = statistics.getSignaturesData().find(
        statistics.refImageId());
    if (processed && signature_data != statistics.getSignaturesData().end()) {
      for (const auto &entry : signature_data->second.getLinks()) {
        append_metric_link(entry.second);
      }
    }
    for (std::size_t index = 0;
         was_mapping && index < new_metric_loop_pairs.size() &&
                                loop_count_ <
                                    std::numeric_limits<std::uint32_t>::max();
         ++index) {
      ++loop_count_;
    }
    const bool rtab_visual_registration =
        processed && visual_inliers >= 15.0F &&
        (accepted_visual_link || statistics.loopClosureId() > 0 ||
         (statistics.proximityDetectionId() > 0 && proximity_visual > 0.0F));
    const bool rtab_geometric_registration =
        processed &&
        (accepted_metric_link ||
         (closure_id > 0 &&
          (!statistics.loopClosureTransform().isNull() ||
           !new_metric_loop_pairs.empty() || proximity_icp_multi > 0.0F ||
           proximity_icp_global > 0.0F)));
    // A proximity path can be accepted by RTAB-Map's native multi-scan ICP
    // even when that path has no BoW inlier count.  It is still a genuine
    // metric revisit: the link is valid, touches the current signature, and
    // RTAB-Map reports that ICP added it.  Treat that path as revisit evidence
    // for the freeze gate, but never as visual registration for the public
    // localization flags.  This keeps the evidence source native while
    // avoiding the previous false negative where every ICP-only revisit was
    // printed as a loop yet could never start a freeze candidate.
    const bool rtab_native_revisit =
        processed && rtab_geometric_registration &&
        (rtab_visual_registration ||
         (accepted_metric_link &&
          (proximity_icp_multi > 0.0F || proximity_icp_global > 0.0F)));
    // A constraint can already be visible in the current Signature and in
    // Statistics::constraints() while RTAB-Map is deliberately delaying the
    // first localization in its odometry cache.  In that case upstream clears
    // loopClosureId()/proximityDetectionId() and publishes
    // Loop/Rejected_hypothesis=1.  Treating the temporary link as success
    // exits read-only recovery without applying map->odom correction, then the
    // drifted pose immediately starts a duplicate map component.  Localization
    // succeeds only after RTAB-Map formally commits the hypothesis.
    const bool localization_hypothesis_committed =
        processed && closure_id > 0 && rejected_hypothesis <= 0.0F;
    const bool metric_localization_evidence =
        localization_hypothesis_committed &&
        (accepted_metric_link || rtab_geometric_registration);
    const bool visual_localization_evidence =
        localization_hypothesis_committed &&
        visual_inliers >= static_cast<float>(kVisualLoopMinInliers) &&
        (accepted_visual_link || rtab_visual_registration);
    const rtabmap::Transform map_correction_after =
        slam_->getMapCorrection().isNull()
            ? rtabmap::Transform::getIdentity()
            : slam_->getMapCorrection().to3DoF();
    if (trace_enabled()) {
      std::cerr << "stats frame=" << frame.meta.frame_id
                << " ref=" << statistics.refImageId()
                << " loop=" << statistics.loopClosureId()
                << " proximity=" << statistics.proximityDetectionId()
                << " constraints=" << statistics.constraints().size()
                << " visual_inliers=" << visual_inliers
                << " visual_matches=" << visual_matches
                << " visual_ratio=" << visual_inlier_ratio
                << " visual_distribution=" << visual_inlier_distribution
                << " highest_id=" << highest_hypothesis_id
                << " highest_value=" << highest_hypothesis_value
                << " accepted_id=" << accepted_hypothesis_id
                << " rejected=" << rejected_hypothesis
                << " opt_error_ratio=" << optimization_error_ratio
                << " opt_angle_ratio=" << optimization_angle_error_ratio
                << " proximity_visual=" << proximity_visual
                << " proximity_icp_multi=" << proximity_icp_multi
                << " proximity_icp_global=" << proximity_icp_global
                << " accepted_metric_link=" << (accepted_metric_link ? 1 : 0)
                << " accepted_visual_link=" << (accepted_visual_link ? 1 : 0)
                << " accepted_metric_peer=" << accepted_metric_peer
                << " new_metric_loops=" << new_metric_loop_pairs.size()
                << " localization_committed="
                << (localization_hypothesis_committed ? 1 : 0)
                << " unique_metric_loops=" << loop_count_ << "\n";
      if (was_mapping &&
          (!new_metric_loop_pairs.empty() || closure_id > 0)) {
        std::map<int, rtabmap::Transform> global_graph_poses;
        std::multimap<int, rtabmap::Link> global_graph_constraints;
        slam_->getGraph(global_graph_poses, global_graph_constraints, true, true);
        std::map<int, rtabmap::Transform> raw_global_graph_poses;
        std::multimap<int, rtabmap::Link> raw_global_graph_constraints;
        slam_->getGraph(raw_global_graph_poses, raw_global_graph_constraints,
                        false, true);
        const auto raw_current = raw_global_graph_poses.find(statistics.refImageId());
        const auto optimized_current = global_graph_poses.find(statistics.refImageId());
        std::cerr << "graph_probe frame=" << frame.meta.frame_id
                  << " statistics_poses=" << statistics.poses().size()
                  << " local_optimized=" << slam_->getLocalOptimizedPoses().size()
                  << " global_optimized=" << global_graph_poses.size()
                  << " raw_global=" << raw_global_graph_poses.size()
                  << " raw_current=" << (raw_current == raw_global_graph_poses.end() ? 0 : 1)
                  << " optimized_current="
                  << (optimized_current == global_graph_poses.end() ? 0 : 1)
                  << " loop_transform_null="
                  << (statistics.loopClosureTransform().isNull() ? 1 : 0)
                  << "\n";
        if (raw_current != raw_global_graph_poses.end()) {
          const auto corrected = (map_correction_after * raw_current->second).to3DoF();
          std::cerr << "graph_pose_probe frame=" << frame.meta.frame_id
                    << " node=" << statistics.refImageId()
                    << " raw_x=" << raw_current->second.x()
                    << " raw_y=" << raw_current->second.y()
                    << " raw_yaw=" << raw_current->second.theta()
                    << " corrected_x=" << corrected.x()
                    << " corrected_y=" << corrected.y()
                    << " corrected_yaw=" << corrected.theta()
                    << " optimized_x="
                    << (optimized_current == global_graph_poses.end()
                            ? 0.0F
                            : optimized_current->second.x())
                    << " optimized_y="
                    << (optimized_current == global_graph_poses.end()
                            ? 0.0F
                            : optimized_current->second.y())
                    << " optimized_yaw="
                    << (optimized_current == global_graph_poses.end()
                            ? 0.0F
                            : optimized_current->second.theta())
                    << "\n";
        }
      }
    }
    last_inliers_ = processed
                        ? static_cast<std::uint32_t>(
                              std::max(0.0F, visual_inliers))
                        : 0U;
    last_features_ = processed
                         ? static_cast<std::uint32_t>(
                               std::max(0.0F, visual_matches))
                         : 0U;
    if (!verified_bridge_committed_this_frame) {
      visual_localized_this_frame_ =
          !was_mapping && visual_localization_evidence;
      geometric_localized_this_frame_ =
          !was_mapping && metric_localization_evidence;
      localized_this_frame_ = visual_localized_this_frame_ &&
                              geometric_localized_this_frame_;
    }
    if (!was_mapping) {
      note_localization_outcome(frame.meta.stamp,
                                metric_localization_evidence);
    }
    const bool loop_closed =
        was_mapping
            ? verified_bridge_committed_this_frame ||
                  !new_metric_loop_pairs.empty()
            : localized_this_frame_;
    const rtabmap::Transform correction_delta =
        (map_correction_before.inverse() * map_correction_after).to3DoF();
    if (trace_enabled() &&
        (std::hypot(correction_delta.x(), correction_delta.y()) > 0.01F ||
         std::abs(correction_delta.theta()) > 0.01F || closure_id > 0)) {
      std::cerr << "map_correction frame=" << frame.meta.frame_id
                << " before_x=" << map_correction_before.x()
                << " before_y=" << map_correction_before.y()
                << " before_yaw=" << map_correction_before.theta()
                << " after_x=" << map_correction_after.x()
                << " after_y=" << map_correction_after.y()
                << " after_yaw=" << map_correction_after.theta()
                << " delta_x=" << correction_delta.x()
                << " delta_y=" << correction_delta.y()
                << " delta_yaw=" << correction_delta.theta()
                << " closure_id=" << closure_id
                << " new_metric_loops=" << new_metric_loop_pairs.size()
                << "\n";
    }
    native_current_pose_ =
        (map_correction_after * fused_odom_pose_).to3DoF();
    if (was_mapping) {
      const bool global_graph_changed =
          !new_metric_loop_pairs.empty() || promoted_query_has_metric_bridge;
      // Refresh before merging Statistics::poses().  That local table already
      // contains the post-optimization coordinates and may include our oldest
      // published node; merging it first would overwrite the very map-frame
      // datum used by refresh_global_graph_poses() to remove RTAB-Map's
      // arbitrary global gauge change.
      if (global_graph_changed) {
        const int current_graph_id =
            statistics.refImageId() > 0 ? statistics.refImageId()
                                        : accepted_signature_id;
        rtabmap::Transform refreshed_current_pose;
        if (current_graph_id <= 0 ||
            !refresh_global_graph_poses(
                current_graph_id, native_current_pose_, current_graph_id,
                &refreshed_current_pose)) {
          throw std::runtime_error(
              "RTAB-Map graph changed without a gauge-aligned current pose");
        }
        native_current_pose_ = refreshed_current_pose;
      }

      // getLocalOptimizedPoses() is deliberately a *local* graph.  RTAB-Map
      // removes signatures that have moved to historical memory (and
      // intermediate/rehearsed nodes) from this container.  Replacing the
      // persistent display table with it makes old scans disappear from the
      // next GlobalMap::update() call; the remaining scans are then assembled
      // in a mixture of old and new graph subsets and appear as duplicate
      // walls.  Merge local updates into the persistent table instead.
      if (!global_graph_changed) {
        for (const auto &entry : statistics.poses()) {
          if (!query_promotion && !ordinary_signature_retained &&
              entry.first == accepted_signature_id) {
            continue;
          }
          if (entry.first > 0 && !entry.second.isNull()) {
            poses_[entry.first] = entry.second.to3DoF();
          }
        }
      }

      rebuild_display_poses();
    } else if (!slam_->getLastLocalizationPose().isNull()) {
      native_current_pose_ = slam_->getLastLocalizationPose().to3DoF();
    }
    // Evaluate novelty against the map before this frame is inserted. In
    // localization mode use RTAB-Map's final corrected pose, not the incoming
    // qvel prediction. A frame entering a new room must invalidate a freeze
    // candidate even when the occupancy cache has not yet grown.
    last_observation_novelty_ratio_ = observation_novelty_ratio(
        frame, native_current_pose_, &last_observation_endpoint_novelty_ratio_,
        &last_observation_ray_novelty_ratio_);
    note_candidate_novelty();
    // Statistics::getLastSignatureData() is an optional publication stream and
    // can legally lag the reference id (with PublishLastSignatureData disabled
    // it stayed at node 1 for the first part of this capture).  The reference
    // id is the authoritative id of the observation accepted by process().
    if (was_mapping && map_updated && accepted_signature_id > 0) {
      last_mapping_node_id_ = accepted_signature_id;
      node_odometry_history_index_.emplace(
          accepted_signature_id, odometry_history_.size());
    }
    current_pose_ = native_current_pose_;
    // RTAB-Map assigns graph node ids independently from evaluator frame ids.
    // They diverge as soon as odometry rejects a frame, so key all movable
    // geometry by the signature actually accepted into the graph.
    if (was_mapping && map_updated && accepted_signature_id > 0 &&
        accepted_signature.id() > 0 &&
        local_grids_.find(accepted_signature_id) == local_grids_.end() &&
        accepted_signature.sensorData().gridCellSize() > 0.0F) {
      cv::Mat ground;
      cv::Mat obstacles;
      cv::Mat empty;
      accepted_signature.sensorData().uncompressDataConst(
          nullptr, nullptr, nullptr, nullptr, &ground, &obstacles, &empty);
      if (trace_enabled() &&
          (local_grids_.empty() || accepted_signature_id <= 3 ||
           accepted_signature_id % 50 == 0 || !new_metric_loop_pairs.empty())) {
        const auto pose_iter = poses_.find(accepted_signature_id);
        const rtabmap::Transform optimized_pose =
            pose_iter != poses_.end() && !pose_iter->second.isNull()
                ? pose_iter->second.to3DoF()
                : rtabmap::Transform();
        const auto scan = accepted_signature.sensorData().laserScanRaw();
        std::cerr << "native_grid_cells frame=" << frame.meta.frame_id
                  << " node=" << accepted_signature_id
                  << " scan_format=" << scan.format()
                  << " scan_is2d=" << (scan.is2d() ? 1 : 0)
                  << " ground=" << ground.cols
                  << " obstacle=" << obstacles.cols
                  << " empty=" << empty.cols
                  << " ground_type=" << ground.type()
                  << " obstacle_type=" << obstacles.type()
                  << " empty_type=" << empty.type()
                  << " cell=" << accepted_signature.sensorData().gridCellSize()
                  << " viewpoint_x="
                  << accepted_signature.sensorData().gridViewPoint().x
                  << " viewpoint_y="
                  << accepted_signature.sensorData().gridViewPoint().y
                  << " viewpoint_z="
                  << accepted_signature.sensorData().gridViewPoint().z
                  << " signature_pose_x=" << accepted_signature.getPose().x()
                  << " signature_pose_y=" << accepted_signature.getPose().y()
                  << " signature_pose_yaw=" << accepted_signature.getPose().theta()
                  << " optimized_pose_x=" << optimized_pose.x()
                  << " optimized_pose_y=" << optimized_pose.y()
                  << " optimized_pose_yaw=" << optimized_pose.theta()
                  << " words=" << accepted_signature.getWords().size()
                  << " descriptors="
                  << accepted_signature.getWordsDescriptors().rows
                  << " cache_size_before=" << local_grids_.size()
                  << " added_nodes=" << grid_->addedNodes().size() << "\n";
      }
      local_grids_.add(accepted_signature_id, ground, obstacles, empty,
                       accepted_signature.sensorData().gridCellSize(),
                       accepted_signature.sensorData().gridViewPoint());
      height_points_[accepted_signature_id] = std::move(frame.height_points);
      occupancy_height_points_[accepted_signature_id] =
          query_promotion ? std::move(promoted_height_points)
                          : std::move(frame.occupancy_height_points);
      if (!query_promotion && !frame.recovered_height_points.empty()) {
        recovered_height_points_[accepted_signature_id] =
            std::move(frame.recovered_height_points);
      }
    }
    if (was_mapping && map_updated && !display_poses_.empty() &&
        (!local_grids_.empty() || !grid_->addedNodes().empty())) {
      if (force_global_grid_rebuild_) {
        // GlobalMap::update() intentionally keeps its assembled-node cache and
        // only notices a moved node when it is still present in the incoming
        // pose set.  A full global graph refresh can also remove an old local
        // node, so explicitly clear the native assembly before feeding the
        // authoritative pose table.  LocalGridCache remains untouched.
        grid_->clear();
        force_global_grid_rebuild_ = false;
      }
      grid_->update(display_poses_);
      refresh_mapping_cache();
    }
    if (was_mapping && !query_promotion && map_updated &&
        accepted_signature_id > 0 &&
        native_novelty_resume_reconciliation_pending_) {
      if (!finite_planar_transform(fused_odom_pose_) ||
          !finite_planar_transform(
              native_novelty_resume_last_progress_pose_)) {
        throw std::runtime_error(
            "novelty resume reconciliation has an invalid progress pose");
      }
      const bool independent_viewpoint =
          !frame.depth_scan.empty() && query_obstacle_views_are_independent(
              native_novelty_resume_last_progress_pose_.x(),
              native_novelty_resume_last_progress_pose_.y(),
              native_novelty_resume_last_progress_pose_.theta(),
              fused_odom_pose_);
      if (independent_viewpoint) {
        native_novelty_resume_last_progress_pose_ = fused_odom_pose_.to3DoF();
        const bool final_viewpoint =
            native_novelty_resume_reconciliation_viewpoints_remaining_ == 1U;
        if (native_novelty_resume_reconciliation_viewpoints_remaining_ > 0U &&
            (!final_viewpoint ||
             native_novelty_resume_frame_alignment_safe_)) {
          --native_novelty_resume_reconciliation_viewpoints_remaining_;
        }
      }
      if (trace_enabled()) {
        std::cerr << "native_novelty_resume_reconciliation_progress frame="
                  << frame.meta.frame_id << " node="
                  << accepted_signature_id << " independent_viewpoint="
                  << (independent_viewpoint ? 1 : 0) << " remaining="
                  << native_novelty_resume_reconciliation_viewpoints_remaining_
                  << " alignment_safe="
                  << (native_novelty_resume_frame_alignment_safe_ ? 1 : 0)
                  << " alignment_supports="
                  << native_novelty_resume_frame_alignment_supports_
                  << " alignment_max_bound_m="
                  << native_novelty_resume_frame_alignment_max_bound_m_
                  << "\n";
      }
      if (native_novelty_resume_reconciliation_viewpoints_remaining_ == 0U) {
        // The transition's accepted free observations become durable vetoes
        // before ordinary unknown obstacles are considered. The optimized
        // graph may have moved the raster by a sub-cell amount since this was
        // armed, so refresh both layers in the current graph frame instead of
        // merging cell indices from two different map gauges.
        native_novelty_resume_snapshot_map_ = cached_map_.clone();
        native_novelty_resume_snapshot_x_min_ = cached_x_min_;
        native_novelty_resume_snapshot_y_min_ = cached_y_min_;
        refresh_novelty_resume_protected_free(
            native_novelty_resume_snapshot_map_,
            native_novelty_resume_snapshot_x_min_,
            native_novelty_resume_snapshot_y_min_);
        native_novelty_resume_reconciliation_pending_ = false;
        native_novelty_resume_last_progress_pose_.setNull();
        if (trace_enabled()) {
          std::cerr << "native_novelty_resume_reconciliation_complete frame="
                    << frame.meta.frame_id << " node="
                    << accepted_signature_id << " protected_free="
                    << cv::countNonZero(
                           native_novelty_resume_protected_free_map_)
                    << "\n";
        }
      }
    }
    if (query_promotion && query_signature_committed) {
      const bool payload_cached =
          local_grids_.find(query_signature_id) != local_grids_.end() &&
          height_points_.find(query_signature_id) != height_points_.end() &&
          occupancy_height_points_.find(query_signature_id) !=
              occupancy_height_points_.end();
      if (!payload_cached ||
          !query_known_map_unchanged(
              query_known_map_before, query_known_x_min_before,
              query_known_y_min_before, cached_map_, cached_x_min_,
              cached_y_min_)) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "query local grid modified committed known occupancy");
      }
      const auto persisted_height =
          occupancy_height_points_.find(query_signature_id);
      const cv::Mat committed_marker =
          persisted_height == occupancy_height_points_.end()
              ? cv::Mat()
              : encode_persisted_query(
                    promoted_query_scope, promoted_query_generation,
                    br::QueryOutcome::kCommitted, promoted_query_kind,
                    promoted_query_anchor_id, promoted_query_candidate_id,
                    persisted_height->second);
      if (committed_marker.empty() ||
          !slam_->finalizeQuerySignature(query_signature_id,
                                         committed_marker)) {
        throw std::runtime_error(
            "query signature could not be durably finalized");
      }
      const rtabmap::Signature finalized = slam_->getSignatureCopy(
          query_signature_id, false, false, true, true, false, false);
      QueryScope finalized_scope = QueryScope::kNone;
      std::uint64_t finalized_generation = 0;
      br::QueryOutcome finalized_outcome = br::QueryOutcome::kNone;
      std::vector<HeightPoint> finalized_height;
      QueryPromotionKind finalized_kind = QueryPromotionKind::kNone;
      int finalized_anchor_id = 0;
      int finalized_candidate_id = 0;
      if (!decode_persisted_query(
              finalized.sensorData(), &finalized_scope,
              &finalized_generation, &finalized_outcome,
              &finalized_height, &finalized_kind, &finalized_anchor_id,
              &finalized_candidate_id) ||
          finalized_scope != promoted_query_scope ||
          finalized_generation != promoted_query_generation ||
          finalized_outcome != br::QueryOutcome::kCommitted ||
          finalized_kind != promoted_query_kind ||
          finalized_anchor_id != promoted_query_anchor_id ||
          finalized_candidate_id != promoted_query_candidate_id ||
          finalized_height.size() != persisted_height->second.size() ||
          !std::equal(
              finalized_height.begin(), finalized_height.end(),
              persisted_height->second.begin(),
              [](const HeightPoint &left, const HeightPoint &right) {
                return left.x == right.x && left.y == right.y &&
                       left.band == right.band;
              })) {
        throw std::runtime_error(
            "query terminal marker failed post-finalize verification");
      }
      query_integrity_failed_ = false;
      pending_slam_covariance_.release();
      mark_query_generation_completed(promoted_query_scope,
                                      promoted_query_generation,
                                      br::QueryOutcome::kCommitted);
      query_buffer_.clear();
      query_outcome_this_frame_ = br::QueryOutcome::kCommitted;
      if (query_promotions_ < std::numeric_limits<std::uint32_t>::max()) {
        ++query_promotions_;
      }
    }
    if (was_mapping) {
      ++mapping_frames_;
      mapping_travel_m_ += std::hypot(fused_increment.x(), fused_increment.y());
      mapping_rotation_rad_ += std::abs(fused_increment.theta());
      if (map_updated && accepted_signature_id > 0) {
        node_motion_progress_m_[accepted_signature_id] =
            mapping_travel_m_ +
            kFreezeRotationEquivalentRadiusM * mapping_rotation_rad_;
      }
      bool native_revisit_hold_qualified = false;
      for (const auto &[left, right] : new_metric_loop_pairs) {
        const auto left_motion = node_motion_progress_m_.find(left);
        const auto right_motion = node_motion_progress_m_.find(right);
        const float loop_motion_separation =
            left_motion != node_motion_progress_m_.end() &&
                    right_motion != node_motion_progress_m_.end()
                ? std::abs(left_motion->second - right_motion->second)
                : 0.0F;
        const int closure_gap = right - left;
        const bool evidence_qualified =
            rtab_native_revisit &&
            loop_has_independent_motion(loop_motion_separation);
        bool evidence_counted = false;
        if (evidence_qualified) {
          native_revisit_hold_qualified = true;
          const auto loop_pose = poses_.find(right);
          const rtabmap::Transform &region_pose =
              loop_pose != poses_.end() && !loop_pose->second.isNull()
                  ? loop_pose->second
                  : native_current_pose_;
          evidence_counted = note_visual_revisit(left, right, region_pose);
          if (evidence_counted) {
            last_visual_revisit_frame_ = mapping_frames_;
            recent_visual_revisit_frames_.push_back(mapping_frames_);
          }
        }
        if (trace_enabled()) {
          std::cerr << "visual_revisit_evidence frame=" << frame.meta.frame_id
                    << " pair=" << left << ":" << right
                    << " node_gap=" << closure_gap
                    << " motion_separation=" << loop_motion_separation
                    << " visual=" << (rtab_visual_registration ? 1 : 0)
                    << " geometric=" << (rtab_geometric_registration ? 1 : 0)
                    << " native_revisit=" << (rtab_native_revisit ? 1 : 0)
                    << " qualified=" << (evidence_qualified ? 1 : 0)
                    << " counted=" << (evidence_counted ? 1 : 0) << "\n";
        }
      }
      if (native_revisit_hold_qualified &&
          soft_mapping_state_ == SoftMappingState::kBuilding) {
        // The current signature and its accepted constraint are already a
        // committed mapping transaction. Arm the no-write state for the next
        // frame so this result never claims that a map-mutating frame was
        // itself read-only.
        native_revisit_hold_pending_ = true;
        native_revisit_hold_candidate_id_ = accepted_metric_peer;
        if (trace_enabled()) {
          std::cerr << "native_revisit_hold_armed frame="
                    << frame.meta.frame_id
                    << " candidate=" << accepted_metric_peer << "\n";
        }
      }
      if (mapping_frames_ % kFreezeMetricPeriodFrames == 0) {
        convergence_ = update_convergence_evidence();
        if (trace_enabled()) {
          std::cerr << "mapping_convergence frame=" << frame.meta.frame_id
                    << " mapping_frames=" << mapping_frames_
                    << " nodes=" << poses_.size()
                    << " travel=" << mapping_travel_m_
                    << " known=" << convergence_.known_cells
                    << " boundary=" << convergence_.boundary_cells
                    << " frontier=" << convergence_.frontier_cells
                    << " frontier_ratio=" << convergence_.frontier_ratio
                    << " growth_ratio=" << convergence_.known_growth_ratio
                      << " recent_growth_ratio="
                      << convergence_.recent_known_growth_ratio
                      << " observation_novelty="
                      << convergence_.observation_novelty_ratio
                      << " endpoint_novelty="
                      << convergence_.observation_endpoint_novelty_ratio
                      << " ray_novelty="
                      << convergence_.observation_ray_novelty_ratio
                    << " visual_revisits="
                    << convergence_.recent_visual_revisits
                    << " accepted_loop_events="
                    << convergence_.accepted_loop_events
                    << " visual_loop_regions="
                    << convergence_.visual_loop_regions
                    << " max_loop_node_span_ratio="
                    << convergence_.max_loop_node_span_ratio
                    << " max_loop_motion_span_ratio="
                    << convergence_.max_loop_motion_span_ratio
                    << " graph_common_nodes="
                    << convergence_.graph_common_nodes
                    << " graph_window_translation="
                    << convergence_.graph_window_translation_m
                    << " graph_window_yaw="
                    << convergence_.graph_window_yaw_rad
                    << " loop_settle_frames="
                    << mapping_frames_ - last_visual_revisit_frame_
                    << " usable_streak=" << usable_observation_streak_
                    << " ready=" << (convergence_.ready ? 1 : 0) << "\n";
        }
        consider_freeze_candidate(frame.meta.frame_id);
      }
    } else {
      if (trace_enabled()) {
        std::cerr << "localization_state frame=" << frame.meta.frame_id
                  << " matched=" << (localized_this_frame_ ? 1 : 0)
                  << " visual="
                  << (visual_localized_this_frame_ ? 1 : 0)
                  << " geometric="
                  << (geometric_localized_this_frame_ ? 1 : 0)
                  << " observations_since_metric="
                  << localization_observations_since_metric_ << "\n";
      }
    }
    remember_observation(frame, data);
    return snapshot(frame.meta.frame_id, true, map_updated, loop_closed);
  }

  ProcessResult snapshot(std::uint64_t frame_id, bool tracking_ok, bool map_updated,
                         bool loop_closed) const {
    ProcessResult result;
    result.meta.frame_id = frame_id;
    result.meta.tracking_ok = tracking_ok ? 1 : 0;
    result.meta.map_updated = map_updated ? 1 : 0;
    result.meta.loop_closed = loop_closed ? 1 : 0;
    result.meta.mode_flags =
        (mode_ == MappingMode::kLocalization ? 1U : 0U) |
        (localized_this_frame_ ? 2U : 0U) |
        (visual_localized_this_frame_ ? 4U : 0U) |
        (geometric_localized_this_frame_ ? 8U : 0U) |
        (read_only_match_this_frame_ ? 16U : 0U) |
        (recovery_hold_this_frame_ ? 32U : 0U) |
        (soft_mapping_state_ != SoftMappingState::kBuilding ? 64U : 0U) |
        (soft_mapping_state_ == SoftMappingState::kUncertainHold ? 128U : 0U);
    result.meta.query_outcome =
        static_cast<std::uint8_t>(query_outcome_this_frame_);
    result.meta.optimizer_backend =
        static_cast<std::uint8_t>(rtabmap::Optimizer::kTypeCeres);
    result.meta.query_scope =
        static_cast<std::uint8_t>(protocol_query_scope(query_scope_this_frame_));
    result.meta.query_generation = query_generation_this_frame_;
    result.meta.novelty_resume_reconciliation_viewpoints_remaining =
        native_novelty_resume_reconciliation_viewpoints_remaining_;
    result.meta.cell_size = kGridCellM;
    result.meta.pose_x = current_pose_.x();
    result.meta.pose_y = current_pose_.y();
    result.meta.pose_yaw = current_pose_.theta();
    result.meta.native_pose_x = native_current_pose_.x();
    result.meta.native_pose_y = native_current_pose_.y();
    result.meta.native_pose_yaw = native_current_pose_.theta();
    result.meta.fused_odom_pose_x = fused_odom_pose_.x();
    result.meta.fused_odom_pose_y = fused_odom_pose_.y();
    result.meta.fused_odom_pose_yaw = fused_odom_pose_.theta();
    result.meta.loop_count = loop_count_;
    result.meta.inliers = last_inliers_;
    result.meta.features = last_features_;
    result.meta.ref_node_id = last_ref_node_id_;

    float x_min = 0.0F;
    float y_min = 0.0F;
    cv::Mat map;
    if (mode_ == MappingMode::kLocalization && !frozen_map_.empty()) {
      map = frozen_map_;
      x_min = frozen_x_min_;
      y_min = frozen_y_min_;
    } else if (!cached_map_.empty()) {
      map = cached_map_;
      x_min = cached_x_min_;
      y_min = cached_y_min_;
    }
    result.meta.x_min = x_min;
    result.meta.y_min = y_min;
    if (!map.empty()) {
      if (map.type() != CV_8S || !map.isContinuous()) {
        map = map.clone();
      }
      result.meta.width = static_cast<std::uint32_t>(map.cols);
      result.meta.height = static_cast<std::uint32_t>(map.rows);
      result.meta.grid_bytes = static_cast<std::uint32_t>(map.total());
      const auto *begin = map.ptr<std::int8_t>(0);
      result.occupancy.assign(begin, begin + map.total());
      if (mode_ == MappingMode::kLocalization &&
          frozen_low_.size() == map.total() &&
          frozen_high_.size() == map.total()) {
        result.low = frozen_low_;
        result.high = frozen_high_;
      } else if (cached_low_.size() == map.total() &&
                 cached_high_.size() == map.total()) {
        result.low = cached_low_;
        result.high = cached_high_;
      } else {
        result.low.assign(map.total(), 0);
        result.high.assign(map.total(), 0);
        assemble_height_layers(
            result.low, result.high, map, x_min, y_min);
      }
      overlay_recovered_obstacles(result);
    }

    for (const auto &entry : poses_) {
      if (entry.first <= 0 || entry.second.isNull()) {
        continue;
      }
      const rtabmap::Transform pose = entry.second.to3DoF();
      result.poses.push_back({entry.first, pose.x(), pose.y(), pose.theta()});
    }
    result.meta.node_count = static_cast<std::uint32_t>(result.poses.size());
    result.meta.pose_count = static_cast<std::uint32_t>(result.poses.size());
    return result;
  }

  bool query_integrity_failed() const {
    return query_integrity_failed_;
  }

 private:
  static std::uint64_t query_cell_key(int x, int y) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x))
            << 32U) |
           static_cast<std::uint32_t>(y);
  }

  static int query_cell_x(std::uint64_t key) {
    return static_cast<std::int32_t>(key >> 32U);
  }

  static int query_cell_y(std::uint64_t key) {
    return static_cast<std::int32_t>(key & 0xffffffffULL);
  }

  static bool query_layer_is_valid(const cv::Mat &layer) {
    return layer.empty() ||
           (layer.rows == 1 && layer.depth() == CV_32F &&
            layer.channels() >= 2 && layer.isContinuous() &&
            cv::checkRange(layer, true, nullptr));
  }

  static std::size_t query_layer_bytes(const cv::Mat &layer) {
    return layer.empty() ? 0U : layer.total() * layer.elemSize();
  }

  static bool query_planar_covariance(const cv::Mat &input,
                                      cv::Matx33d *output) {
    if (output == nullptr || input.type() != CV_64FC1 ||
        !((input.rows == 3 && input.cols == 3) ||
          (input.rows == 6 && input.cols == 6)) ||
        !cv::checkRange(input, true, nullptr)) {
      return false;
    }
    constexpr int kPlanarAxes[3] = {0, 1, 5};
    cv::Matx33d covariance;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        covariance(row, column) =
            input.rows == 3
                ? input.at<double>(row, column)
                : input.at<double>(kPlanarAxes[row], kPlanarAxes[column]);
      }
    }
    covariance = (covariance + covariance.t()) * 0.5;
    cv::Mat eigenvalues;
    if (!cv::eigen(cv::Mat(covariance), eigenvalues) ||
        eigenvalues.total() != 3U ||
        eigenvalues.ptr<double>(0)[eigenvalues.total() - 1U] < -1.0e-10) {
      return false;
    }
    *output = covariance;
    return true;
  }

  static bool inverse_planar_covariance(
      const rtabmap::Transform &forward, const cv::Matx33d &forward_covariance,
      rtabmap::Transform *inverse, cv::Matx33d *inverse_covariance) {
    if (inverse == nullptr || inverse_covariance == nullptr ||
        !finite_planar_transform(forward)) {
      return false;
    }
    const rtabmap::Transform inverted = forward.inverse().to3DoF();
    if (!finite_planar_transform(inverted)) {
      return false;
    }
    const double cosine = std::cos(forward.theta());
    const double sine = std::sin(forward.theta());
    const cv::Matx33d jacobian(
        -cosine, -sine, inverted.y(),
         sine,   -cosine, -inverted.x(),
         0.0,     0.0,    -1.0);
    cv::Matx33d covariance =
        jacobian * forward_covariance * jacobian.t();
    covariance = (covariance + covariance.t()) * 0.5;
    if (!cv::checkRange(cv::Mat(covariance), true, nullptr)) {
      return false;
    }
    *inverse = inverted;
    *inverse_covariance = covariance;
    return true;
  }

  static bool query_point_position_covariance(
      const rtabmap::Transform &transform, const cv::Matx33d &pose_covariance,
      float local_x, float local_y, cv::Matx22d *point_covariance) {
    if (point_covariance == nullptr || !finite_planar_transform(transform) ||
        !std::isfinite(local_x) || !std::isfinite(local_y)) {
      return false;
    }
    const double cosine = std::cos(transform.theta());
    const double sine = std::sin(transform.theta());
    const cv::Matx<double, 2, 3> jacobian(
        1.0, 0.0, -sine * local_x - cosine * local_y,
        0.0, 1.0,  cosine * local_x - sine * local_y);
    cv::Matx22d covariance = jacobian * pose_covariance * jacobian.t();
    covariance = (covariance + covariance.t()) * 0.5;
    if (!cv::checkRange(cv::Mat(covariance), true, nullptr)) {
      return false;
    }
    *point_covariance = covariance;
    return true;
  }

  static bool query_point_one_sigma(const cv::Matx22d &covariance,
                                    double *one_sigma) {
    if (one_sigma == nullptr) {
      return false;
    }
    const double trace = covariance(0, 0) + covariance(1, 1);
    const double determinant = covariance(0, 0) * covariance(1, 1) -
                               covariance(0, 1) * covariance(1, 0);
    const double discriminant = trace * trace - 4.0 * determinant;
    if (!std::isfinite(trace) || !std::isfinite(determinant) ||
        discriminant < -1.0e-10) {
      return false;
    }
    const double largest =
        0.5 * (trace + std::sqrt(std::max(0.0, discriminant)));
    const double smallest = trace - largest;
    if (!std::isfinite(largest) || largest < -1.0e-10 ||
        smallest < -1.0e-10) {
      return false;
    }
    *one_sigma = std::sqrt(std::max(0.0, largest));
    return true;
  }

  static bool query_known_map_unchanged(
      const cv::Mat &before, float before_x_min, float before_y_min,
      const cv::Mat &after, float after_x_min, float after_y_min) {
    if (before.empty()) {
      return true;
    }
    if (after.empty() || before.type() != CV_8SC1 ||
        after.type() != CV_8SC1 || !std::isfinite(before_x_min) ||
        !std::isfinite(before_y_min) || !std::isfinite(after_x_min) ||
        !std::isfinite(after_y_min)) {
      return false;
    }
    const double column_shift_f =
        (static_cast<double>(before_x_min) - after_x_min) / kGridCellM;
    const double row_shift_f =
        (static_cast<double>(before_y_min) - after_y_min) / kGridCellM;
    const int column_shift = static_cast<int>(std::llround(column_shift_f));
    const int row_shift = static_cast<int>(std::llround(row_shift_f));
    if (std::abs(column_shift_f - column_shift) > 1.0e-3 ||
        std::abs(row_shift_f - row_shift) > 1.0e-3) {
      return false;
    }
    for (int row = 0; row < before.rows; ++row) {
      const std::int8_t *old_values = before.ptr<std::int8_t>(row);
      const int new_row = row + row_shift;
      if (new_row < 0 || new_row >= after.rows) {
        return false;
      }
      const std::int8_t *new_values = after.ptr<std::int8_t>(new_row);
      for (int column = 0; column < before.cols; ++column) {
        if (old_values[column] < 0) {
          continue;
        }
        const int new_column = column + column_shift;
        if (new_column < 0 || new_column >= after.cols ||
            new_values[new_column] != old_values[column]) {
          return false;
        }
      }
    }
    return true;
  }

  static bool query_constraints_equal(
      const std::multimap<int, rtabmap::Link> &left,
      const std::multimap<int, rtabmap::Link> &right) {
    if (left.size() != right.size()) {
      return false;
    }
    auto left_iter = left.begin();
    auto right_iter = right.begin();
    for (; left_iter != left.end(); ++left_iter, ++right_iter) {
      const rtabmap::Link &a = left_iter->second;
      const rtabmap::Link &b = right_iter->second;
      if (left_iter->first != right_iter->first || a.from() != b.from() ||
          a.to() != b.to() || a.type() != b.type()) {
        return false;
      }
      const bool both_null = a.transform().isNull() && b.transform().isNull();
      const rtabmap::Transform delta =
          both_null ? rtabmap::Transform::getIdentity()
                    : (a.transform().inverse() * b.transform()).to3DoF();
      if ((!both_null && (a.transform().isNull() || b.transform().isNull())) ||
          !finite_planar_transform(delta) ||
          std::hypot(delta.x(), delta.y()) > 1.0e-6F ||
          std::abs(wrap_angle(delta.theta())) > 1.0e-6F ||
          a.infMatrix().size() != b.infMatrix().size() ||
          a.infMatrix().type() != b.infMatrix().type() ||
          cv::norm(a.infMatrix(), b.infMatrix(), cv::NORM_INF) > 1.0e-9) {
        return false;
      }
    }
    return true;
  }

  static bool query_link_matches(
      const rtabmap::Link &link, int from, int to,
      const rtabmap::Transform &transform, const cv::Mat &information,
      rtabmap::Link::Type type = rtabmap::Link::kNeighbor) {
    if (link.from() != from || link.to() != to ||
        link.type() != type ||
        link.transform().isNull() || transform.isNull() ||
        link.infMatrix().size() != information.size() ||
        link.infMatrix().type() != information.type()) {
      return false;
    }
    const rtabmap::Transform delta =
        (transform.inverse() * link.transform()).to3DoF();
    return finite_planar_transform(delta) &&
           std::hypot(delta.x(), delta.y()) <= 1.0e-6F &&
           std::abs(wrap_angle(delta.theta())) <= 1.0e-6F &&
           cv::norm(link.infMatrix(), information, cv::NORM_INF) <= 1.0e-8;
  }

  bool query_terminal_outcome(QueryScope scope, std::uint64_t generation,
                              br::QueryOutcome *outcome) const {
    if (scope == QueryScope::kNone || generation == 0U || outcome == nullptr) {
      return false;
    }
    const auto &completed =
        scope == QueryScope::kRecovery ? completed_recovery_queries_
                                       : completed_normal_queries_;
    const auto found = completed.find(generation);
    if (found == completed.end()) {
      return false;
    }
    *outcome = found->second;
    return true;
  }

  bool query_generation_completed(QueryScope scope,
                                  std::uint64_t generation) const {
    br::QueryOutcome ignored = br::QueryOutcome::kNone;
    return query_terminal_outcome(scope, generation, &ignored);
  }

  static std::uint64_t query_session_nonce(std::uint64_t generation) {
    return generation >> 16U;
  }

  static std::uint16_t query_session_counter(std::uint64_t generation) {
    return static_cast<std::uint16_t>(generation & 0xffffU);
  }

  bool query_generation_is_stale(QueryScope scope,
                                 std::uint64_t generation) const {
    if (scope == QueryScope::kNone || generation == 0U ||
        query_session_nonce(generation) == 0U ||
        query_session_counter(generation) == 0U) {
      return true;
    }
    const auto &high_water =
        scope == QueryScope::kRecovery ? recovery_query_high_water_
                                       : normal_query_high_water_;
    const auto found = high_water.find(query_session_nonce(generation));
    return found != high_water.end() &&
           query_session_counter(generation) <= found->second;
  }

  void mark_query_generation_completed(QueryScope scope,
                                       std::uint64_t generation,
                                       br::QueryOutcome outcome) {
    UASSERT(generation > 0U);
    UASSERT(scope == QueryScope::kRecovery || scope == QueryScope::kNormal);
    UASSERT(outcome == br::QueryOutcome::kCommitted ||
            outcome == br::QueryOutcome::kAllKnown ||
            outcome == br::QueryOutcome::kBridgeOnlyDiscarded ||
            outcome == br::QueryOutcome::kNoveltyResumed);
    UASSERT(outcome != br::QueryOutcome::kNoveltyResumed ||
            scope == QueryScope::kNormal);
    auto &completed =
        scope == QueryScope::kRecovery ? completed_recovery_queries_
                                       : completed_normal_queries_;
    auto &high_water =
        scope == QueryScope::kRecovery ? recovery_query_high_water_
                                       : normal_query_high_water_;
    const std::uint64_t nonce = query_session_nonce(generation);
    const std::uint16_t counter = query_session_counter(generation);
    UASSERT(nonce > 0U && counter > 0U);
    const auto watermark = high_water.find(nonce);
    if (watermark == high_water.end()) {
      high_water.emplace(nonce, counter);
    } else {
      watermark->second = std::max(watermark->second, counter);
    }
    const auto existing = completed.find(generation);
    if (existing == completed.end()) {
      completed.emplace(generation, outcome);
      return;
    }
    // A durable bridge is written before a Q node. A later committed Q for
    // the same transaction is the stronger terminal certificate.
    if (existing->second == br::QueryOutcome::kBridgeOnlyDiscarded &&
        outcome == br::QueryOutcome::kCommitted) {
      existing->second = outcome;
      return;
    }
    UASSERT(existing->second == outcome);
  }

  std::string query_terminal_ledger_path() const {
    return database_path_ + ".query-terminals-v1";
  }

  void persist_graphless_query_terminal(QueryScope scope,
                                        std::uint64_t generation,
                                        int anchor_id,
                                        br::QueryOutcome outcome) {
    const bool valid_outcome =
        outcome == br::QueryOutcome::kAllKnown ||
        (outcome == br::QueryOutcome::kNoveltyResumed &&
         scope == QueryScope::kNormal);
    if ((scope != QueryScope::kRecovery && scope != QueryScope::kNormal) ||
        !valid_outcome || generation == 0U || anchor_id <= 0 ||
        durable_query_terminal_count_ >= kMaxDurableQueryTerminals) {
      throw std::runtime_error("invalid or full query terminal ledger");
    }
    DurableQueryTerminalRecord record{};
    std::memcpy(record.magic, kQueryTerminalLedgerMagic.data(),
                kQueryTerminalLedgerMagic.size());
    record.version = 1U;
    record.scope = static_cast<std::uint8_t>(scope);
    record.outcome = static_cast<std::uint8_t>(outcome);
    record.generation = generation;
    record.anchor_id = anchor_id;
    record.checksum = query_terminal_checksum(record);
    const std::string path = query_terminal_ledger_path();
    const bool ledger_existed = ::access(path.c_str(), F_OK) == 0;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
    if (fd < 0) {
      throw std::runtime_error("cannot open query terminal ledger: " +
                               std::string(std::strerror(errno)));
    }
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(&record);
    std::size_t written = 0U;
    while (written < sizeof(record)) {
      const ssize_t count =
          ::write(fd, bytes + written, sizeof(record) - written);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        const int saved_errno = errno;
        ::close(fd);
        query_integrity_failed_ = true;
        throw std::runtime_error("cannot append query terminal ledger: " +
                                 std::string(std::strerror(saved_errno)));
      }
      written += static_cast<std::size_t>(count);
    }
    if (::fsync(fd) != 0) {
      const int saved_errno = errno;
      ::close(fd);
      query_integrity_failed_ = true;
      throw std::runtime_error("cannot sync query terminal ledger: " +
                               std::string(std::strerror(saved_errno)));
    }
    ::close(fd);
    if (!ledger_existed) {
      const std::size_t separator = path.find_last_of('/');
      const std::string directory =
          separator == std::string::npos ? "." : path.substr(0, separator);
      const int directory_fd =
          ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
      if (directory_fd < 0 || ::fsync(directory_fd) != 0) {
        const int saved_errno = errno;
        if (directory_fd >= 0) {
          ::close(directory_fd);
        }
        query_integrity_failed_ = true;
        throw std::runtime_error("cannot sync query terminal directory: " +
                                 std::string(std::strerror(saved_errno)));
      }
      ::close(directory_fd);
    }
    ++durable_query_terminal_count_;
  }

  void restore_graphless_query_terminals(
      const std::map<int, rtabmap::Transform> &restored_poses) {
    const std::string path = query_terminal_ledger_path();
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      if (errno == ENOENT) {
        durable_query_terminal_count_ = 0U;
        return;
      }
      throw std::runtime_error("cannot open query terminal ledger: " +
                               std::string(std::strerror(errno)));
    }
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0) {
      const int saved_errno = errno;
      ::close(fd);
      throw std::runtime_error("cannot stat query terminal ledger: " +
                               std::string(std::strerror(saved_errno)));
    }
    const std::size_t bytes = static_cast<std::size_t>(status.st_size);
    const std::size_t records = bytes / sizeof(DurableQueryTerminalRecord);
    if (records > kMaxDurableQueryTerminals) {
      ::close(fd);
      throw std::runtime_error("query terminal ledger exceeds resource cap");
    }
    std::vector<DurableQueryTerminalRecord> decoded;
    decoded.reserve(records);
    for (std::size_t index = 0; index < records; ++index) {
      DurableQueryTerminalRecord record{};
      std::size_t consumed = 0U;
      while (consumed < sizeof(record)) {
        const ssize_t count = ::pread(
            fd, reinterpret_cast<std::uint8_t *>(&record) + consumed,
            sizeof(record) - consumed,
            static_cast<off_t>(index * sizeof(record) + consumed));
        if (count < 0 && errno == EINTR) {
          continue;
        }
        if (count <= 0) {
          ::close(fd);
          throw std::runtime_error("cannot read query terminal ledger");
        }
        consumed += static_cast<std::size_t>(count);
      }
      const QueryScope scope = static_cast<QueryScope>(record.scope);
      const br::QueryOutcome outcome =
          static_cast<br::QueryOutcome>(record.outcome);
      const bool valid_outcome =
          outcome == br::QueryOutcome::kAllKnown ||
          (outcome == br::QueryOutcome::kNoveltyResumed &&
           scope == QueryScope::kNormal);
      if (std::memcmp(record.magic, kQueryTerminalLedgerMagic.data(),
                      kQueryTerminalLedgerMagic.size()) != 0 ||
          record.version != 1U ||
          (scope != QueryScope::kRecovery && scope != QueryScope::kNormal) ||
          !valid_outcome ||
          record.generation == 0U || record.anchor_id <= 0 ||
          restored_poses.find(record.anchor_id) == restored_poses.end() ||
          !finite_planar_transform(restored_poses.at(record.anchor_id)) ||
          record.checksum != query_terminal_checksum(record)) {
        ::close(fd);
        throw std::runtime_error("query terminal ledger is corrupt or orphaned");
      }
      decoded.push_back(record);
    }
    const off_t valid_bytes =
        static_cast<off_t>(records * sizeof(DurableQueryTerminalRecord));
    if (valid_bytes != status.st_size) {
      if (::ftruncate(fd, valid_bytes) != 0 || ::fsync(fd) != 0) {
        const int saved_errno = errno;
        ::close(fd);
        throw std::runtime_error("cannot repair partial query terminal ledger: " +
                                 std::string(std::strerror(saved_errno)));
      }
    }
    ::close(fd);
    durable_query_terminal_count_ = records;
    for (const DurableQueryTerminalRecord &record : decoded) {
      mark_query_generation_completed(
          static_cast<QueryScope>(record.scope), record.generation,
          static_cast<br::QueryOutcome>(record.outcome));
    }
  }

  void discard_invalid_query_frame(std::uint64_t frame_id,
                                   const char *reason) {
    UASSERT(query_buffer_.active());
    if (query_invalid_frame_discards_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++query_invalid_frame_discards_;
    }
    if (trace_enabled()) {
      std::cerr << "query_frame_discarded frame=" << frame_id
                << " generation="
                << query_buffer_.generation << " scope="
                << static_cast<int>(query_buffer_.scope)
                << " reason=" << reason
                << " discarded_frames=" << query_invalid_frame_discards_
                << " keyframes=" << query_buffer_.keyframes
                << " chunks=" << query_buffer_.chunks.size()
                << " bytes=" << query_buffer_.bytes
                << " latched_release_kind="
                << static_cast<int>(query_buffer_.latched_release_kind)
                << "\n";
    }
    // A malformed local grid contains no usable evidence about this frame. It
    // cannot invalidate or erase already accepted observations: doing so could
    // remove the transaction's only free/occupied contradiction and authorize
    // a later unsafe commit. Keep the exact buffer, identity, anchor and
    // release certificate; the caller remains quarantined until another valid
    // frame can be checked.
  }

  [[noreturn]] void fail_query_buffer_integrity(const char *reason) {
    UASSERT(query_buffer_.active());
    if (query_integrity_failures_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++query_integrity_failures_;
    }
    if (trace_enabled()) {
      std::cerr << "query_buffer_integrity_failure generation="
                << query_buffer_.generation << " scope="
                << static_cast<int>(query_buffer_.scope)
                << " reason=" << reason
                << " failures=" << query_integrity_failures_
                << " keyframes=" << query_buffer_.keyframes
                << " chunks=" << query_buffer_.chunks.size()
                << " bytes=" << query_buffer_.bytes << "\n";
    }
    // A non-finite relative pose breaks the common coordinate frame of all
    // buffered evidence. Neither dropping the bad frame nor rebuilding from a
    // suffix can prove the retained observations consistent, so stop this
    // native session before any RTAB mutation instead of weakening the query.
    query_integrity_failed_ = true;
    throw std::runtime_error(
        std::string("provisional query integrity failure: ") + reason);
  }

  [[noreturn]] void fail_query_buffer_resource_limit(const char *reason) {
    UASSERT(query_buffer_.active());
    if (query_buffer_overflows_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++query_buffer_overflows_;
    }
    if (trace_enabled()) {
      std::cerr << "query_buffer_resource_limit generation="
                << query_buffer_.generation << " scope="
                << static_cast<int>(query_buffer_.scope)
                << " reason=" << reason
                << " keyframes=" << query_buffer_.keyframes
                << " chunks=" << query_buffer_.chunks.size()
                << " bytes=" << query_buffer_.bytes << "\n";
    }
    // Once accepted observations have contributed evidence, discarding any of
    // them can erase the only free/occupied contradiction and turn a reject
    // into an unsafe commit. Resource exhaustion is therefore terminal for this
    // worker session. It happens before RTAB mutation, so the committed graph
    // and raster remain unchanged.
    query_integrity_failed_ = true;
    throw std::runtime_error(std::string("provisional query exceeded ") + reason);
  }

  bool collect_query_view(QueryScope scope, std::uint64_t generation,
                          const FrameInput &frame,
                          const rtabmap::Transform &fused_pose,
                          QueryReleaseKind release_kind,
                          bool force_keyframe) {
    if (scope == QueryScope::kNone || generation == 0U ||
        query_generation_completed(scope, generation) ||
        query_generation_is_stale(scope, generation) ||
        frame.depth_scan.empty() || !finite_planar_transform(fused_pose) ||
        !query_grid_maker_) {
      return false;
    }
    if (!query_buffer_.active()) {
      const auto anchor_history =
          node_odometry_history_index_.find(last_mapping_node_id_);
      if (last_mapping_node_id_ <= 0 ||
          anchor_history == node_odometry_history_index_.end() ||
          anchor_history->second > odometry_history_.size()) {
        return false;
      }
      query_buffer_.scope = scope;
      query_buffer_.generation = generation;
      query_buffer_.anchor_node_id = last_mapping_node_id_;
      query_buffer_.anchor_odometry_history_index = anchor_history->second;
      query_buffer_.chunks.emplace_back();
    } else if (query_buffer_.scope != scope ||
               query_buffer_.generation != generation) {
      // A transaction has no implicit abort. Its identity remains fixed until
      // a terminal ACK is produced, even if the caller starts another search.
      return false;
    }
    if (release_kind == QueryReleaseKind::kPositiveMetric) {
      // A positive C->Q measurement is valid only for this frame and revokes
      // any older negative no-mode certificate. It is deliberately not
      // latched: an invalid local grid must wait for a fresh metric match.
      query_buffer_.latched_release_kind = QueryReleaseKind::kNone;
    } else if (release_kind == QueryReleaseKind::kNegativeNoMode) {
      query_buffer_.latched_release_kind = QueryReleaseKind::kNegativeNoMode;
    }
    if (query_buffer_.overflowed) {
      return false;
    }

    const QueryView *last = query_buffer_.last_view();
    float step_translation = 0.0F;
    float step_yaw = 0.0F;
    if (last != nullptr) {
      if (last->frame_id == frame.meta.frame_id) {
        return true;
      }
      const rtabmap::Transform step =
          (last->fused_pose.inverse() * fused_pose).to3DoF();
      if (!finite_planar_transform(step)) {
        fail_query_buffer_integrity("non_finite_relative_pose");
      }
      step_translation = std::hypot(step.x(), step.y());
      step_yaw = static_cast<float>(std::abs(wrap_angle(step.theta())));
      if (!force_keyframe && step_translation < kQueryKeyframeTranslationM &&
          step_yaw < kQueryKeyframeYawRad) {
        return true;
      }
    }

    bool start_new_chunk = false;
    if (!query_buffer_.chunks.empty() &&
        !query_buffer_.chunks.back().views.empty()) {
      const QueryChunk &chunk = query_buffer_.chunks.back();
      start_new_chunk =
          chunk.views.size() >= kQueryMaxKeyframesPerChunk ||
          chunk.travel_m + step_translation > kQueryChunkMaxTravelM ||
          chunk.turn_rad + step_yaw > kQueryChunkMaxTurnRad;
    }
    if (query_buffer_.keyframes >= kQueryMaxKeyframesHard) {
      fail_query_buffer_resource_limit("keyframe hard limit");
    }

    cv::Mat ground;
    cv::Mat obstacles;
    cv::Mat empty;
    cv::Point3f view_point(0.0F, 0.0F, 0.0F);
    query_grid_maker_->createLocalMap(
        frame.depth_scan, rtabmap::Transform::getIdentity(), ground,
        obstacles, empty, view_point);
    if (!query_layer_is_valid(ground) ||
        !query_layer_is_valid(obstacles) ||
        !query_layer_is_valid(empty)) {
      discard_invalid_query_frame(frame.meta.frame_id, "invalid_local_grid");
      return false;
    }
    if (ground.empty() && obstacles.empty() && empty.empty()) {
      return false;
    }
    const std::size_t bytes = query_layer_bytes(ground) +
                              query_layer_bytes(obstacles) +
                              query_layer_bytes(empty) +
                              frame.height_points.size() *
                                  sizeof(HeightPoint);
    if (bytes > kQueryMaxBytes ||
        query_buffer_.bytes > kQueryMaxBytes - bytes) {
      fail_query_buffer_resource_limit("64 MiB byte limit");
    }
    if (start_new_chunk) {
      query_buffer_.chunks.emplace_back();
    }
    QueryChunk &chunk = query_buffer_.chunks.back();
    if (!chunk.views.empty()) {
      chunk.travel_m += step_translation;
      chunk.turn_rad += step_yaw;
    }
    QueryView view;
    view.frame_id = frame.meta.frame_id;
    view.fused_pose = fused_pose.to3DoF();
    view.odometry_history_index = odometry_history_.size();
    view.ground = ground.clone();
    view.obstacles = obstacles.clone();
    view.empty = empty.clone();
    view.height_points = frame.height_points;
    view.bytes = bytes;
    chunk.views.push_back(std::move(view));
    ++query_buffer_.keyframes;
    query_buffer_.bytes += bytes;
    // A clean pre-insert rejection applies only to the evidence revision that
    // was checked. A genuinely new keyframe may resolve insufficient support,
    // so permit exactly one new attempt without changing transaction identity.
    query_buffer_.promotion_attempted = false;
    if (query_buffered_keyframes_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++query_buffered_keyframes_;
    }
    if (trace_enabled()) {
      std::cerr << "query_buffered frame=" << frame.meta.frame_id
                << " generation=" << generation << " scope="
                << static_cast<int>(scope)
                << " keyframes=" << query_buffer_.keyframes
                << " chunks=" << query_buffer_.chunks.size()
                << " bytes=" << query_buffer_.bytes << "\n";
    }
    return true;
  }

  static double query_covariance_diagonal(const cv::Mat &covariance,
                                          int planar_axis) {
    if (covariance.type() != CV_64FC1) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    int axis = planar_axis;
    if (covariance.rows == 6 && covariance.cols == 6) {
      axis = planar_axis == 2 ? 5 : planar_axis;
    } else if (covariance.rows != 3 || covariance.cols != 3) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double value = covariance.at<double>(axis, axis);
    return std::isfinite(value) && value >= 0.0
               ? value
               : std::numeric_limits<double>::quiet_NaN();
  }

  const PlanarOdometryStep &query_history_step(
      std::size_t index,
      const PlanarOdometryStep &endpoint_step_before_rebase) const {
    UASSERT(index < odometry_history_.size());
    return index + 1U == odometry_history_.size()
               ? endpoint_step_before_rebase
               : odometry_history_[index];
  }

  bool query_anchor_edge_covariance(
      int anchor_id, rtabmap::Transform *anchor_to_query,
      cv::Mat *covariance_6d) const {
    const auto anchor = node_odometry_history_index_.find(anchor_id);
    if (anchor_to_query == nullptr || covariance_6d == nullptr ||
        anchor_id <= 0 ||
        anchor == node_odometry_history_index_.end() ||
        anchor->second > raw_qvel_history_.size() ||
        raw_qvel_history_.size() != odometry_history_.size()) {
      return false;
    }
    rtabmap::Transform anchor_to_endpoint =
        rtabmap::Transform::getIdentity();
    cv::Matx33d planar_covariance = cv::Matx33d::zeros();
    for (std::size_t index = anchor->second;
         index < raw_qvel_history_.size(); ++index) {
      propagate_planar_odometry(raw_qvel_history_[index],
                                &anchor_to_endpoint,
                                &planar_covariance);
    }
    if (!finite_planar_transform(anchor_to_endpoint) ||
        !cv::checkRange(cv::Mat(planar_covariance), true, nullptr)) {
      return false;
    }
    // Numerical floors make the accumulated planar covariance invertible
    // without pretending that z/roll/pitch were observed. Those unused axes
    // receive a deliberately weak, finite one-metre/radian variance.
    planar_covariance =
        (planar_covariance + planar_covariance.t()) * 0.5;
    planar_covariance(0, 0) = std::max(
        planar_covariance(0, 0), kObservedTranslationVarianceFloor);
    planar_covariance(1, 1) = std::max(
        planar_covariance(1, 1), kObservedTranslationVarianceFloor);
    planar_covariance(2, 2) = std::max(
        planar_covariance(2, 2), kObservedYawVarianceFloor);
    cv::Mat eigenvalues;
    if (!cv::eigen(cv::Mat(planar_covariance), eigenvalues) ||
        eigenvalues.total() != 3U ||
        eigenvalues.ptr<double>(0)[eigenvalues.total() - 1U] <= 1.0e-12) {
      return false;
    }
    cv::Mat output = cv::Mat::eye(6, 6, CV_64FC1);
    constexpr int kPlanarAxes[3] = {0, 1, 5};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        output.at<double>(kPlanarAxes[row], kPlanarAxes[column]) =
            planar_covariance(row, column);
      }
    }
    if (!cv::checkRange(output, true, nullptr)) {
      return false;
    }
    *anchor_to_query = anchor_to_endpoint.to3DoF();
    *covariance_6d = std::move(output);
    return true;
  }

  QueryAggregate build_query_aggregate(
      const rtabmap::Transform &query_endpoint_odom_pose,
      const rtabmap::Transform &query_map_pose,
      const cv::Mat &pose_covariance,
      const rtabmap::Transform &pose_covariance_transform,
      bool verified_old_place,
      const PlanarOdometryStep &endpoint_step_before_rebase) const {
    QueryAggregate output;
    if (!query_buffer_.active() || query_buffer_.overflowed ||
        query_buffer_.keyframes == 0U ||
        !finite_planar_transform(query_endpoint_odom_pose) ||
        !finite_planar_transform(query_map_pose) ||
        !finite_planar_transform(pose_covariance_transform)) {
      return output;
    }
    cv::Matx33d endpoint_pose_covariance;
    if (!query_planar_covariance(pose_covariance,
                                 &endpoint_pose_covariance)) {
      return output;
    }
    double maximum_support_bound_m = 0.0;
    double maximum_safe_support_bound_m = 0.0;
    bool saw_uncertainty_support = false;

    // Aggregate directly on the committed global-map lattice. Rotating a
    // Q-local grid and then quantizing is not injective: two distinct local
    // cells can land in one final map cell. Merging only in Q coordinates can
    // therefore miss a free/occupied contradiction and permanently create a
    // second wall. Every evidence kind is merged after the final SE(2) map
    // transform, before unknown-only or old-map filtering.
    std::unordered_map<std::uint64_t, QueryCellEvidence> evidence;
    // Bounds are coalesced only when supports of the same polarity quantize to
    // the same final global cell. Keeping the largest radius preserves the
    // complete union of their uncertainty domains without dropping evidence.
    std::unordered_map<std::uint64_t, std::uint16_t> safe_free_bounds;
    std::unordered_map<std::uint64_t, std::uint16_t> safe_obstacle_bounds;
    std::unordered_map<std::uint64_t, std::uint16_t> unsafe_free_bounds;
    std::unordered_map<std::uint64_t, std::uint16_t> unsafe_obstacle_bounds;
    const std::size_t expected_cells = std::max<std::size_t>(
        256U, query_buffer_.bytes / (sizeof(cv::Vec2f) * 3U));
    evidence.reserve(expected_cells);
    safe_free_bounds.reserve(expected_cells / 2U);
    safe_obstacle_bounds.reserve(expected_cells / 4U);
    unsafe_free_bounds.reserve(expected_cells / 2U);
    unsafe_obstacle_bounds.reserve(expected_cells / 4U);
    rtabmap::Transform point_anchor_transform;
    cv::Matx33d point_anchor_covariance;
    rtabmap::Transform point_relative_transform;
    cv::Matx33d point_relative_covariance;
    bool include_relative_uncertainty = false;
    const auto point_uncertainty_bound = [&](
        float local_x, float local_y,
        const rtabmap::Transform &support_view_to_endpoint,
        double *bound_m) {
      if (bound_m == nullptr) {
        return false;
      }
      cv::Matx22d total;
      if (verified_old_place) {
        const float endpoint_x =
            support_view_to_endpoint.r11() * local_x +
            support_view_to_endpoint.r12() * local_y +
            support_view_to_endpoint.x();
        const float endpoint_y =
            support_view_to_endpoint.r21() * local_x +
            support_view_to_endpoint.r22() * local_y +
            support_view_to_endpoint.y();
        cv::Matx22d anchor_point;
        if (!query_point_position_covariance(
                point_anchor_transform, point_anchor_covariance, endpoint_x,
                endpoint_y, &anchor_point)) {
          return false;
        }
        double anchor_one_sigma = 0.0;
        if (!query_point_one_sigma(anchor_point, &anchor_one_sigma)) {
          return false;
        }
        double one_sigma = anchor_one_sigma;
        if (include_relative_uncertainty) {
          cv::Matx22d relative_point;
          if (!query_point_position_covariance(
                  point_relative_transform, point_relative_covariance,
                  local_x, local_y, &relative_point)) {
            return false;
          }
          // anchor_point is expressed in the old-place C frame, while the
          // V->Q history term above is expressed in Q.  Covariances can only
          // be added after transporting the relative point error through the
          // rotational part of C_T_Q.  Omitting this rotation can make two
          // complementary anisotropic errors appear aligned and understate
          // both the promotion gate and the old-wall safety tube.
          const double anchor_cosine =
              std::cos(point_anchor_transform.theta());
          const double anchor_sine =
              std::sin(point_anchor_transform.theta());
          const cv::Matx22d query_to_anchor_rotation(
              anchor_cosine, -anchor_sine,
              anchor_sine, anchor_cosine);
          const cv::Matx22d transported_relative =
              query_to_anchor_rotation * relative_point *
              query_to_anchor_rotation.t();
          double relative_one_sigma = 0.0;
          if (!query_point_one_sigma(transported_relative,
                                     &relative_one_sigma)) {
            return false;
          }
          // C->Q registration and V->Q tracking share endpoint-Q image
          // evidence, so their cross-covariance is not known to be zero. The
          // scalar triangle bound is valid even for perfectly correlated
          // errors, unlike adding the two covariance matrices as if the
          // estimates were independent.
          one_sigma += relative_one_sigma;
        }
        *bound_m =
            2.0 * one_sigma +
            std::sqrt(2.0) * static_cast<double>(kGridCellM);
        return std::isfinite(*bound_m) && *bound_m >= 0.0;
      } else {
        if (!query_point_position_covariance(
                point_anchor_transform, point_anchor_covariance, local_x,
                local_y, &total)) {
          return false;
        }
      }
      double one_sigma = 0.0;
      if (!query_point_one_sigma(total, &one_sigma)) {
        return false;
      }
      *bound_m =
          2.0 * one_sigma +
          std::sqrt(2.0) * static_cast<double>(kGridCellM);
      return std::isfinite(*bound_m) && *bound_m >= 0.0;
    };
    const auto note_spatial_bound = [&maximum_support_bound_m,
                                     &maximum_safe_support_bound_m,
                                     &saw_uncertainty_support,
                                     &output](
        std::unordered_map<std::uint64_t, std::uint16_t> *safe,
        std::unordered_map<std::uint64_t, std::uint16_t> *unsafe,
        std::uint64_t key, double bound_m, bool *safe_support) {
      if (safe == nullptr || unsafe == nullptr || !std::isfinite(bound_m) ||
          bound_m < 0.0 || safe_support == nullptr) {
        return false;
      }
      const double radius_f =
          std::ceil(bound_m / static_cast<double>(kGridCellM));
      if (!std::isfinite(radius_f) || radius_f < 0.0 ||
          radius_f > std::numeric_limits<std::uint16_t>::max()) {
        return false;
      }
      const auto radius = static_cast<std::uint16_t>(radius_f);
      maximum_support_bound_m = std::max(maximum_support_bound_m, bound_m);
      saw_uncertainty_support = true;
      const bool is_safe = bound_m <= kQueryMaxPromotionUncertaintyM;
      *safe_support = is_safe;
      auto *target = is_safe ? safe : unsafe;
      const auto inserted = target->emplace(key, radius);
      if (!inserted.second) {
        inserted.first->second = std::max(inserted.first->second, radius);
      }
      if (is_safe) {
        maximum_safe_support_bound_m =
            std::max(maximum_safe_support_bound_m, bound_m);
        ++output.safe_supports;
      } else {
        output.uncertainty_exceeded = true;
        ++output.unsafe_supports;
      }
      return true;
    };
    const auto note_layer = [&](const cv::Mat &layer,
                                const rtabmap::Transform &view_to_endpoint,
                                int kind) {
      std::unordered_set<std::uint64_t> seen;
      seen.reserve(layer.total());
      const int channels = layer.channels();
      const float *values = layer.empty() ? nullptr : layer.ptr<float>(0);
      for (std::size_t index = 0; index < layer.total(); ++index) {
        const float local_x = values[index * channels];
        const float local_y = values[index * channels + 1U];
        double support_bound_m = 0.0;
        if (!point_uncertainty_bound(local_x, local_y, view_to_endpoint,
                                     &support_bound_m)) {
          return false;
        }
        const float endpoint_x = view_to_endpoint.r11() * local_x +
                                 view_to_endpoint.r12() * local_y +
                                 view_to_endpoint.x();
        const float endpoint_y = view_to_endpoint.r21() * local_x +
                                 view_to_endpoint.r22() * local_y +
                                 view_to_endpoint.y();
        const float world_x = query_map_pose.r11() * endpoint_x +
                              query_map_pose.r12() * endpoint_y +
                              query_map_pose.x();
        const float world_y = query_map_pose.r21() * endpoint_x +
                              query_map_pose.r22() * endpoint_y +
                              query_map_pose.y();
        const int map_column = static_cast<int>(std::floor(
            (world_x - cached_x_min_) / kGridCellM));
        const int map_row = static_cast<int>(std::floor(
            (world_y - cached_y_min_) / kGridCellM));
        const std::uint64_t key = query_cell_key(map_column, map_row);
        const bool obstacle = kind == 1;
        bool safe_support = false;
        if (!note_spatial_bound(
                obstacle ? &safe_obstacle_bounds : &safe_free_bounds,
                obstacle ? &unsafe_obstacle_bounds : &unsafe_free_bounds,
                key, support_bound_m, &safe_support)) {
          return false;
        }
        if (!safe_support) {
          continue;
        }
        if (!seen.insert(key).second) {
          continue;
        }
        QueryCellEvidence &cell = evidence[key];
        if (kind == 1) {
          // A forced release frame is retained even when the robot has not
          // moved far enough to become a normal query keyframe. It still
          // contributes free, height and conflict evidence, but a repeated
          // observation from effectively the same pose cannot turn a
          // single-view depth artefact into a permanent wall. Two is the only
          // obstacle support threshold downstream, so saturate there.
          if (cell.obstacle_views == 0U) {
            cell.first_obstacle_view_x = view_to_endpoint.x();
            cell.first_obstacle_view_y = view_to_endpoint.y();
            cell.first_obstacle_view_yaw = view_to_endpoint.theta();
            cell.obstacle_views = 1U;
          } else if (
              cell.obstacle_views == 1U &&
              query_obstacle_views_are_independent(
                  cell.first_obstacle_view_x, cell.first_obstacle_view_y,
                  cell.first_obstacle_view_yaw, view_to_endpoint)) {
            cell.obstacle_views = 2U;
          }
        } else {
          std::uint16_t *count =
              kind == 0 ? &cell.ground_views : &cell.empty_views;
          if (*count < std::numeric_limits<std::uint16_t>::max()) {
            ++*count;
          }
        }
      }
      return true;
    };

    for (const QueryChunk &chunk : query_buffer_.chunks) {
      for (const QueryView &view : chunk.views) {
        if (view.odometry_history_index > odometry_history_.size()) {
          return output;
        }
        rtabmap::Transform view_to_endpoint_history =
            rtabmap::Transform::getIdentity();
        cv::Matx33d view_to_endpoint_history_covariance =
            cv::Matx33d::zeros();
        for (std::size_t index = view.odometry_history_index;
             index < odometry_history_.size(); ++index) {
          propagate_planar_odometry(
              query_history_step(index, endpoint_step_before_rebase),
              &view_to_endpoint_history,
              &view_to_endpoint_history_covariance);
        }
        const rtabmap::Transform measured_view_to_endpoint =
            (view.fused_pose.inverse() * query_endpoint_odom_pose).to3DoF();
        const rtabmap::Transform history_error =
            (view_to_endpoint_history.inverse() * measured_view_to_endpoint)
                .to3DoF();
        if (!finite_planar_transform(history_error) ||
            std::hypot(history_error.x(), history_error.y()) > kGridCellM ||
            std::abs(wrap_angle(history_error.theta())) >
                kQueryKeyframeYawRad) {
          return output;
        }
        rtabmap::Transform view_to_endpoint =
            (query_endpoint_odom_pose.inverse() * view.fused_pose).to3DoF();
        if (verified_old_place) {
          point_anchor_transform = pose_covariance_transform.to3DoF();
          point_anchor_covariance = endpoint_pose_covariance;
          if (!inverse_planar_covariance(
                  measured_view_to_endpoint,
                  view_to_endpoint_history_covariance,
                  &point_relative_transform,
                  &point_relative_covariance)) {
            return output;
          }
          include_relative_uncertainty =
              view.odometry_history_index < odometry_history_.size();
        } else {
          if (query_buffer_.anchor_odometry_history_index >
                  view.odometry_history_index ||
              view.odometry_history_index > raw_qvel_history_.size() ||
              raw_qvel_history_.size() != odometry_history_.size()) {
            return output;
          }
          point_anchor_transform = rtabmap::Transform::getIdentity();
          point_anchor_covariance = cv::Matx33d::zeros();
          for (std::size_t index =
                   query_buffer_.anchor_odometry_history_index;
               index < view.odometry_history_index; ++index) {
            propagate_planar_odometry(raw_qvel_history_[index],
                                      &point_anchor_transform,
                                      &point_anchor_covariance);
          }
          // A negative query has no old-place metric factor.  Its Q node and
          // sole A-Q graph edge are both defined by the raw compliant qvel
          // chain, so each buffered view must use the same A->Vi chain for its
          // nominal point positions. Mixing raw A->Q with fused Q->Vi would
          // rotate earlier walls when a later commanded turn was blocked,
          // while the direct A->Vi covariance correctly remained small.
          view_to_endpoint =
              (pose_covariance_transform.inverse() * point_anchor_transform)
                  .to3DoF();
          include_relative_uncertainty = false;
        }
        if (!finite_planar_transform(view_to_endpoint)) {
          return QueryAggregate{};
        }
        if (!note_layer(view.ground, view_to_endpoint, 0) ||
            !note_layer(view.obstacles, view_to_endpoint, 1) ||
            !note_layer(view.empty, view_to_endpoint, 2)) {
          return output;
        }
        for (const HeightPoint &point : view.height_points) {
          double support_bound_m = 0.0;
          if (!point_uncertainty_bound(point.x, point.y, view_to_endpoint,
                                       &support_bound_m)) {
            return output;
          }
          const float endpoint_x = view_to_endpoint.r11() * point.x +
                                   view_to_endpoint.r12() * point.y +
                                   view_to_endpoint.x();
          const float endpoint_y = view_to_endpoint.r21() * point.x +
                                   view_to_endpoint.r22() * point.y +
                                   view_to_endpoint.y();
          const float world_x = query_map_pose.r11() * endpoint_x +
                                query_map_pose.r12() * endpoint_y +
                                query_map_pose.x();
          const float world_y = query_map_pose.r21() * endpoint_x +
                                query_map_pose.r22() * endpoint_y +
                                query_map_pose.y();
          const std::uint64_t key = query_cell_key(
              static_cast<int>(std::floor(
                  (world_x - cached_x_min_) / kGridCellM)),
              static_cast<int>(std::floor(
                  (world_y - cached_y_min_) / kGridCellM)));
          bool safe_support = false;
          if (!note_spatial_bound(
                  &safe_obstacle_bounds, &unsafe_obstacle_bounds, key,
                  support_bound_m, &safe_support)) {
            return output;
          }
          if (safe_support) {
            evidence[key].height_bands |=
                static_cast<std::uint8_t>(1U << point.band);
          }
        }
      }
    }

    if (!saw_uncertainty_support) {
      return output;
    }
    output.promotion_uncertainty_m = maximum_support_bound_m;
    output.safe_promotion_uncertainty_m = maximum_safe_support_bound_m;
    if (!std::isfinite(output.promotion_uncertainty_m) ||
        !std::isfinite(output.safe_promotion_uncertainty_m)) {
      return output;
    }
    for (auto &entry : evidence) {
      QueryCellEvidence &cell = entry.second;
      cell.intrinsic_polarity_conflict =
          cell.obstacle_views != 0U &&
          (cell.ground_views != 0U || cell.empty_views != 0U);
    }
    // A negative query has no endpoint metric anchor. Its early observations
    // are positioned from A but would be owned by the endpoint Q local grid;
    // partially accepting them would become invalid when a future closure
    // moves Q. Keep the original all-support bound until query data is owned
    // by multiple submap nodes. Selective spatial backfill is positive-only.
    if (!verified_old_place && output.unsafe_supports != 0U) {
      return output;
    }

    if (verified_old_place && output.unsafe_supports != 0U) {
      using SpatialBounds =
          std::unordered_map<std::uint64_t, std::uint16_t>;
      std::int64_t minimum_column = std::numeric_limits<std::int64_t>::max();
      std::int64_t minimum_row = std::numeric_limits<std::int64_t>::max();
      std::int64_t maximum_column = std::numeric_limits<std::int64_t>::min();
      std::int64_t maximum_row = std::numeric_limits<std::int64_t>::min();
      const auto include_bounds = [&](const SpatialBounds &bounds) {
        for (const auto &entry : bounds) {
          const std::int64_t column = query_cell_x(entry.first);
          const std::int64_t row = query_cell_y(entry.first);
          const std::int64_t radius = entry.second;
          minimum_column = std::min(minimum_column, column - radius);
          minimum_row = std::min(minimum_row, row - radius);
          maximum_column = std::max(maximum_column, column + radius);
          maximum_row = std::max(maximum_row, row + radius);
        }
      };
      include_bounds(safe_free_bounds);
      include_bounds(safe_obstacle_bounds);
      include_bounds(unsafe_free_bounds);
      include_bounds(unsafe_obstacle_bounds);
      const std::int64_t domain_columns = maximum_column - minimum_column + 1;
      const std::int64_t domain_rows = maximum_row - minimum_row + 1;
      if (minimum_column > maximum_column || minimum_row > maximum_row ||
          domain_columns <= 0 || domain_rows <= 0 ||
          domain_columns > std::numeric_limits<int>::max() ||
          domain_rows > std::numeric_limits<int>::max() ||
          domain_columns > static_cast<std::int64_t>(
                               kQueryMaxUncertaintyDomainCells) /
                               domain_rows) {
        output.uncertainty_domain_exceeded = true;
        return output;
      }
      output.uncertainty_domain_cells =
          static_cast<std::size_t>(domain_columns) *
          static_cast<std::size_t>(domain_rows);
      const auto radius_bucket_count = [](const SpatialBounds &bounds) {
        std::unordered_set<std::uint16_t> radii;
        radii.reserve(bounds.size());
        for (const auto &entry : bounds) {
          radii.insert(entry.second);
        }
        return radii.size();
      };
      output.uncertainty_radius_buckets =
          radius_bucket_count(safe_free_bounds) +
          radius_bucket_count(safe_obstacle_bounds) +
          radius_bucket_count(unsafe_free_bounds) +
          radius_bucket_count(unsafe_obstacle_bounds);
      if (output.uncertainty_radius_buckets != 0U &&
          output.uncertainty_domain_cells >
              kQueryMaxUncertaintyDomainWorkCells /
                  output.uncertainty_radius_buckets) {
        output.uncertainty_domain_exceeded = true;
        return output;
      }
      output.uncertainty_domain_work_cells =
          output.uncertainty_domain_cells *
          output.uncertainty_radius_buckets;

      const cv::Size domain_size(static_cast<int>(domain_columns),
                                 static_cast<int>(domain_rows));
      const auto rasterize_domains = [&](const SpatialBounds &bounds,
                                         cv::Mat *mask) {
        if (mask == nullptr) {
          return false;
        }
        *mask = cv::Mat::zeros(domain_size, CV_8UC1);
        if (bounds.empty()) {
          return true;
        }
        std::map<int, std::vector<cv::Point>> centers_by_radius;
        for (const auto &entry : bounds) {
          const int column = static_cast<int>(
              static_cast<std::int64_t>(query_cell_x(entry.first)) -
              minimum_column);
          const int row = static_cast<int>(
              static_cast<std::int64_t>(query_cell_y(entry.first)) -
              minimum_row);
          if (column < 0 || row < 0 || column >= domain_size.width ||
              row >= domain_size.height) {
            return false;
          }
          centers_by_radius[entry.second].emplace_back(column, row);
        }
        for (const auto &bucket : centers_by_radius) {
          cv::Mat inverse_centers(domain_size, CV_8UC1, cv::Scalar(255));
          for (const cv::Point &center : bucket.second) {
            inverse_centers.at<std::uint8_t>(center) = 0U;
          }
          cv::Mat distance;
          cv::distanceTransform(inverse_centers, distance, cv::DIST_L2,
                                cv::DIST_MASK_PRECISE);
          cv::Mat covered;
          cv::compare(distance, static_cast<float>(bucket.first), covered,
                      cv::CMP_LE);
          cv::bitwise_or(*mask, covered, *mask);
        }
        return true;
      };

      cv::Mat safe_free_domain;
      cv::Mat safe_obstacle_domain;
      cv::Mat unsafe_free_domain;
      cv::Mat unsafe_obstacle_domain;
      if (!rasterize_domains(safe_free_bounds, &safe_free_domain) ||
          !rasterize_domains(safe_obstacle_bounds,
                             &safe_obstacle_domain) ||
          !rasterize_domains(unsafe_free_bounds, &unsafe_free_domain) ||
          !rasterize_domains(unsafe_obstacle_bounds,
                             &unsafe_obstacle_domain)) {
        output.uncertainty_domain_exceeded = true;
        return output;
      }
      cv::Mat unsafe_domain;
      cv::bitwise_or(unsafe_free_domain, unsafe_obstacle_domain,
                     unsafe_domain);
      cv::Mat component_labels;
      const int component_count = cv::connectedComponents(
          unsafe_domain, component_labels, 8, CV_32S);
      if (component_count <= 0) {
        output.uncertainty_domain_exceeded = true;
        return output;
      }
      output.unsafe_components =
          static_cast<std::size_t>(component_count - 1);

      cv::Mat old_occupied = cv::Mat::zeros(domain_size, CV_8UC1);
      cv::Mat old_free = cv::Mat::zeros(domain_size, CV_8UC1);
      if (!cached_map_.empty()) {
        for (int domain_row = 0; domain_row < domain_size.height;
             ++domain_row) {
          const std::int64_t map_row =
              minimum_row + static_cast<std::int64_t>(domain_row);
          if (map_row < 0 || map_row >= cached_map_.rows) {
            continue;
          }
          const std::int8_t *map_values =
              cached_map_.ptr<std::int8_t>(static_cast<int>(map_row));
          std::uint8_t *occupied_values =
              old_occupied.ptr<std::uint8_t>(domain_row);
          std::uint8_t *free_values = old_free.ptr<std::uint8_t>(domain_row);
          for (int domain_column = 0;
               domain_column < domain_size.width; ++domain_column) {
            const std::int64_t map_column =
                minimum_column + static_cast<std::int64_t>(domain_column);
            if (map_column < 0 || map_column >= cached_map_.cols) {
              continue;
            }
            const std::int8_t value = map_values[map_column];
            occupied_values[domain_column] = value >= 65 ? 255U : 0U;
            free_values[domain_column] =
                value >= 0 && value < 65 ? 255U : 0U;
          }
        }
      }

      struct UncertainComponentState {
        bool bad = false;
        bool touches_old_occupied = false;
        bool touches_safe_obstacle = false;
      };
      std::vector<UncertainComponentState> component_state(
          static_cast<std::size_t>(component_count));
      for (int row = 0; row < domain_size.height; ++row) {
        const int *labels = component_labels.ptr<int>(row);
        const std::uint8_t *unsafe_free =
            unsafe_free_domain.ptr<std::uint8_t>(row);
        const std::uint8_t *unsafe_obstacle =
            unsafe_obstacle_domain.ptr<std::uint8_t>(row);
        const std::uint8_t *safe_free =
            safe_free_domain.ptr<std::uint8_t>(row);
        const std::uint8_t *safe_obstacle =
            safe_obstacle_domain.ptr<std::uint8_t>(row);
        const std::uint8_t *occupied =
            old_occupied.ptr<std::uint8_t>(row);
        const std::uint8_t *free = old_free.ptr<std::uint8_t>(row);
        for (int column = 0; column < domain_size.width; ++column) {
          const int label = labels[column];
          if (label <= 0) {
            continue;
          }
          UncertainComponentState &state =
              component_state[static_cast<std::size_t>(label)];
          state.touches_old_occupied |= occupied[column] != 0U;
          state.touches_safe_obstacle |= safe_obstacle[column] != 0U;
          state.bad |=
              (unsafe_free[column] != 0U &&
               unsafe_obstacle[column] != 0U) ||
              (unsafe_free[column] != 0U && occupied[column] != 0U) ||
              (unsafe_obstacle[column] != 0U && free[column] != 0U) ||
              (unsafe_free[column] != 0U &&
               safe_obstacle[column] != 0U) ||
              (unsafe_obstacle[column] != 0U && safe_free[column] != 0U);
        }
      }
      for (std::size_t label = 1; label < component_state.size(); ++label) {
        UncertainComponentState &state = component_state[label];
        // A connected uncertain domain joining an established wall to a new
        // obstacle could be the same wall at two poses. Keep that entire
        // component provisional even when its evidence has the same polarity.
        state.bad |=
            state.touches_old_occupied && state.touches_safe_obstacle;
        if (state.bad) {
          ++output.uncertain_veto_components;
        }
      }

      if (output.uncertain_veto_components != 0U) {
        cv::Mat veto_domain = cv::Mat::zeros(domain_size, CV_8UC1);
        for (int row = 0; row < domain_size.height; ++row) {
          const int *labels = component_labels.ptr<int>(row);
          std::uint8_t *veto = veto_domain.ptr<std::uint8_t>(row);
          for (int column = 0; column < domain_size.width; ++column) {
            const int label = labels[column];
            if (label > 0 &&
                component_state[static_cast<std::size_t>(label)].bad) {
              veto[column] = 255U;
            }
          }
        }
        cv::Mat inverse_veto;
        cv::bitwise_not(veto_domain, inverse_veto);
        cv::Mat distance_to_veto;
        cv::distanceTransform(inverse_veto, distance_to_veto, cv::DIST_L2,
                              cv::DIST_MASK_PRECISE);
        const auto overlaps_veto = [&](const SpatialBounds &bounds,
                                       std::uint64_t key) {
          const auto found = bounds.find(key);
          if (found == bounds.end()) {
            return false;
          }
          const int column = static_cast<int>(
              static_cast<std::int64_t>(query_cell_x(key)) - minimum_column);
          const int row = static_cast<int>(
              static_cast<std::int64_t>(query_cell_y(key)) - minimum_row);
          return column >= 0 && row >= 0 && column < domain_size.width &&
                 row < domain_size.height &&
                 distance_to_veto.at<float>(row, column) <= found->second;
        };
        for (auto &entry : evidence) {
          QueryCellEvidence &cell = entry.second;
          bool vetoed = false;
          if ((cell.ground_views != 0U || cell.empty_views != 0U) &&
              overlaps_veto(safe_free_bounds, entry.first)) {
            cell.ground_views = 0U;
            cell.empty_views = 0U;
            vetoed = true;
          }
          if ((cell.obstacle_views != 0U || cell.height_bands != 0U) &&
              overlaps_veto(safe_obstacle_bounds, entry.first)) {
            cell.obstacle_views = 0U;
            cell.height_bands = 0U;
            vetoed = true;
          }
          if (vetoed) {
            ++output.uncertain_veto_cells;
          }
        }
      }
    }

    cv::Mat distance_to_old_occupied;
    int old_occupied_padding = 0;
    if (!cached_map_.empty()) {
      cv::Mat occupied = cached_map_ >= 65;
      if (cv::countNonZero(occupied) != 0) {
        const auto note_maximum_radius = [&](const auto &bounds) {
          for (const auto &entry : bounds) {
            old_occupied_padding = std::max(
                old_occupied_padding, static_cast<int>(entry.second));
          }
        };
        note_maximum_radius(safe_free_bounds);
        note_maximum_radius(safe_obstacle_bounds);
        cv::Mat padded_occupied;
        cv::copyMakeBorder(occupied, padded_occupied, old_occupied_padding,
                           old_occupied_padding, old_occupied_padding,
                           old_occupied_padding, cv::BORDER_CONSTANT,
                           cv::Scalar(0));
        cv::Mat inverse_occupied;
        cv::bitwise_not(padded_occupied, inverse_occupied);
        cv::distanceTransform(inverse_occupied, distance_to_old_occupied,
                              cv::DIST_L2, cv::DIST_MASK_PRECISE);
      }
    }

    std::vector<cv::Vec2f> accepted_ground;
    std::vector<cv::Vec2f> accepted_obstacles;
    std::vector<cv::Vec2f> accepted_empty;
    accepted_ground.reserve(evidence.size() / 4U);
    accepted_obstacles.reserve(evidence.size() / 4U);
    accepted_empty.reserve(evidence.size() / 2U);
    const rtabmap::Transform map_to_query = query_map_pose.inverse().to3DoF();
    if (!finite_planar_transform(map_to_query)) {
      return output;
    }
    for (const auto &entry : evidence) {
      const int map_column = query_cell_x(entry.first);
      const int map_row = query_cell_y(entry.first);
      const QueryCellEvidence &cell = entry.second;
      if (cell.intrinsic_polarity_conflict) {
        // An unsafe component may quarantine one side of a contradiction, but
        // it cannot retroactively turn the other already-safe observation into
        // an authorization to write this cell.
        ++output.conflict_cells;
        continue;
      }
      if (cell.ground_views == 0U && cell.obstacle_views == 0U &&
          cell.empty_views == 0U) {
        continue;
      }
      if (cell.obstacle_views > 0U &&
          (cell.ground_views > 0U || cell.empty_views > 0U)) {
        // Multiple held viewpoints disagree about this same Q-frame cell.
        // Treat it like an old-map conflict: selecting the obstacle vote would
        // turn occlusion/dynamic evidence into a permanent wall.
        ++output.conflict_cells;
        continue;
      }
      const float world_x = cached_x_min_ +
                            (static_cast<float>(map_column) + 0.5F) *
                                kGridCellM;
      const float world_y = cached_y_min_ +
                            (static_cast<float>(map_row) + 0.5F) *
                                kGridCellM;
      const float local_x = map_to_query.r11() * world_x +
                            map_to_query.r12() * world_y + map_to_query.x();
      const float local_y = map_to_query.r21() * world_x +
                            map_to_query.r22() * world_y + map_to_query.y();
      const bool in_map = !cached_map_.empty() && map_column >= 0 &&
                          map_row >= 0 && map_column < cached_map_.cols &&
                          map_row < cached_map_.rows;
      const std::int8_t old_value =
          in_map ? cached_map_.at<std::int8_t>(map_row, map_column) : -1;
      const bool old_known = old_value >= 0;
      const bool old_occupied = old_value >= 65;
      const bool obstacle = cell.obstacle_views > 0U;
      const auto &cell_bounds =
          obstacle ? safe_obstacle_bounds : safe_free_bounds;
      const auto cell_bound = cell_bounds.find(entry.first);
      if (cell_bound == cell_bounds.end()) {
        ++output.conflict_cells;
        continue;
      }
      const int wall_distance_column = map_column + old_occupied_padding;
      const int wall_distance_row = map_row + old_occupied_padding;
      const bool near_old_wall =
          !distance_to_old_occupied.empty() && wall_distance_column >= 0 &&
          wall_distance_row >= 0 &&
          wall_distance_column < distance_to_old_occupied.cols &&
          wall_distance_row < distance_to_old_occupied.rows &&
          distance_to_old_occupied.at<float>(wall_distance_row,
                                             wall_distance_column) <=
              static_cast<float>(cell_bound->second);
      if (obstacle) {
        if (old_occupied || near_old_wall) {
          ++output.duplicate_cells;
          continue;
        }
        if (old_known) {
          ++output.conflict_cells;
          continue;
        }
        const std::uint16_t minimum_views = verified_old_place ? 2U : 1U;
        if (cell.obstacle_views < minimum_views) {
          ++output.unsupported_cells;
          continue;
        }
        accepted_obstacles.emplace_back(local_x, local_y);
        ++output.accepted_obstacle_cells;
        for (std::uint8_t band = 0; band < kWallBandCount; ++band) {
          if ((cell.height_bands & (1U << band)) != 0U) {
            output.height_points.push_back({local_x, local_y, band});
          }
        }
      } else {
        if (old_occupied) {
          ++output.conflict_cells;
          continue;
        }
        if (old_known || near_old_wall) {
          ++output.duplicate_cells;
          continue;
        }
        if (cell.ground_views > 0U) {
          accepted_ground.emplace_back(local_x, local_y);
        } else {
          accepted_empty.emplace_back(local_x, local_y);
        }
        ++output.accepted_free_cells;
      }
    }

    const std::size_t accepted_cells = output.accepted_free_cells +
                                       output.accepted_obstacle_cells;
    // Old-known cells are immutable during backfill. Even one contradiction
    // means the anchor/filter is not trustworthy enough to write elsewhere in
    // the same aggregate; dropping only the conflicting cell could still draw
    // a displaced second wall in an adjacent unknown band.
    if (output.conflict_cells != 0U) {
      return output;
    }
    if (accepted_cells == 0U) {
      output.all_known = output.conflict_cells == 0U &&
                         output.unsupported_cells == 0U &&
                         output.duplicate_cells > 0U;
      output.valid = output.all_known;
      return output;
    }
    const auto point_matrix = [](const std::vector<cv::Vec2f> &points) {
      return points.empty()
                 ? cv::Mat()
                 : cv::Mat(1, static_cast<int>(points.size()), CV_32FC2,
                           const_cast<cv::Vec2f *>(points.data())).clone();
    };
    output.ground = point_matrix(accepted_ground);
    output.obstacles = point_matrix(accepted_obstacles);
    output.empty = point_matrix(accepted_empty);
    output.valid = query_layer_is_valid(output.ground) &&
                   query_layer_is_valid(output.obstacles) &&
                   query_layer_is_valid(output.empty);
    return output;
  }

  void note_localization_outcome(double stamp_s, bool metric_registration) {
    ++localization_frames_;
    if (!metric_registration) {
      if (localization_observations_since_metric_ <
          std::numeric_limits<std::uint32_t>::max()) {
        ++localization_observations_since_metric_;
      }
      return;
    }
    localization_observations_since_metric_ = 0;
    if (std::isfinite(stamp_s)) {
      localization_last_metric_stamp_s_ = stamp_s;
    }
  }

  void restore_persisted_state() {
    if (!slam_ || !grid_) {
      return;
    }
    std::map<int, rtabmap::Transform> restored_poses;
    std::multimap<int, rtabmap::Link> restored_constraints;
    std::map<int, rtabmap::Signature> restored_signatures;
    slam_->getGraph(restored_poses, restored_constraints, true, true,
                    &restored_signatures, false, false, true, true, false,
                    false);
    if (restored_poses.empty()) {
      restore_graphless_query_terminals(restored_poses);
      return;
    }

    for (const auto &entry : restored_poses) {
      if (entry.first > 0 && finite_planar_transform(entry.second)) {
        poses_[entry.first] = entry.second.to3DoF();
        last_mapping_node_id_ = std::max(last_mapping_node_id_, entry.first);
      }
    }
    using PersistedQueryKey = std::pair<int, std::uint64_t>;
    const auto valid_persisted_query_link = [](const rtabmap::Link &link) {
      if (link.transform().isNull() ||
          !finite_planar_transform(link.transform().to3DoF()) ||
          link.infMatrix().rows != 6 || link.infMatrix().cols != 6 ||
          link.infMatrix().type() != CV_64FC1 ||
          !cv::checkRange(link.infMatrix(), true, nullptr)) {
        return false;
      }
      for (int axis = 0; axis < 6; ++axis) {
        if (!(link.infMatrix().at<double>(axis, axis) > 0.0)) {
          return false;
        }
      }
      return true;
    };
    struct PersistedBridgeTerminal {
      QueryScope scope = QueryScope::kNone;
      std::uint64_t generation = 0;
      int anchor_id = 0;
      int candidate_id = 0;
    };
    std::map<PersistedQueryKey, PersistedBridgeTerminal> bridge_terminals;
    for (const auto &entry : restored_constraints) {
      const cv::Mat marker = entry.second.uncompressUserDataConst();
      QueryScope scope = QueryScope::kNone;
      std::uint64_t generation = 0;
      br::QueryOutcome outcome = br::QueryOutcome::kNone;
      std::vector<HeightPoint> no_height_points;
      QueryPromotionKind kind = QueryPromotionKind::kNone;
      int anchor_id = 0;
      int candidate_id = 0;
      if (decode_persisted_query_data(
              marker, &scope, &generation, &outcome, &no_height_points, &kind,
              &anchor_id, &candidate_id)) {
        if (outcome != br::QueryOutcome::kBridgeOnlyDiscarded ||
            kind != QueryPromotionKind::kBridgeOnly ||
            anchor_id == candidate_id ||
            poses_.find(anchor_id) == poses_.end() ||
            poses_.find(candidate_id) == poses_.end() ||
            !valid_persisted_query_link(entry.second) ||
            entry.second.type() != rtabmap::Link::kGlobalClosure ||
            !((entry.second.from() == anchor_id &&
               entry.second.to() == candidate_id) ||
              (entry.second.from() == candidate_id &&
               entry.second.to() == anchor_id))) {
          query_integrity_failed_ = true;
          throw std::runtime_error(
              "database query bridge marker has invalid topology");
        }
        const PersistedQueryKey key{static_cast<int>(scope), generation};
        const auto inserted = bridge_terminals.emplace(
            key, PersistedBridgeTerminal{
                     scope, generation, anchor_id, candidate_id});
        if (!inserted.second &&
            (inserted.first->second.anchor_id != anchor_id ||
             inserted.first->second.candidate_id != candidate_id)) {
          query_integrity_failed_ = true;
          throw std::runtime_error(
              "database contains conflicting query bridge terminals");
        }
      }
    }
    std::set<PersistedQueryKey> committed_query_terminals;
    for (const auto &entry : restored_signatures) {
      const int node_id = entry.first;
      const rtabmap::SensorData &sensor_data = entry.second.sensorData();
      rtabmap::LaserScan registration_scan;
      cv::Mat user_data;
      cv::Mat ground;
      cv::Mat obstacles;
      cv::Mat empty;
      sensor_data.uncompressDataConst(nullptr, nullptr, &registration_scan,
                                      &user_data,
                                      &ground, &obstacles, &empty);
      std::vector<HeightPoint> registration_height_points =
          height_points_from_scan(registration_scan);
      QueryScope scope = QueryScope::kNone;
      std::uint64_t generation = 0;
      br::QueryOutcome outcome = br::QueryOutcome::kNone;
      std::vector<HeightPoint> height_points;
      QueryPromotionKind kind = QueryPromotionKind::kNone;
      int anchor_id = 0;
      int candidate_id = 0;
      const bool persisted_query = decode_persisted_query_data(
          user_data, &scope, &generation, &outcome, &height_points, &kind,
          &anchor_id, &candidate_id);
      std::vector<HeightPoint> ordinary_height_points;
      const bool persisted_height = decode_persisted_height_data(
          user_data, &ordinary_height_points);
      if ((persisted_query_magic(user_data) && !persisted_query) ||
          (persisted_height_magic(user_data) && !persisted_height)) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "database contains a corrupt persistent map payload");
      }
      if (node_id <= 0 || poses_.find(node_id) == poses_.end()) {
        if (persisted_query || persisted_height) {
          query_integrity_failed_ = true;
          throw std::runtime_error(
              "database persistent map payload has no finite restored pose");
        }
        continue;
      }
      const cv::Point3f grid_viewpoint = sensor_data.gridViewPoint();
      const bool local_grid_valid =
          std::isfinite(sensor_data.gridCellSize()) &&
          sensor_data.gridCellSize() > 0.0F &&
          query_layer_is_valid(ground) && query_layer_is_valid(obstacles) &&
          query_layer_is_valid(empty) && std::isfinite(grid_viewpoint.x) &&
          std::isfinite(grid_viewpoint.y) && std::isfinite(grid_viewpoint.z);
      if (persisted_query &&
          (!local_grid_valid ||
           std::abs(sensor_data.gridCellSize() - kGridCellM) > 1.0e-6F ||
           (ground.empty() && obstacles.empty() && empty.empty()))) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "database query signature has an invalid or empty local grid");
      }
      if (local_grid_valid) {
        local_grids_.add(node_id, ground, obstacles, empty,
                         sensor_data.gridCellSize(),
                         grid_viewpoint);
      }

      if (persisted_height) {
        if (!local_grid_valid ||
            std::abs(sensor_data.gridCellSize() - kGridCellM) > 1.0e-6F ||
            (ground.empty() && obstacles.empty() && empty.empty())) {
          query_integrity_failed_ = true;
          throw std::runtime_error(
              "database ordinary height payload has no valid local grid");
        }
        std::vector<HeightPoint> recovered_height_points =
            recovered_height_points_from_persisted(
                registration_height_points, ordinary_height_points);
        height_points_[node_id] = registration_height_points.empty()
                                      ? ordinary_height_points
                                      : std::move(registration_height_points);
        occupancy_height_points_[node_id] =
            std::move(ordinary_height_points);
        if (!recovered_height_points.empty()) {
          recovered_height_points_[node_id] =
              std::move(recovered_height_points);
        }
      }

      if (persisted_query) {
        if (local_grids_.find(node_id) == local_grids_.end()) {
          query_integrity_failed_ = true;
          throw std::runtime_error(
              "database query signature local grid was not restored");
        }
        if (outcome == br::QueryOutcome::kHolding) {
          query_integrity_failed_ = true;
          throw std::runtime_error(
              "database contains an unfinalized query signature");
        }
        if (outcome == br::QueryOutcome::kCommitted) {
          const PersistedQueryKey key{static_cast<int>(scope), generation};
          std::size_t anchor_neighbors = 0U;
          std::size_t candidate_closures = 0U;
          bool unexpected_constraint = false;
          for (const auto &constraint_entry : restored_constraints) {
            const rtabmap::Link &link = constraint_entry.second;
            const bool incident =
                link.from() == node_id || link.to() == node_id;
            if (!incident) {
              continue;
            }
            const int peer_id =
                link.from() == node_id ? link.to() : link.from();
            const bool anchor_neighbor =
                link.type() == rtabmap::Link::kNeighbor &&
                ((link.from() == anchor_id && link.to() == node_id) ||
                 (link.from() == node_id && link.to() == anchor_id));
            const bool candidate_closure =
                link.type() == rtabmap::Link::kGlobalClosure &&
                ((link.from() == candidate_id && link.to() == node_id) ||
                 (link.from() == node_id && link.to() == candidate_id));
            if (anchor_neighbor) {
              ++anchor_neighbors;
            }
            if (candidate_closure) {
              ++candidate_closures;
            }
            bool valid_later_constraint = false;
            if (!anchor_neighbor && !candidate_closure &&
                peer_id > node_id && poses_.find(peer_id) != poses_.end()) {
              valid_later_constraint =
                  link.type() == rtabmap::Link::kNeighbor ||
                  link.type() == rtabmap::Link::kGlobalClosure ||
                  link.type() == rtabmap::Link::kLocalSpaceClosure ||
                  link.type() == rtabmap::Link::kLocalTimeClosure ||
                  link.type() == rtabmap::Link::kUserClosure;
            }
            if (!anchor_neighbor && !candidate_closure &&
                peer_id > 0 && peer_id < node_id &&
                link.type() == rtabmap::Link::kGlobalClosure) {
              QueryScope later_scope = QueryScope::kNone;
              std::uint64_t later_generation = 0;
              br::QueryOutcome later_outcome = br::QueryOutcome::kNone;
              std::vector<HeightPoint> no_height_points;
              QueryPromotionKind later_kind = QueryPromotionKind::kNone;
              int later_anchor_id = 0;
              int later_candidate_id = 0;
              const cv::Mat later_marker = link.uncompressUserDataConst();
              if (decode_persisted_query_data(
                      later_marker, &later_scope, &later_generation,
                      &later_outcome, &no_height_points, &later_kind,
                      &later_anchor_id, &later_candidate_id)) {
                const PersistedQueryKey decoded_later_key{
                    static_cast<int>(later_scope), later_generation};
                const auto terminal =
                    bridge_terminals.find(decoded_later_key);
                valid_later_constraint =
                    decoded_later_key != key &&
                    later_outcome ==
                        br::QueryOutcome::kBridgeOnlyDiscarded &&
                    later_kind == QueryPromotionKind::kBridgeOnly &&
                    terminal != bridge_terminals.end() &&
                    ((later_anchor_id == node_id &&
                      later_candidate_id == peer_id) ||
                     (later_candidate_id == node_id &&
                      later_anchor_id == peer_id));
              }
            }
            if ((!anchor_neighbor && !candidate_closure &&
                 !valid_later_constraint) ||
                peer_id <= 0 || poses_.find(peer_id) == poses_.end() ||
                !valid_persisted_query_link(link)) {
              unexpected_constraint = true;
            }
          }
          const bool positive = kind == QueryPromotionKind::kPositive;
          const bool negative = kind == QueryPromotionKind::kNegative;
          if ((!positive && !negative) || anchor_id <= 0 ||
              anchor_id >= node_id ||
              poses_.find(anchor_id) == poses_.end() ||
              (positive &&
               (candidate_id <= 0 ||
                candidate_id >= node_id || candidate_id == anchor_id ||
                poses_.find(candidate_id) == poses_.end())) ||
              anchor_id == node_id || (negative && candidate_id != 0) ||
              unexpected_constraint || anchor_neighbors != 1U ||
              candidate_closures != (positive ? 1U : 0U) ||
              bridge_terminals.find(key) != bridge_terminals.end() ||
              !committed_query_terminals.insert(key).second) {
            query_integrity_failed_ = true;
            throw std::runtime_error(
                "database committed query marker has invalid A-Q/C-Q topology");
          }
          height_points_[node_id] = registration_height_points.empty()
                                        ? height_points
                                        : std::move(registration_height_points);
          occupancy_height_points_[node_id] = std::move(height_points);
          query_footprint_ignored_ids_.insert(node_id);
          mark_query_generation_completed(scope, generation, outcome);
        }
      }
    }
    for (const auto &entry : bridge_terminals) {
      if (committed_query_terminals.find(entry.first) !=
          committed_query_terminals.end()) {
        query_integrity_failed_ = true;
        throw std::runtime_error(
            "database transaction contains both Q and A-C terminals");
      }
      mark_query_generation_completed(
          entry.second.scope, entry.second.generation,
          br::QueryOutcome::kBridgeOnlyDiscarded);
    }
    restore_graphless_query_terminals(restored_poses);
    if (last_mapping_node_id_ > 0) {
      node_odometry_history_index_[last_mapping_node_id_] = 0U;
    }
    rebuild_display_poses();
    grid_->setFootprintIgnoredIds(query_footprint_ignored_ids_);
    if (!display_poses_.empty() && !local_grids_.empty()) {
      grid_->update(display_poses_);
      refresh_mapping_cache();
    }
    if (trace_enabled()) {
      std::cerr << "persisted_state_restored poses=" << poses_.size()
                << " grids=" << local_grids_.size()
                << " query_nodes=" << query_footprint_ignored_ids_.size()
                << " constraints=" << restored_constraints.size() << "\n";
    }
  }

  void refresh_novelty_resume_protected_free(
      const cv::Mat &snapshot, float x_min, float y_min) {
    if (snapshot.empty() || snapshot.channels() != 1 ||
        (snapshot.depth() != CV_8S && snapshot.depth() != CV_8U) ||
        !std::isfinite(x_min) || !std::isfinite(y_min)) {
      throw std::runtime_error(
          "novelty resume free-space protection has an invalid snapshot");
    }
    const std::size_t snapshot_cells =
        static_cast<std::size_t>(snapshot.rows) *
        static_cast<std::size_t>(snapshot.cols);
    if (snapshot_cells > kNativeNoveltyResumeMaxProtectedCells) {
      throw std::runtime_error(
          "novelty resume free-space protection exceeds its raster cap");
    }

    // Grid origins are expressed in RTAB-Map's optimized map frame. A valid
    // loop closure may move that frame by an arbitrary sub-cell translation,
    // so masks from different graph revisions cannot be combined by integer
    // row/column offsets. The current cached raster is a complete rebuild of
    // all graph local grids; deriving the protection from it preserves every
    // still-free cell while following the latest graph correction.
    cv::Mat protected_free;
    cv::compare(snapshot, cv::Scalar(0), protected_free, cv::CMP_EQ);
    native_novelty_resume_protected_free_map_ = std::move(protected_free);
    native_novelty_resume_protected_free_x_min_ = x_min;
    native_novelty_resume_protected_free_y_min_ = y_min;
  }

  void arm_novelty_resume_occupancy_reconciliation(
      std::uint64_t frame_id, const rtabmap::Transform &anchor_map_pose) {
    // A novelty terminal releases the current frame only after the native
    // query transaction has been closed.  Keep a copy of the last committed
    // raster while independent ordinary mapping viewpoints reconcile the
    // transition. The copy is intentionally local to this worker: it is a
    // safety veto, not another persisted map or a second pose source.
    if (native_novelty_resume_reconciliation_pending_) {
      return;
    }
    if (!finite_planar_transform(anchor_map_pose) ||
        raw_qvel_history_.size() != odometry_history_.size()) {
      throw std::runtime_error(
          "novelty resume reconciliation has an invalid metric anchor");
    }
    native_novelty_resume_snapshot_map_ = cached_map_.clone();
    native_novelty_resume_snapshot_x_min_ = cached_x_min_;
    native_novelty_resume_snapshot_y_min_ = cached_y_min_;
    refresh_novelty_resume_protected_free(
        native_novelty_resume_snapshot_map_,
        native_novelty_resume_snapshot_x_min_,
        native_novelty_resume_snapshot_y_min_);
    native_novelty_resume_reconciliation_pending_ =
        !native_novelty_resume_snapshot_map_.empty();
    native_novelty_resume_reconciliation_viewpoints_remaining_ =
        native_novelty_resume_reconciliation_pending_
            ? kNativeNoveltyResumeReconciliationViewpoints
            : 0U;
    native_novelty_resume_anchor_map_pose_ = anchor_map_pose.to3DoF();
    native_novelty_resume_anchor_history_index_ = raw_qvel_history_.size();
    native_novelty_resume_obstacle_evidence_.clear();
    native_novelty_resume_frame_alignment_safe_ = false;
    native_novelty_resume_frame_alignment_supports_ = 0;
    native_novelty_resume_frame_alignment_max_bound_m_ = 0.0;
    if (native_novelty_resume_reconciliation_pending_) {
      if (!finite_planar_transform(fused_odom_pose_)) {
        throw std::runtime_error(
            "novelty resume reconciliation has an invalid anchor pose");
      }
      native_novelty_resume_last_progress_pose_ = fused_odom_pose_.to3DoF();
    } else {
      native_novelty_resume_last_progress_pose_.setNull();
    }
    if (trace_enabled()) {
      std::cerr << "native_novelty_resume_reconciliation_armed frame="
                << frame_id << " enabled="
                << (native_novelty_resume_reconciliation_pending_ ? 1 : 0)
                << " viewpoints="
                << native_novelty_resume_reconciliation_viewpoints_remaining_
                << " rows=" << native_novelty_resume_snapshot_map_.rows
                  << " cols=" << native_novelty_resume_snapshot_map_.cols
                  << " x_min=" << native_novelty_resume_snapshot_x_min_
                  << " y_min=" << native_novelty_resume_snapshot_y_min_
                  << " protected_free="
                  << cv::countNonZero(
                         native_novelty_resume_protected_free_map_)
                  << "\n";
    }
  }

  bool novelty_resume_relative_covariance(
      rtabmap::Transform *anchor_to_current,
      cv::Matx33d *covariance) const {
    if (anchor_to_current == nullptr || covariance == nullptr ||
        !finite_planar_transform(native_novelty_resume_anchor_map_pose_) ||
        native_novelty_resume_anchor_history_index_ >
            raw_qvel_history_.size() ||
        raw_qvel_history_.size() != odometry_history_.size()) {
      return false;
    }
    *anchor_to_current = rtabmap::Transform::getIdentity();
    *covariance = cv::Matx33d::zeros();
    for (std::size_t index = native_novelty_resume_anchor_history_index_;
         index < raw_qvel_history_.size(); ++index) {
      propagate_planar_odometry(
          raw_qvel_history_[index], anchor_to_current, covariance);
    }
    *covariance = (*covariance + covariance->t()) * 0.5;
    return finite_planar_transform(*anchor_to_current) &&
           cv::checkRange(cv::Mat(*covariance), true, nullptr);
  }

  static bool novelty_resume_obstacle_bound(
      const rtabmap::Transform &anchor_to_current,
      const cv::Matx33d &covariance,
      const rtabmap::Transform &expected_map_pose,
      const rtabmap::Transform &actual_map_pose,
      float local_x, float local_y, double *bound_m) {
    if (bound_m == nullptr ||
        !finite_planar_transform(anchor_to_current) ||
        !finite_planar_transform(expected_map_pose) ||
        !finite_planar_transform(actual_map_pose)) {
      return false;
    }
    cv::Matx22d point_covariance;
    double one_sigma = 0.0;
    if (!query_point_position_covariance(
            anchor_to_current, covariance, local_x, local_y,
            &point_covariance) ||
        !query_point_one_sigma(point_covariance, &one_sigma)) {
      return false;
    }
    const double expected_x =
        expected_map_pose.r11() * local_x +
        expected_map_pose.r12() * local_y + expected_map_pose.x();
    const double expected_y =
        expected_map_pose.r21() * local_x +
        expected_map_pose.r22() * local_y + expected_map_pose.y();
    const double actual_x =
        actual_map_pose.r11() * local_x +
        actual_map_pose.r12() * local_y + actual_map_pose.x();
    const double actual_y =
        actual_map_pose.r21() * local_x +
        actual_map_pose.r22() * local_y + actual_map_pose.y();
    const double nominal_disagreement =
        std::hypot(actual_x - expected_x, actual_y - expected_y);
    *bound_m =
        nominal_disagreement + 2.0 * one_sigma +
        std::sqrt(2.0) * static_cast<double>(kGridCellM);
    return std::isfinite(*bound_m) && *bound_m >= 0.0;
  }

  bool note_novelty_resume_obstacle_support(
      std::uint64_t cell_key,
      const rtabmap::Transform &anchor_to_current,
      bool *capacity_exhausted) {
    if (capacity_exhausted == nullptr ||
        !finite_planar_transform(anchor_to_current)) {
      return false;
    }
    *capacity_exhausted = false;
    auto found = native_novelty_resume_obstacle_evidence_.find(cell_key);
    if (found == native_novelty_resume_obstacle_evidence_.end()) {
      if (native_novelty_resume_obstacle_evidence_.size() >=
          kNativeNoveltyResumeMaxObstacleCandidateCells) {
        *capacity_exhausted = true;
        return false;
      }
      NoveltyResumeObstacleEvidence evidence;
      evidence.first_view_x = anchor_to_current.x();
      evidence.first_view_y = anchor_to_current.y();
      evidence.first_view_yaw = anchor_to_current.theta();
      native_novelty_resume_obstacle_evidence_.emplace(cell_key, evidence);
      return false;
    }
    if (found->second.confirmed) {
      return true;
    }
    if (novelty_resume_obstacle_views_are_independent(
            found->second.first_view_x, found->second.first_view_y,
            found->second.first_view_yaw, anchor_to_current)) {
      found->second.confirmed = true;
      return true;
    }
    return false;
  }

  bool novelty_resume_snapshot_cell_is_occupied(float world_x,
                                                 float world_y) const {
    if (native_novelty_resume_snapshot_map_.empty()) {
      return false;
    }
    const double column_value = std::floor(
        (static_cast<double>(world_x) -
         static_cast<double>(native_novelty_resume_snapshot_x_min_)) /
        static_cast<double>(kGridCellM));
    const double row_value = std::floor(
        (static_cast<double>(world_y) -
         static_cast<double>(native_novelty_resume_snapshot_y_min_)) /
        static_cast<double>(kGridCellM));
    if (!std::isfinite(column_value) || !std::isfinite(row_value)) {
      throw std::runtime_error(
          "novelty resume free-space filter has a non-finite snapshot cell");
    }
    if (column_value < 0.0 || row_value < 0.0 ||
        column_value >= native_novelty_resume_snapshot_map_.cols ||
        row_value >= native_novelty_resume_snapshot_map_.rows) {
      return false;
    }
    const int column = static_cast<int>(column_value);
    const int row = static_cast<int>(row_value);
    const int value =
        native_novelty_resume_snapshot_map_.depth() == CV_8S
            ? static_cast<int>(native_novelty_resume_snapshot_map_
                                   .at<std::int8_t>(row, column))
            : static_cast<int>(native_novelty_resume_snapshot_map_
                                   .at<std::uint8_t>(row, column));
    return value >= 65 && value <= 100;
  }

  bool novelty_resume_cell_was_traversed(float world_x, float world_y) const {
    const double global_column =
        std::floor(static_cast<double>(world_x) /
                   static_cast<double>(kGridCellM));
    const double global_row =
        std::floor(static_cast<double>(world_y) /
                   static_cast<double>(kGridCellM));
    if (!std::isfinite(global_column) || !std::isfinite(global_row) ||
        global_column < std::numeric_limits<std::int32_t>::min() ||
        global_column > std::numeric_limits<std::int32_t>::max() ||
        global_row < std::numeric_limits<std::int32_t>::min() ||
        global_row > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error(
          "novelty resume free-space filter has an invalid global cell");
    }
    return traversed_free_cells_.find(height_cell_key(
               static_cast<std::int32_t>(global_column),
               static_cast<std::int32_t>(global_row), 0U)) !=
           traversed_free_cells_.end();
  }

  cv::Mat filter_novelty_resume_free_layer(
      const cv::Mat &layer, const rtabmap::Transform &map_pose,
      std::size_t *vetoed) const {
    if (vetoed == nullptr) {
      throw std::runtime_error(
          "novelty resume free-space filter has no counter");
    }
    *vetoed = 0U;
    if (layer.empty() || !native_novelty_resume_reconciliation_pending_) {
      return layer;
    }
    if (layer.rows != 1 || layer.depth() != CV_32F ||
        layer.channels() < 2 || !layer.isContinuous()) {
      throw std::runtime_error(
          "novelty resume free-space filter has an unsupported layer");
    }
    const int channels = layer.channels();
    const float *values = layer.ptr<float>(0);
    std::vector<int> keep;
    keep.reserve(static_cast<std::size_t>(layer.cols));
    for (int column = 0; column < layer.cols; ++column) {
      const float local_x = values[column * channels];
      const float local_y = values[column * channels + 1];
      if (!std::isfinite(local_x) || !std::isfinite(local_y)) {
        throw std::runtime_error(
            "novelty resume free-space filter saw a non-finite point");
      }
      const float world_x =
          map_pose.r11() * local_x + map_pose.r12() * local_y + map_pose.x();
      const float world_y =
          map_pose.r21() * local_x + map_pose.r22() * local_y + map_pose.y();
      const bool clears_committed_wall =
          novelty_resume_snapshot_cell_is_occupied(world_x, world_y) &&
          !novelty_resume_cell_was_traversed(world_x, world_y);
      if (clears_committed_wall) {
        ++*vetoed;
      } else {
        keep.push_back(column);
      }
    }
    if (keep.size() == static_cast<std::size_t>(layer.cols)) {
      return layer;
    }
    cv::Mat filtered;
    if (!keep.empty()) {
      filtered.create(1, static_cast<int>(keep.size()), layer.type());
      for (std::size_t output = 0; output < keep.size(); ++output) {
        layer.col(keep[output]).copyTo(
            filtered.col(static_cast<int>(output)));
      }
    }
    return filtered;
  }

  void apply_novelty_resume_occupancy_filter(
      rtabmap::SensorData &data, const rtabmap::LaserScan &scan,
      const rtabmap::Transform &map_pose, std::uint64_t frame_id) {
    if (!native_novelty_resume_reconciliation_pending_ &&
        native_novelty_resume_protected_free_map_.empty() &&
        native_novelty_resume_snapshot_map_.empty()) {
      return;
    }
    native_novelty_resume_frame_alignment_safe_ = false;
    native_novelty_resume_frame_alignment_supports_ = 0;
    native_novelty_resume_frame_alignment_max_bound_m_ = 0.0;
    if (native_novelty_resume_snapshot_map_.empty()) {
      throw std::runtime_error(
          "novelty resume occupancy filter has no committed raster snapshot");
    }
    if (!finite_planar_transform(map_pose)) {
      throw std::runtime_error(
          "novelty resume occupancy filter has an invalid map pose");
    }
    if (scan.empty()) {
      // There cannot be a LaserScan-derived obstacle without a scan.  Leave
      // the SensorData untouched so RTAB-Map's normal empty-input behavior is
      // preserved; the pending latch is cleared only after a successful
      // ordinary mapping update below.
      if (trace_enabled()) {
        std::cerr << "native_novelty_resume_obstacle_veto frame=" << frame_id
                  << " total=0 filtered=0 protected=0 transition=0"
                  << " uncertainty=0 confirmation=0 capacity=0"
                  << " scan_empty=1\n";
      }
      return;
    }

    cv::Mat ground;
    cv::Mat obstacles;
    cv::Mat empty;
    cv::Point3f view_point(0.0F, 0.0F, 0.0F);
    if (!query_grid_maker_) {
      throw std::runtime_error(
          "novelty resume occupancy filter has no local grid maker");
    }
    query_grid_maker_->createLocalMap(
        scan, rtabmap::Transform::getIdentity(), ground, obstacles, empty,
        view_point);
    if (!query_layer_is_valid(ground) || !query_layer_is_valid(obstacles) ||
        !query_layer_is_valid(empty)) {
      throw std::runtime_error(
          "novelty resume occupancy filter produced an invalid local grid");
    }
    if (native_novelty_resume_snapshot_map_.channels() != 1 ||
        (native_novelty_resume_snapshot_map_.depth() != CV_8S &&
         native_novelty_resume_snapshot_map_.depth() != CV_8U) ||
        !std::isfinite(native_novelty_resume_snapshot_x_min_) ||
        !std::isfinite(native_novelty_resume_snapshot_y_min_)) {
      throw std::runtime_error(
          "novelty resume occupancy filter has an invalid raster snapshot");
    }
    if (!native_novelty_resume_protected_free_map_.empty() &&
        (native_novelty_resume_protected_free_map_.type() != CV_8UC1 ||
         !std::isfinite(native_novelty_resume_protected_free_x_min_) ||
         !std::isfinite(native_novelty_resume_protected_free_y_min_))) {
      throw std::runtime_error(
          "novelty resume occupancy filter has an invalid protected raster");
    }
    rtabmap::Transform anchor_to_current;
    cv::Matx33d anchor_covariance;
    if (!novelty_resume_relative_covariance(
            &anchor_to_current, &anchor_covariance)) {
      throw std::runtime_error(
          "novelty resume occupancy filter has an invalid odometry certificate");
    }
    const rtabmap::Transform expected_map_pose =
        (native_novelty_resume_anchor_map_pose_ * anchor_to_current).to3DoF();
    if (!finite_planar_transform(expected_map_pose)) {
      throw std::runtime_error(
          "novelty resume occupancy filter has an invalid expected map pose");
    }

    const int obstacle_channels = obstacles.empty() ? 0 : obstacles.channels();
    std::size_t vetoed = 0U;
    std::size_t protected_vetoed = 0U;
    std::size_t transition_vetoed = 0U;
    std::size_t uncertainty_vetoed = 0U;
    std::size_t confirmation_vetoed = 0U;
    std::size_t capacity_vetoed = 0U;
    std::size_t total = obstacles.empty()
                            ? 0U
                            : static_cast<std::size_t>(obstacles.cols);
    cv::Mat filtered_obstacles;
    if (!obstacles.empty()) {
      if (obstacles.rows != 1 || obstacles.depth() != CV_32F ||
          obstacle_channels < 2) {
        throw std::runtime_error(
            "novelty resume occupancy filter has an unsupported obstacle layer");
      }
      std::vector<int> keep;
      keep.reserve(static_cast<std::size_t>(obstacles.cols));
      const float *values = obstacles.ptr<float>(0);
      for (int column = 0; column < obstacles.cols; ++column) {
        const float local_x = values[column * obstacle_channels];
        const float local_y = values[column * obstacle_channels + 1];
        if (!std::isfinite(local_x) || !std::isfinite(local_y)) {
          throw std::runtime_error(
              "novelty resume occupancy filter saw a non-finite obstacle");
        }
        const float world_x =
            map_pose.r11() * local_x + map_pose.r12() * local_y + map_pose.x();
        const float world_y =
            map_pose.r21() * local_x + map_pose.r22() * local_y + map_pose.y();
        double support_bound_m = 0.0;
        if (!novelty_resume_obstacle_bound(
                anchor_to_current, anchor_covariance, expected_map_pose,
                map_pose, local_x, local_y, &support_bound_m)) {
          throw std::runtime_error(
              "novelty resume occupancy filter has an invalid support bound");
        }
        const bool support_safe =
            support_bound_m <=
            static_cast<double>(kQueryMaxPromotionUncertaintyM);
        const bool snapshot_occupied =
            novelty_resume_snapshot_cell_is_occupied(world_x, world_y);
        const int protected_column = static_cast<int>(std::floor(
            (world_x - native_novelty_resume_protected_free_x_min_) /
            kGridCellM));
        const int protected_row = static_cast<int>(std::floor(
            (world_y - native_novelty_resume_protected_free_y_min_) /
            kGridCellM));
        const bool protected_free =
            !native_novelty_resume_protected_free_map_.empty() &&
            protected_column >= 0 && protected_row >= 0 &&
            protected_column < native_novelty_resume_protected_free_map_.cols &&
            protected_row < native_novelty_resume_protected_free_map_.rows &&
            native_novelty_resume_protected_free_map_.at<std::uint8_t>(
                protected_row, protected_column) != 0U;
        const double global_column_value =
            std::floor(static_cast<double>(world_x) /
                       static_cast<double>(kGridCellM));
        const double global_row_value =
            std::floor(static_cast<double>(world_y) /
                       static_cast<double>(kGridCellM));
        if (!std::isfinite(global_column_value) ||
            !std::isfinite(global_row_value) ||
            global_column_value < std::numeric_limits<std::int32_t>::min() ||
            global_column_value > std::numeric_limits<std::int32_t>::max() ||
            global_row_value < std::numeric_limits<std::int32_t>::min() ||
            global_row_value > std::numeric_limits<std::int32_t>::max()) {
          throw std::runtime_error(
              "novelty resume occupancy filter has an invalid global cell");
        }
        bool obstacle_confirmed = false;
        bool capacity_exhausted = false;
        if (!native_novelty_resume_reconciliation_pending_ &&
            !protected_free && !snapshot_occupied && support_safe) {
          obstacle_confirmed = note_novelty_resume_obstacle_support(
              query_cell_key(static_cast<std::int32_t>(global_column_value),
                             static_cast<std::int32_t>(global_row_value)),
              anchor_to_current, &capacity_exhausted);
        }
        if (protected_free) {
          ++vetoed;
          ++protected_vetoed;
        } else if (snapshot_occupied) {
          keep.push_back(column);
          ++native_novelty_resume_frame_alignment_supports_;
          native_novelty_resume_frame_alignment_max_bound_m_ = std::max(
              native_novelty_resume_frame_alignment_max_bound_m_,
              support_bound_m);
        } else if (native_novelty_resume_reconciliation_pending_) {
          ++vetoed;
          ++transition_vetoed;
          if (!support_safe) {
            ++uncertainty_vetoed;
          } else if (capacity_exhausted) {
            ++capacity_vetoed;
          }
        } else if (!support_safe) {
          ++vetoed;
          ++uncertainty_vetoed;
        } else if (!obstacle_confirmed) {
          ++vetoed;
          if (capacity_exhausted) {
            ++capacity_vetoed;
          } else {
            ++confirmation_vetoed;
          }
        } else {
          keep.push_back(column);
        }
      }
      native_novelty_resume_frame_alignment_safe_ =
          native_novelty_resume_frame_alignment_supports_ > 0U &&
          native_novelty_resume_frame_alignment_max_bound_m_ <=
              static_cast<double>(kQueryMaxPromotionUncertaintyM);
      if (!keep.empty()) {
        filtered_obstacles.create(
            1, static_cast<int>(keep.size()), obstacles.type());
        for (std::size_t output = 0; output < keep.size(); ++output) {
          obstacles.col(keep[output]).copyTo(
              filtered_obstacles.col(static_cast<int>(output)));
        }
      }
    }
    std::size_t ground_vetoed = 0U;
    std::size_t empty_vetoed = 0U;
    const cv::Mat filtered_ground = filter_novelty_resume_free_layer(
        ground, map_pose, &ground_vetoed);
    const cv::Mat filtered_empty = filter_novelty_resume_free_layer(
        empty, map_pose, &empty_vetoed);
    data.setOccupancyGrid(filtered_ground, filtered_obstacles, filtered_empty,
                          kGridCellM, view_point);
    if (trace_enabled()) {
      std::cerr << "native_novelty_resume_obstacle_veto frame=" << frame_id
                << " total=" << total << " filtered=" << vetoed
                << " protected=" << protected_vetoed
                << " transition=" << transition_vetoed
                << " uncertainty=" << uncertainty_vetoed
                << " confirmation=" << confirmation_vetoed
                << " capacity=" << capacity_vetoed
                << " candidates="
                << native_novelty_resume_obstacle_evidence_.size()
                << " alignment_safe="
                << (native_novelty_resume_frame_alignment_safe_ ? 1 : 0)
                << " alignment_supports="
                << native_novelty_resume_frame_alignment_supports_
                << " alignment_max_bound_m="
                << native_novelty_resume_frame_alignment_max_bound_m_
                << " kept=" << (total - vetoed)
                << " free_total="
                << static_cast<std::size_t>(ground.cols + empty.cols)
                << " free_filtered=" << (ground_vetoed + empty_vetoed)
                << " ground_filtered=" << ground_vetoed
                << " empty_filtered=" << empty_vetoed << "\n";
    }
  }

  void refresh_mapping_cache() {
    if (!grid_) {
      return;
    }
    float x_min = 0.0F;
    float y_min = 0.0F;
    const cv::Mat map = grid_->getMap(x_min, y_min);
    if (map.empty()) {
      return;
    }
    cached_map_ = map.clone();
    cached_x_min_ = x_min;
    cached_y_min_ = y_min;
    cached_low_.assign(cached_map_.total(), 0);
    cached_high_.assign(cached_map_.total(), 0);
    assemble_height_layers(cached_low_, cached_high_, cached_map_,
                           cached_x_min_, cached_y_min_);
  }

  float observation_novelty_ratio(
      const FrameInput &frame, const rtabmap::Transform &pose,
      float *endpoint_ratio_out = nullptr, float *ray_ratio_out = nullptr) const {
    auto publish = [&](float combined, float endpoint, float ray) {
      if (endpoint_ratio_out != nullptr) {
        *endpoint_ratio_out = endpoint;
      }
      if (ray_ratio_out != nullptr) {
        *ray_ratio_out = ray;
      }
      return combined;
    };
    if (cached_map_.empty()) {
      return publish(1.0F, 1.0F, 1.0F);
    }
    if (frame.height_points.empty()) {
      // No structural endpoint is a weak observation, not evidence that the
      // current area is already covered. Keep it neutral until a usable frame
      // arrives instead of forcing a false positive or false negative.
      return publish(0.0F, 0.0F, 0.0F);
    }
    std::unordered_set<std::uint64_t> seen_endpoints;
    std::unordered_set<std::uint64_t> seen_rays;
    seen_endpoints.reserve(frame.height_points.size());
    seen_rays.reserve(frame.height_points.size() * 8U);
    std::size_t unknown_endpoints = 0;
    std::size_t endpoint_total = 0;
    std::size_t unknown_rays = 0;
    std::size_t ray_total = 0;
    for (const HeightPoint &point : frame.height_points) {
      const float world_x =
          pose.r11() * point.x + pose.r12() * point.y + pose.x();
      const float world_y =
          pose.r21() * point.x + pose.r22() * point.y + pose.y();
      const int column = static_cast<int>(std::floor(
          (world_x - cached_x_min_) / kGridCellM));
      const int row = static_cast<int>(std::floor(
          (world_y - cached_y_min_) / kGridCellM));
      const std::uint64_t endpoint_key = height_cell_key(column, row, 0);
      if (!seen_endpoints.insert(endpoint_key).second) {
        continue;
      }
      ++endpoint_total;
      if (column < 0 || row < 0 || column >= cached_map_.cols ||
          row >= cached_map_.rows ||
          cached_map_.at<std::int8_t>(row, column) < 0) {
        ++unknown_endpoints;
      }

      // Endpoint-only novelty misses a new room when its visible wall happens
      // to overlap a previously observed edge. Sample the compliant camera
      // ray through the free-space interior as well. These samples are used
      // only as a read-only freeze signal; they never alter the occupancy map.
      const float range = std::hypot(point.x, point.y);
      const float usable_range =
          range - kFreezeRayStartM - kFreezeRayEndBackoffM;
      const int steps = usable_range > 0.0F
                            ? static_cast<int>(std::floor(
                                  usable_range / kFreezeRaySampleStepM))
                            : 0;
      for (int step = 0; step < steps; ++step) {
        const float distance =
            kFreezeRayStartM +
            (static_cast<float>(step) + 0.5F) * kFreezeRaySampleStepM;
        const float fraction = distance / range;
        const float local_x = point.x * fraction;
        const float local_y = point.y * fraction;
        const float ray_world_x =
            pose.r11() * local_x + pose.r12() * local_y + pose.x();
        const float ray_world_y =
            pose.r21() * local_x + pose.r22() * local_y + pose.y();
        const int ray_column = static_cast<int>(std::floor(
            (ray_world_x - cached_x_min_) / kGridCellM));
        const int ray_row = static_cast<int>(std::floor(
            (ray_world_y - cached_y_min_) / kGridCellM));
        const std::uint64_t ray_key =
            height_cell_key(ray_column, ray_row, 1);
        if (!seen_rays.insert(ray_key).second) {
          continue;
        }
        ++ray_total;
        if (ray_column < 0 || ray_row < 0 ||
            ray_column >= cached_map_.cols || ray_row >= cached_map_.rows ||
            cached_map_.at<std::int8_t>(ray_row, ray_column) < 0) {
          ++unknown_rays;
        }
      }
    }
    const float endpoint_ratio =
        endpoint_total == 0
            ? 0.0F
            : static_cast<float>(unknown_endpoints) /
                  static_cast<float>(endpoint_total);
    const float ray_ratio =
        ray_total == 0
            ? 0.0F
            : static_cast<float>(unknown_rays) / static_cast<float>(ray_total);
    return publish(std::max(endpoint_ratio, ray_ratio), endpoint_ratio,
                   ray_ratio);
  }

  CoverageSample measure_coverage() const {
    CoverageSample sample;
    sample.mapping_frame = mapping_frames_;
    sample.travel_m = mapping_travel_m_;
    sample.rotation_rad = mapping_rotation_rad_;
    sample.observation_novelty_ratio = last_observation_novelty_ratio_;
    sample.observation_endpoint_novelty_ratio =
        last_observation_endpoint_novelty_ratio_;
    sample.observation_ray_novelty_ratio = last_observation_ray_novelty_ratio_;
    sample.graph_poses = poses_;
    if (cached_map_.empty()) {
      return sample;
    }
    const cv::Mat &map = cached_map_;
    sample.known_cells = static_cast<std::size_t>(cv::countNonZero(map >= 0));
    for (int row = 0; row < map.rows; ++row) {
      const std::int8_t *values = map.ptr<std::int8_t>(row);
      for (int column = 0; column < map.cols; ++column) {
        if (values[column] < 0) {
          continue;
        }
        const bool unknown_neighbor =
            (row > 0 && map.at<std::int8_t>(row - 1, column) < 0) ||
            (row + 1 < map.rows && map.at<std::int8_t>(row + 1, column) < 0) ||
            (column > 0 && values[column - 1] < 0) ||
            (column + 1 < map.cols && values[column + 1] < 0);
        if (!unknown_neighbor) {
          continue;
        }
        ++sample.boundary_cells;
        // 只有与未知区相邻的自由空间才是尚未闭合的开放边界。占据边界已经由
        // 观测到的结构封住，不能与开放边界一起计入分子。
        sample.frontier_cells += values[column] == 0 ? 1U : 0U;
      }
    }
    return sample;
  }

  static GraphShapeChange compare_graph_shape(
      const CoverageSample &reference, const CoverageSample &current) {
    GraphShapeChange change;
    std::vector<int> common_ids;
    common_ids.reserve(
        std::min(reference.graph_poses.size(), current.graph_poses.size()));
    for (const auto &entry : reference.graph_poses) {
      const auto current_pose = current.graph_poses.find(entry.first);
      if (!entry.second.isNull() && current_pose != current.graph_poses.end() &&
          !current_pose->second.isNull()) {
        common_ids.push_back(entry.first);
      }
    }
    change.common_nodes = common_ids.size();
    if (common_ids.size() < kFreezeGraphMinCommonNodes) {
      return change;
    }

    // 用最新的共同节点固定两份图的规范自由度，再比较所有共同节点的相对
    // 几何。这样坐标原点整体平移或旋转不会被误判为地图形变。
    const int anchor_id = common_ids.back();
    const rtabmap::Transform reference_anchor =
        reference.graph_poses.at(anchor_id).to3DoF();
    const rtabmap::Transform current_anchor =
        current.graph_poses.at(anchor_id).to3DoF();
    std::vector<float> translations;
    std::vector<float> yaws;
    translations.reserve(common_ids.size());
    yaws.reserve(common_ids.size());
    for (const int id : common_ids) {
      const rtabmap::Transform reference_relative =
          (reference_anchor.inverse() * reference.graph_poses.at(id)).to3DoF();
      const rtabmap::Transform current_relative =
          (current_anchor.inverse() * current.graph_poses.at(id)).to3DoF();
      const rtabmap::Transform delta =
          (reference_relative.inverse() * current_relative).to3DoF();
      translations.push_back(std::hypot(delta.x(), delta.y()));
      yaws.push_back(std::abs(std::atan2(
          std::sin(delta.theta()), std::cos(delta.theta()))));
    }
    const auto percentile = [](std::vector<float> values) {
      const std::size_t index = static_cast<std::size_t>(std::floor(
          kFreezeGraphPercentile * static_cast<float>(values.size() - 1U)));
      std::nth_element(values.begin(), values.begin() + index, values.end());
      return values[index];
    };
    change.translation_m = percentile(std::move(translations));
    change.yaw_rad = percentile(std::move(yaws));
    change.valid = true;
    return change;
  }

  bool note_visual_revisit(int reference_id, int closure_id,
                           const rtabmap::Transform &pose) {
    int left = std::min(reference_id, closure_id);
    int right = std::max(reference_id, closure_id);
    for (const auto &accepted : accepted_loop_pairs_) {
      if (std::abs(left - accepted.first) <= kFreezeLoopPairDedupNodes &&
          std::abs(right - accepted.second) <= kFreezeLoopPairDedupNodes) {
        return false;
      }
    }
    accepted_loop_pairs_.insert({left, right});
    visual_loop_regions_.insert(
        {static_cast<int>(std::floor(pose.x() / kFreezeLoopRegionM)),
         static_cast<int>(std::floor(pose.y() / kFreezeLoopRegionM))});
    return true;
  }

  ConvergenceEvidence update_convergence_evidence() {
    const CoverageSample current = measure_coverage();
    coverage_history_.push_back(current);
    while (coverage_history_.size() > 1 &&
           coverage_history_.front().mapping_frame +
                   kFreezeEvidenceWindowFrames <
               current.mapping_frame) {
      coverage_history_.pop_front();
    }

    ConvergenceEvidence evidence;
    evidence.known_cells = current.known_cells;
    evidence.boundary_cells = current.boundary_cells;
    evidence.frontier_cells = current.frontier_cells;
    evidence.observation_novelty_ratio = current.observation_novelty_ratio;
    evidence.observation_endpoint_novelty_ratio =
        current.observation_endpoint_novelty_ratio;
    evidence.observation_ray_novelty_ratio =
        current.observation_ray_novelty_ratio;
    while (!recent_visual_revisit_frames_.empty() &&
           recent_visual_revisit_frames_.front() +
                   kFreezeVisualRevisitWindowFrames <
               mapping_frames_) {
      recent_visual_revisit_frames_.pop_front();
    }
    evidence.recent_visual_revisits = recent_visual_revisit_frames_.size();
    evidence.accepted_loop_events = accepted_loop_pairs_.size();
    evidence.visual_loop_regions = visual_loop_regions_.size();
    const int graph_first_id = poses_.empty() ? 0 : poses_.begin()->first;
    const int graph_last_id = poses_.empty() ? 0 : poses_.rbegin()->first;
    const int graph_id_span = std::max(1, graph_last_id - graph_first_id);
    for (const auto &[left, right] : accepted_loop_pairs_) {
      evidence.max_loop_node_span_ratio = std::max(
          evidence.max_loop_node_span_ratio,
          static_cast<float>(right - left) /
              static_cast<float>(graph_id_span));
      const auto left_motion = node_motion_progress_m_.find(left);
      const auto right_motion = node_motion_progress_m_.find(right);
      const float mapping_motion_progress =
          mapping_travel_m_ +
          kFreezeRotationEquivalentRadiusM * mapping_rotation_rad_;
      if (mapping_motion_progress > 0.0F &&
          left_motion != node_motion_progress_m_.end() &&
          right_motion != node_motion_progress_m_.end()) {
        evidence.max_loop_motion_span_ratio = std::max(
            evidence.max_loop_motion_span_ratio,
            std::abs(right_motion->second - left_motion->second) /
                mapping_motion_progress);
      }
    }
    evidence.frontier_ratio =
        current.known_cells == 0
            ? std::numeric_limits<float>::infinity()
            : static_cast<float>(current.frontier_cells) /
                  static_cast<float>(current.known_cells);
    if (!coverage_history_.empty()) {
      const CoverageSample &oldest = coverage_history_.front();
      evidence.window_complete =
          current.mapping_frame >= oldest.mapping_frame +
                                       kFreezeEvidenceWindowFrames;
      // 图优化后已知区域突然缩小同样说明地图尚未稳定，不能把它当成零增长
      // 而误触发冻结。字段名为兼容已有报告保留，实际记录窗口两端的对称变化率。
      const std::size_t known_change =
          std::max(current.known_cells, oldest.known_cells) -
          std::min(current.known_cells, oldest.known_cells);
      evidence.known_growth_ratio =
          std::max(current.known_cells, oldest.known_cells) == 0
              ? std::numeric_limits<float>::infinity()
              : static_cast<float>(known_change) /
                    static_cast<float>(
                        std::max(current.known_cells, oldest.known_cells));
    }

    const CoverageSample *recent_reference = nullptr;
    for (auto iter = coverage_history_.rbegin();
         iter != coverage_history_.rend(); ++iter) {
      if (current.mapping_frame >=
          iter->mapping_frame + kFreezeRecentGrowthWindowFrames) {
        recent_reference = &*iter;
        break;
      }
    }
    if (recent_reference != nullptr) {
      evidence.recent_window_complete = true;
      const std::size_t recent_change =
          std::max(current.known_cells, recent_reference->known_cells) -
          std::min(current.known_cells, recent_reference->known_cells);
      evidence.recent_known_growth_ratio =
          std::max(current.known_cells, recent_reference->known_cells) == 0
              ? std::numeric_limits<float>::infinity()
              : static_cast<float>(recent_change) /
                    static_cast<float>(std::max(
                        current.known_cells, recent_reference->known_cells));
    }

    const CoverageSample *graph_reference = nullptr;
    for (const CoverageSample &sample : coverage_history_) {
      if (current.mapping_frame >=
          sample.mapping_frame + kFreezeGraphQuietFrames) {
        graph_reference = &sample;
      } else {
        break;
      }
    }
    if (graph_reference != nullptr) {
      const GraphShapeChange graph_change =
          compare_graph_shape(*graph_reference, current);
      evidence.graph_window_complete = graph_change.valid;
      evidence.graph_window_translation_m = graph_change.translation_m;
      evidence.graph_window_yaw_rad = graph_change.yaw_rad;
      evidence.graph_common_nodes = graph_change.common_nodes;
    }

    const bool loop_settled =
        last_visual_revisit_frame_ > 0 &&
        mapping_frames_ >=
            last_visual_revisit_frame_ + kFreezeLoopSettleFrames;
    evidence.ready =
        current.known_cells > 0 &&
        mapping_has_revisit_evidence(
            visual_loop_regions_.size(), accepted_loop_pairs_.size()) &&
        loop_spans_map(evidence.max_loop_node_span_ratio,
                       evidence.max_loop_motion_span_ratio) &&
        evidence.window_complete && evidence.recent_window_complete &&
        evidence.graph_window_complete &&
        loop_settled &&
        // The long window is retained for diagnostics, but it necessarily
        // includes the initial exploration burst.  Freeze must be gated by
        // the complete recent window so growth that has actually stopped can
        // be recognized without a hand-tuned episode frame.
        evidence.recent_known_growth_ratio <= kFreezeMaxKnownGrowthRatio &&
        evidence.frontier_ratio <= kFreezeMaxFrontierRatio &&
        evidence.observation_novelty_ratio <=
            kFreezeMaxObservationNoveltyRatio &&
        evidence.graph_window_translation_m <= kFreezeGraphCorrectionM &&
        evidence.graph_window_yaw_rad <= kFreezeGraphCorrectionRad;
    return evidence;
  }

  void reset_soft_novelty_evidence() {
    read_only_revisit_unknown_streak_ = 0;
    soft_novelty_start_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    soft_novelty_anchor_pose_.setNull();
    soft_novelty_last_view_pose_.setNull();
    soft_novelty_max_translation_m_ = 0.0F;
    soft_novelty_distinct_viewpoints_ = 0;
  }

  void note_soft_novelty_evidence(
      float novelty, const rtabmap::Transform &pose, double stamp,
      bool observed_static) {
    if (!std::isfinite(novelty) ||
        novelty <= kReadOnlyRevisitMaxNoveltyRatio) {
      reset_soft_novelty_evidence();
      return;
    }
    if (read_only_revisit_unknown_streak_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++read_only_revisit_unknown_streak_;
    }
    if (!std::isfinite(soft_novelty_start_stamp_s_)) {
      soft_novelty_start_stamp_s_ = stamp;
      soft_novelty_anchor_pose_ = pose.to3DoF();
      soft_novelty_last_view_pose_ = soft_novelty_anchor_pose_;
      soft_novelty_max_translation_m_ = 0.0F;
      soft_novelty_distinct_viewpoints_ = 1;
      return;
    }
    if (observed_static || soft_novelty_anchor_pose_.isNull() ||
        soft_novelty_last_view_pose_.isNull()) {
      return;
    }
    const rtabmap::Transform anchor_delta =
        (soft_novelty_anchor_pose_.inverse() * pose).to3DoF();
    soft_novelty_max_translation_m_ = std::max(
        soft_novelty_max_translation_m_,
        std::hypot(anchor_delta.x(), anchor_delta.y()));
    const rtabmap::Transform delta =
        (soft_novelty_last_view_pose_.inverse() * pose).to3DoF();
    if (std::hypot(delta.x(), delta.y()) < kSoftNoveltyViewTranslationM &&
        std::abs(delta.theta()) < kSoftNoveltyViewYawRad) {
      return;
    }
    soft_novelty_last_view_pose_ = pose.to3DoF();
    if (soft_novelty_distinct_viewpoints_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++soft_novelty_distinct_viewpoints_;
    }
  }

  bool soft_novelty_ready(double stamp) const {
    const double duration_s = std::isfinite(soft_novelty_start_stamp_s_)
                                  ? stamp - soft_novelty_start_stamp_s_
                                  : 0.0;
    return soft_novelty_can_resume_mapping(
        read_only_revisit_unknown_streak_,
        soft_novelty_distinct_viewpoints_,
        soft_novelty_max_translation_m_, duration_s);
  }

  void enter_soft_localization_mode(std::uint64_t frame_id) {
    // Keep the one RTAB-Map instance incremental for the whole session. A
    // true Mem/IncrementalMemory transition increments RTAB-Map's map id and a
    // later mapping resume can lose the sequential neighbor edge. Soft freeze
    // therefore suppresses process() while reusing the existing historical
    // scan matcher; it never reparses RTAB parameters or snapshots a second
    // map representation.
    soft_mapping_state_ = SoftMappingState::kKnownLocalizing;
    read_only_revisit_active_ = true;
    // The convergence transition starts at the last mapped pose, so local
    // place identity is known by continuous odometry even without a new
    // external appearance proposal on this exact frame.
    read_only_revisit_identity_verified_ = true;
    read_only_revisit_candidate_id_ = 0;
    read_only_revisit_start_travel_m_ = session_motion_progress_m_;
    read_only_revisit_no_observation_streak_ = 0;
    reset_soft_novelty_evidence();
    normal_global_probe_complete_ = false;
    read_only_candidate_active_ = false;
    read_only_candidate_start_motion_m_ = 0.0F;
    read_only_candidate_unknown_streak_ = 0;
    read_only_candidate_no_observation_streak_ = 0;
    read_only_scan_match_failures_ = 0;
    read_only_scan_match_next_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    clear_read_only_scan_match_proposal();
    localized_this_frame_ = false;
    visual_localized_this_frame_ = false;
    geometric_localized_this_frame_ = false;
    if (trace_enabled()) {
      std::cerr << "mapping_soft_frozen frame=" << frame_id
                << " nodes=" << poses_.size()
                << " travel=" << mapping_travel_m_
                << " known=" << convergence_.known_cells
                << " frontier_ratio=" << convergence_.frontier_ratio
                << " novelty=" << convergence_.observation_novelty_ratio
                << " map_id_preserved=1\n";
    }
  }

  void enter_native_revisit_hold(std::uint64_t frame_id, int candidate_id) {
    // RTAB-Map has just accepted an appearance-backed metric constraint to an
    // independently visited part of the graph. Keep that graph correction,
    // but make the next observation read-only. Continuing incremental writes
    // in the confirmed old region creates correlated duplicate nodes before a
    // later closure can repair them, which thickens and forks wall rasters.
    soft_mapping_state_ = SoftMappingState::kKnownLocalizing;
    read_only_revisit_active_ = true;
    read_only_revisit_identity_verified_ = true;
    read_only_revisit_candidate_id_ = candidate_id;
    read_only_revisit_start_travel_m_ = session_motion_progress_m_;
    read_only_revisit_no_observation_streak_ = 0;
    reset_soft_novelty_evidence();
    normal_global_probe_complete_ = false;
    read_only_candidate_active_ = false;
    read_only_candidate_start_motion_m_ = 0.0F;
    read_only_candidate_unknown_streak_ = 0;
    read_only_candidate_no_observation_streak_ = 0;
    read_only_scan_match_failures_ = 0;
    read_only_scan_match_next_stamp_s_ =
        -std::numeric_limits<double>::infinity();
    clear_read_only_scan_match_proposal();
    if (read_only_revisit_count_ <
        std::numeric_limits<std::uint32_t>::max()) {
      ++read_only_revisit_count_;
    }
    if (trace_enabled()) {
      std::cerr << "native_revisit_hold_started frame=" << frame_id
                << " candidate=" << candidate_id
                << " nodes=" << poses_.size()
                << " loops=" << loop_count_
                << " travel=" << mapping_travel_m_ << "\n";
    }
  }

  void note_candidate_novelty() {
    if (freeze_candidate_start_frame_ == 0) {
      return;
    }
    freeze_candidate_max_novelty_ratio_ = std::max(
        freeze_candidate_max_novelty_ratio_, last_observation_novelty_ratio_);
  }

  void consider_freeze_candidate(std::uint64_t frame_id) {
    if (mode_ != MappingMode::kMapping ||
        soft_mapping_state_ != SoftMappingState::kBuilding) {
      return;
    }
    if (freeze_candidate_start_frame_ == 0) {
      if (!convergence_.ready) {
        return;
      }
      freeze_candidate_start_frame_ = mapping_frames_;
      freeze_candidate_motion_progress_m_ = mapping_travel_m_;
      freeze_candidate_known_cells_ = convergence_.known_cells;
      freeze_candidate_revisit_count_ = accepted_loop_pairs_.size();
      freeze_candidate_max_novelty_ratio_ =
          convergence_.observation_novelty_ratio;
      if (trace_enabled()) {
        std::cerr << "mapping_freeze_candidate frame=" << frame_id
                  << " mapping_frames=" << mapping_frames_
                  << " known=" << freeze_candidate_known_cells_
                  << " revisits=" << freeze_candidate_revisit_count_
                  << " novelty=" << freeze_candidate_max_novelty_ratio_
                  << " hold_frames=" << kFreezeCandidateHoldFrames << "\n";
      }
      return;
    }

    // A candidate may span ordinary loop-closure observations.  Cancel only
    // when the frozen-map hypothesis is contradicted by measurable novelty or
    // an unstable graph, rather than because the loop-settle counter briefly
    // becomes false after a valid revisit.
    const std::size_t current_known = convergence_.known_cells;
    const std::size_t maximum_known =
        std::max(current_known, freeze_candidate_known_cells_);
    const std::size_t minimum_known =
        std::min(current_known, freeze_candidate_known_cells_);
    const float candidate_growth_ratio =
        maximum_known == 0
            ? std::numeric_limits<float>::infinity()
            : static_cast<float>(maximum_known - minimum_known) /
                  static_cast<float>(maximum_known);
    const bool graph_unstable =
        !convergence_.graph_window_complete ||
        convergence_.graph_window_translation_m > kFreezeGraphCorrectionM ||
        convergence_.graph_window_yaw_rad > kFreezeGraphCorrectionRad;
    const bool post_candidate_revisit =
        accepted_loop_pairs_.size() > freeze_candidate_revisit_count_;
    const bool quiet_completion =
        mapping_frames_ - freeze_candidate_start_frame_ >=
            kFreezeCandidateQuietCompletionFrames;
    const bool candidate_invalid =
        !convergence_.recent_window_complete ||
        candidate_growth_ratio > kFreezeMaxKnownGrowthRatio ||
        convergence_.frontier_ratio > kFreezeMaxFrontierRatio ||
        freeze_candidate_max_novelty_ratio_ > kFreezeMaxObservationNoveltyRatio ||
        graph_unstable;
    if (candidate_invalid) {
      if (trace_enabled()) {
        std::cerr << "mapping_freeze_candidate_cancelled frame=" << frame_id
                  << " start_frame=" << freeze_candidate_start_frame_
                  << " known=" << current_known
                  << " known_growth=" << candidate_growth_ratio
                  << " frontier_ratio=" << convergence_.frontier_ratio
                  << " observation_novelty="
                  << convergence_.observation_novelty_ratio
                  << " novelty_peak=" << freeze_candidate_max_novelty_ratio_
                  << " post_candidate_revisit="
                  << (accepted_loop_pairs_.size() > freeze_candidate_revisit_count_ ? 1 : 0)
                  << " graph_translation="
                  << convergence_.graph_window_translation_m
                  << " graph_yaw=" << convergence_.graph_window_yaw_rad << "\n";
      }
      freeze_candidate_start_frame_ = 0;
      freeze_candidate_motion_progress_m_ = 0.0F;
      freeze_candidate_known_cells_ = 0;
      freeze_candidate_revisit_count_ = 0;
      freeze_candidate_max_novelty_ratio_ = 1.0F;
      return;
    }
    const bool hold_complete =
        mapping_frames_ - freeze_candidate_start_frame_ >=
        kFreezeCandidateHoldFrames;
    const float current_motion_progress = mapping_travel_m_;
    const float observed_path_progress = std::max(
        current_motion_progress, freeze_candidate_motion_progress_m_);
    const float candidate_motion_budget = std::max(
        kMaxRangeM * kFreezeCandidateMotionRangeFraction,
        observed_path_progress * kFreezeCandidateMotionPathFraction);
    const bool candidate_motion_complete =
        current_motion_progress - freeze_candidate_motion_progress_m_ >=
        candidate_motion_budget;
    // The candidate may span a newly accepted loop, but the transition itself
    // must be evaluated against the latest convergence sample.  In
    // particular, do not freeze while that loop's graph correction is still
    // settling: the candidate hold is a debounce interval, not permission to
    // bypass the ready gate that started it.
    const bool ready_at_switch = convergence_.ready;
    // A fresh native revisit completes the normal debounce path.  If RTAB-Map
    // deduplicates later links, an uninterrupted interval twice as long is an
    // equivalent completion signal: candidate_invalid above has already
    // enforced low coverage growth, frontier, observation novelty and graph
    // motion throughout that interval.
    // A fresh independent loop is already a new viewpoint challenge. Without
    // one, quiet frames prove convergence only after the sensor footprint has
    // actually translated: a manipulation pause or panoramic turn at one pose
    // cannot establish that an unseen branch does not exist. The budget is
    // derived from depth range and accumulated translational path scale, so it
    // is independent of task, room dimensions, frame rate and route.
    if (!freeze_candidate_can_switch(hold_complete, post_candidate_revisit,
                                     quiet_completion,
                                     candidate_motion_complete,
                                     ready_at_switch)) {
      if (trace_enabled() && hold_complete &&
          (post_candidate_revisit || quiet_completion) && !ready_at_switch) {
        std::cerr << "mapping_freeze_candidate_waiting frame=" << frame_id
                  << " start_frame=" << freeze_candidate_start_frame_
                  << " reason=convergence_not_ready"
                  << " loop_settle_frames="
                  << (mapping_frames_ - last_visual_revisit_frame_)
                  << " graph_translation="
                  << convergence_.graph_window_translation_m
                  << " graph_yaw=" << convergence_.graph_window_yaw_rad
                  << " recent_visual_revisits="
                  << convergence_.recent_visual_revisits << "\n";
      }
      if (trace_enabled() && hold_complete && !candidate_motion_complete) {
        std::cerr << "mapping_freeze_candidate_motion_wait frame=" << frame_id
                  << " start_frame=" << freeze_candidate_start_frame_
                  << " motion_progress=" << current_motion_progress
                  << " start_motion_progress="
                  << freeze_candidate_motion_progress_m_
                  << " required_motion=" << candidate_motion_budget << "\n";
      }
      return;
    }
    if (trace_enabled()) {
      std::cerr << "mapping_freeze_candidate_complete frame=" << frame_id
                << " start_frame=" << freeze_candidate_start_frame_
                << " completion="
                << (post_candidate_revisit ? "native_revisit"
                                           : "extended_quiet")
                << "\n";
    }
    freeze_candidate_start_frame_ = 0;
    freeze_candidate_motion_progress_m_ = 0.0F;
    freeze_candidate_known_cells_ = 0;
    freeze_candidate_revisit_count_ = 0;
    freeze_candidate_max_novelty_ratio_ = 1.0F;
    enter_soft_localization_mode(frame_id);
  }

  void rebuild_display_poses() {
    // OccupancyGrid::update() 必须使用优化后的图位姿，否则会丢掉闭环修正。
    bool changed = display_poses_.size() != poses_.size();
    if (!changed) {
      auto previous = display_poses_.begin();
      auto current = poses_.begin();
      for (; current != poses_.end(); ++current, ++previous) {
        if (current->first != previous->first ||
            current->second.isNull() != previous->second.isNull() ||
            (!current->second.isNull() &&
             (current->second.x() != previous->second.x() ||
              current->second.y() != previous->second.y() ||
              current->second.theta() != previous->second.theta()))) {
          changed = true;
          break;
        }
      }
    }
    if (!changed) {
      return;
    }
    display_poses_ = poses_;
    rebuild_traversed_free_cells();
  }

  void protect_traversed_segment(float first_x, float first_y,
                                 float second_x, float second_y) {
    const float minimum_x =
        std::min(first_x, second_x) - kTraversedFreeCellRadiusM;
    const float maximum_x =
        std::max(first_x, second_x) + kTraversedFreeCellRadiusM;
    const float minimum_y =
        std::min(first_y, second_y) - kTraversedFreeCellRadiusM;
    const float maximum_y =
        std::max(first_y, second_y) + kTraversedFreeCellRadiusM;
    const int first_column =
        static_cast<int>(std::floor(minimum_x / kGridCellM));
    const int last_column =
        static_cast<int>(std::floor(maximum_x / kGridCellM));
    const int first_row =
        static_cast<int>(std::floor(minimum_y / kGridCellM));
    const int last_row =
        static_cast<int>(std::floor(maximum_y / kGridCellM));
    const float delta_x = second_x - first_x;
    const float delta_y = second_y - first_y;
    const float length_squared = delta_x * delta_x + delta_y * delta_y;
    const float radius_squared =
        kTraversedFreeCellRadiusM * kTraversedFreeCellRadiusM;
    for (int row = first_row; row <= last_row; ++row) {
      for (int column = first_column; column <= last_column; ++column) {
        const float cell_x =
            (static_cast<float>(column) + 0.5F) * kGridCellM;
        const float cell_y =
            (static_cast<float>(row) + 0.5F) * kGridCellM;
        float fraction = 0.0F;
        if (length_squared > 1.0e-12F) {
          fraction = std::clamp(
              ((cell_x - first_x) * delta_x +
               (cell_y - first_y) * delta_y) /
                  length_squared,
              0.0F, 1.0F);
        }
        const float nearest_x = first_x + fraction * delta_x;
        const float nearest_y = first_y + fraction * delta_y;
        const float error_x = cell_x - nearest_x;
        const float error_y = cell_y - nearest_y;
        if (error_x * error_x + error_y * error_y <= radius_squared) {
          traversed_free_cells_.insert(height_cell_key(column, row, 0U));
        }
      }
    }
  }

  void rebuild_traversed_free_cells() {
    traversed_free_cells_.clear();
    bool have_previous = false;
    float previous_x = 0.0F;
    float previous_y = 0.0F;
    for (const auto &entry : display_poses_) {
      if (entry.first <= 0 || entry.second.isNull() ||
          !finite_planar_transform(entry.second)) {
        continue;
      }
      const float current_x = entry.second.x();
      const float current_y = entry.second.y();
      protect_traversed_segment(current_x, current_y, current_x, current_y);
      if (have_previous &&
          std::hypot(current_x - previous_x, current_y - previous_y) <=
              kTraversedFreeMaxNodeStepM) {
        protect_traversed_segment(previous_x, previous_y, current_x,
                                  current_y);
      }
      previous_x = current_x;
      previous_y = current_y;
      have_previous = true;
    }
  }

  bool refresh_global_graph_poses(
      int current_id, const rtabmap::Transform &target_current_pose,
      int requested_id = 0,
      rtabmap::Transform *requested_pose = nullptr) {
    if (!slam_) {
      return false;
    }
    std::map<int, rtabmap::Transform> global_poses;
    std::multimap<int, rtabmap::Link> global_constraints;
    slam_->getGraph(global_poses, global_constraints, true, true);
    if (global_poses.empty()) {
      if (trace_enabled()) {
        std::cerr << "global_graph_refresh_empty current_id=" << current_id
                  << "\n";
      }
      return false;
    }

    // getGraph(..., global=true) optimizes the complete graph and may choose a
    // different but equivalent SE(2) gauge. Preserve the oldest already
    // published graph node as the map-frame datum. Anchoring at the current
    // observation instead cancels that observation's loop correction by
    // rigidly rotating/translating all historical occupancy around it.
    rtabmap::Transform gauge = rtabmap::Transform::getIdentity();
    bool anchored = false;
    int gauge_anchor_id = 0;
    for (const auto &old_entry : poses_) {
      if (old_entry.first <= 0 || old_entry.second.isNull()) {
        continue;
      }
      const auto candidate = global_poses.find(old_entry.first);
      if (candidate != global_poses.end() && !candidate->second.isNull()) {
        gauge = (old_entry.second * candidate->second.inverse()).to3DoF();
        gauge_anchor_id = old_entry.first;
        anchored = true;
        break;
      }
    }
    if (!anchored) {
      const auto current = global_poses.find(current_id);
      if (current != global_poses.end() && !current->second.isNull() &&
          !target_current_pose.isNull()) {
        gauge = (target_current_pose * current->second.inverse()).to3DoF();
        gauge_anchor_id = current_id;
        anchored = true;
      }
    }

    std::map<int, rtabmap::Transform> refreshed;
    for (const auto &entry : global_poses) {
      if (entry.first <= 0 || entry.second.isNull()) {
        continue;
      }
      refreshed[entry.first] = (gauge * entry.second).to3DoF();
    }
    if (refreshed.empty()) {
      return false;
    }
    if (requested_id > 0) {
      const auto requested = refreshed.find(requested_id);
      if (requested_pose == nullptr || requested == refreshed.end() ||
          !finite_planar_transform(requested->second)) {
        return false;
      }
      *requested_pose = requested->second;
    }
    // getGraph(..., global=true) is seeded from RTAB-Map's local optimized
    // table in this configuration, so it can legitimately return only the
    // connected local subset.  Replacing the persistent table would discard
    // poses for older occupancy grids.  Merge the returned native corrections
    // and retain every previously accepted node instead.
    const std::size_t poses_before = poses_.size();
    for (const auto &entry : refreshed) {
      poses_[entry.first] = entry.second;
    }
    force_global_grid_rebuild_ = true;
    ++global_graph_refresh_count_;
    if (trace_enabled()) {
      std::cerr << "global_graph_refresh frame_count="
                << global_graph_refresh_count_
                << " current_id=" << current_id
                << " poses=" << poses_.size()
                << " poses_before=" << poses_before
                << " refreshed=" << refreshed.size()
                << " constraints=" << global_constraints.size()
                << " anchored=" << (anchored ? 1 : 0)
                << " gauge_anchor_id=" << gauge_anchor_id
                << " gauge_x=" << gauge.x()
                << " gauge_y=" << gauge.y()
                << " gauge_yaw=" << gauge.theta() << "\n";
    }
    return true;
  }

  void remember_observation(const FrameInput &frame,
                            const rtabmap::SensorData &data) {
    previous_visual_data_ = data;
    previous_rgb_ = frame.rgb.clone();
    previous_depth_ = frame.depth.clone();
    previous_camera_to_base_ = frame.camera_to_base;
  }

  void forget_observation_reference() {
    previous_visual_data_ = rtabmap::SensorData();
    previous_rgb_.release();
    previous_depth_.release();
    previous_camera_to_base_.setNull();
  }

  bool is_static_observation(const FrameInput &frame) const {
    if (previous_rgb_.empty() || previous_depth_.empty() ||
        previous_camera_to_base_.isNull() ||
        previous_rgb_.size() != frame.rgb.size() ||
        previous_rgb_.type() != frame.rgb.type() ||
        previous_depth_.size() != frame.depth.size()) {
      return false;
    }
    const rtabmap::Transform camera_delta =
        previous_camera_to_base_.inverse() * frame.camera_to_base;
    if (camera_delta.getNorm() > kStaticCameraTranslationM ||
        camera_delta.getAngle(rtabmap::Transform::getIdentity()) >
            kStaticCameraRotationRad) {
      return false;
    }

    int previous_valid = 0;
    int current_valid = 0;
    int texture_samples = 0;
    int textured_samples = 0;
    std::vector<float> depth_differences;
    std::vector<std::uint8_t> rgb_differences;
    const std::size_t reserve = frame.depth.total() /
                                (kStaticObservationStride *
                                 kStaticObservationStride);
    depth_differences.reserve(reserve);
    rgb_differences.reserve(reserve);
    for (int row = 0; row < frame.depth.rows;
         row += kStaticObservationStride) {
      const float *previous_depth = previous_depth_.ptr<float>(row);
      const float *current_depth = frame.depth.ptr<float>(row);
      const cv::Vec3b *previous_rgb = previous_rgb_.ptr<cv::Vec3b>(row);
      const cv::Vec3b *current_rgb = frame.rgb.ptr<cv::Vec3b>(row);
      for (int column = 0; column < frame.depth.cols;
           column += kStaticObservationStride) {
        const bool previous_ok = std::isfinite(previous_depth[column]) &&
                                 previous_depth[column] > 0.0F;
        const bool current_ok = std::isfinite(current_depth[column]) &&
                                current_depth[column] > 0.0F;
        previous_valid += previous_ok ? 1 : 0;
        current_valid += current_ok ? 1 : 0;
        if (!previous_ok || !current_ok) {
          continue;
        }
        depth_differences.push_back(
            std::abs(previous_depth[column] - current_depth[column]));
        const cv::Vec3b &before = previous_rgb[column];
        const cv::Vec3b &after = current_rgb[column];
        const int channel_difference = std::max(
            {std::abs(static_cast<int>(before[0]) - after[0]),
             std::abs(static_cast<int>(before[1]) - after[1]),
             std::abs(static_cast<int>(before[2]) - after[2])});
        rgb_differences.push_back(
            static_cast<std::uint8_t>(std::min(channel_difference, 255)));
        if (column + kStaticObservationStride < frame.rgb.cols &&
            row + kStaticObservationStride < frame.rgb.rows) {
          const cv::Vec3b &right =
              current_rgb[column + kStaticObservationStride];
          const cv::Vec3b &down =
              frame.rgb.ptr<cv::Vec3b>(row + kStaticObservationStride)[column];
          const int spatial_difference = std::max(
              {std::abs(static_cast<int>(after[0]) - right[0]),
               std::abs(static_cast<int>(after[1]) - right[1]),
               std::abs(static_cast<int>(after[2]) - right[2]),
               std::abs(static_cast<int>(after[0]) - down[0]),
               std::abs(static_cast<int>(after[1]) - down[1]),
               std::abs(static_cast<int>(after[2]) - down[2])});
          ++texture_samples;
          textured_samples +=
              spatial_difference >= kStaticTextureDifference ? 1 : 0;
        }
      }
    }
    const int valid_reference =
        std::max(1, std::min(previous_valid, current_valid));
    if (static_cast<int>(depth_differences.size()) <
            kStaticObservationMinOverlap ||
        static_cast<float>(depth_differences.size()) / valid_reference <
            kStaticObservationMinOverlapRatio ||
        texture_samples == 0 ||
        static_cast<float>(textured_samples) / texture_samples <
            kStaticMinTexturedRatio) {
      return false;
    }
    const std::size_t percentile_index = static_cast<std::size_t>(
        0.90 * static_cast<double>(depth_differences.size() - 1));
    std::nth_element(
        depth_differences.begin(),
        depth_differences.begin() +
            static_cast<std::ptrdiff_t>(percentile_index),
        depth_differences.end());
    std::nth_element(
        rgb_differences.begin(),
        rgb_differences.begin() +
            static_cast<std::ptrdiff_t>(percentile_index),
        rgb_differences.end());
    return depth_differences[percentile_index] <= kStaticDepthP90M &&
           rgb_differences[percentile_index] <= kStaticRgbP90;
  }

  void assemble_height_layers(std::vector<std::uint8_t> &low,
                              std::vector<std::uint8_t> &high,
                              const cv::Mat &occupancy, float x_min,
                              float y_min) const {
    const int width = occupancy.cols;
    const int height = occupancy.rows;
    std::vector<std::uint8_t> bands(occupancy.total(), 0);
    for (const auto &entry : occupancy_height_points_) {
      const auto pose_iter = display_poses_.find(entry.first);
      if (pose_iter == display_poses_.end() || pose_iter->second.isNull()) {
        continue;
      }
      const rtabmap::Transform &pose = pose_iter->second;
      for (const HeightPoint &point : entry.second) {
        const float world_x = pose.r11() * point.x + pose.r12() * point.y + pose.x();
        const float world_y = pose.r21() * point.x + pose.r22() * point.y + pose.y();
        const int column = static_cast<int>(std::floor((world_x - x_min) / kGridCellM));
        const int row = static_cast<int>(std::floor((world_y - y_min) / kGridCellM));
        if (column < 0 || row < 0 || column >= width || row >= height) {
          continue;
        }
        const std::size_t index = static_cast<std::size_t>(row) * width + column;
        bands[index] |= static_cast<std::uint8_t>(1U << point.band);
      }
    }
    const auto *cells = occupancy.ptr<std::int8_t>(0);
    const std::uint8_t grounded_mask = static_cast<std::uint8_t>(
        (1U << kWallMinGroundedRun) - 1U);
    for (std::size_t index = 0; index < occupancy.total(); ++index) {
      if (cells[index] < 65) {
        continue;
      }
      if ((bands[index] & grounded_mask) == grounded_mask) {
        low[index] = 1;
      } else {
        high[index] = 1;
      }
    }
  }

  void overlay_recovered_obstacles(ProcessResult &result) const {
    const std::size_t cell_count =
        static_cast<std::size_t>(result.meta.width) * result.meta.height;
    if (result.meta.width == 0U || result.meta.height == 0U ||
        result.occupancy.size() != cell_count ||
        result.low.size() != cell_count || result.high.size() != cell_count) {
      return;
    }
    for (const auto &entry : recovered_height_points_) {
      const auto pose_iter = display_poses_.find(entry.first);
      if (pose_iter == display_poses_.end() || pose_iter->second.isNull()) {
        continue;
      }
      const rtabmap::Transform &pose = pose_iter->second;
      std::unordered_set<std::uint64_t> local_cells;
      local_cells.reserve(entry.second.size());
      for (const HeightPoint &point : entry.second) {
        const int local_cell_x =
            static_cast<int>(std::floor(point.x / kGridCellM));
        const int local_cell_y =
            static_cast<int>(std::floor(point.y / kGridCellM));
        if (!local_cells.insert(
                height_cell_key(local_cell_x, local_cell_y, 0U)).second) {
          continue;
        }
        const float local_x =
            (static_cast<float>(local_cell_x) + 0.5F) * kGridCellM;
        const float local_y =
            (static_cast<float>(local_cell_y) + 0.5F) * kGridCellM;
        const float world_x =
            pose.r11() * local_x + pose.r12() * local_y + pose.x();
        const float world_y =
            pose.r21() * local_x + pose.r22() * local_y + pose.y();
        const int world_cell_x =
            static_cast<int>(std::floor(world_x / kGridCellM));
        const int world_cell_y =
            static_cast<int>(std::floor(world_y / kGridCellM));
        const bool current_pose_traversed =
            !current_pose_.isNull() && finite_planar_transform(current_pose_) &&
            std::hypot(world_x - current_pose_.x(),
                       world_y - current_pose_.y()) <=
                kTraversedFreeCellRadiusM;
        if (current_pose_traversed ||
            traversed_free_cells_.find(
                height_cell_key(world_cell_x, world_cell_y, 0U)) !=
                traversed_free_cells_.end()) {
          continue;
        }
        const int column = static_cast<int>(
            std::floor((world_x - result.meta.x_min) / kGridCellM));
        const int row = static_cast<int>(
            std::floor((world_y - result.meta.y_min) / kGridCellM));
        if (column < 0 || row < 0 ||
            column >= static_cast<int>(result.meta.width) ||
            row >= static_cast<int>(result.meta.height)) {
          continue;
        }
        const std::size_t index =
            static_cast<std::size_t>(row) * result.meta.width +
            static_cast<std::size_t>(column);
        result.occupancy[index] = 100;
        result.low[index] = 1U;
        result.high[index] = 0U;
      }
    }
  }

  std::string database_path_;
  bool owns_database_ = false;
  WorkerProfile profile_;
  FeatureBackend feature_backend_;
  std::string python_detector_path_;
  std::string python_matcher_path_;
  rtabmap::ParametersMap parameters_;
  rtabmap::ParametersMap read_only_rgbd_parameters_;
  std::unique_ptr<rtabmap::Odometry> odometry_;
  std::unique_ptr<rtabmap::Registration> visual_registration_;
  std::unique_ptr<rtabmap::Registration> read_only_rgbd_registration_;
  std::unique_ptr<rtabmap::Registration> map_registration_;
  std::unique_ptr<rtabmap::Rtabmap> slam_;
  std::unique_ptr<rtabmap::LocalGridMaker> query_grid_maker_;
  rtabmap::LocalGridCache local_grids_;
  std::unique_ptr<rtabmap::OccupancyGrid> grid_;
  std::unordered_map<int, std::vector<HeightPoint>> height_points_;
  std::unordered_map<int, std::vector<HeightPoint>>
      occupancy_height_points_;
  std::unordered_map<int, std::vector<HeightPoint>>
      recovered_height_points_;
  std::unordered_set<std::uint64_t> traversed_free_cells_;
  std::unordered_map<int, float> node_motion_progress_m_;
  std::unordered_map<int, std::size_t> node_odometry_history_index_;
  std::vector<PlanarOdometryStep> odometry_history_;
  std::vector<PlanarOdometryStep> raw_qvel_history_;
  std::map<int, rtabmap::Transform> poses_;
  std::map<int, rtabmap::Transform> display_poses_;
  bool force_global_grid_rebuild_ = false;
  std::uint32_t global_graph_refresh_count_ = 0;
  rtabmap::Transform current_pose_;
  rtabmap::Transform native_current_pose_;
  rtabmap::Transform fused_odom_pose_;
  rtabmap::SensorData previous_visual_data_;
  cv::Mat previous_rgb_;
  cv::Mat previous_depth_;
  rtabmap::Transform previous_camera_to_base_;
  std::uint32_t loop_count_ = 0;
  std::uint32_t last_inliers_ = 0;
  std::uint32_t last_features_ = 0;
  std::int32_t last_ref_node_id_ = 0;
  std::int32_t last_mapping_node_id_ = 0;
  MappingMode mode_ = MappingMode::kMapping;
  SoftMappingState soft_mapping_state_ = SoftMappingState::kBuilding;
  bool localized_this_frame_ = false;
  bool visual_localized_this_frame_ = false;
  bool geometric_localized_this_frame_ = false;
  // Mapping remains the session mode while a confirmed revisit is in flight,
  // but the current observation is transient and must not create a signature
  // or mutate occupancy.  A separate bit makes this distinction visible to
  // the Python interface without invoking RTAB-Map's expensive irreversible
  // mapping-to-localization transition.
  bool read_only_match_this_frame_ = false;
  bool recovery_hold_this_frame_ = false;
  std::uint32_t recovery_graph_bridge_count_ = 0;
  std::uint32_t verified_graph_bridge_count_ = 0;
  int verified_bridge_anchor_id_this_frame_ = 0;
  bool normal_global_probe_complete_ = false;
  bool normal_global_search_pending_ = false;
  bool recovery_interruption_active_ = false;
  float recovery_unobserved_translation_m_ = 0.0F;
  float recovery_unobserved_yaw_rad_ = 0.0F;
  bool read_only_revisit_active_ = false;
  bool native_revisit_hold_pending_ = false;
  std::int32_t native_revisit_hold_candidate_id_ = 0;
  bool read_only_revisit_identity_verified_ = false;
  std::int32_t read_only_revisit_candidate_id_ = 0;
  float read_only_revisit_start_travel_m_ = 0.0F;
  std::uint32_t read_only_revisit_unknown_streak_ = 0;
  std::uint32_t read_only_revisit_no_observation_streak_ = 0;
  std::uint32_t native_novelty_resume_guard_updates_ = 0;
  bool native_novelty_resume_reconciliation_pending_ = false;
  std::uint32_t native_novelty_resume_reconciliation_viewpoints_remaining_ = 0;
  rtabmap::Transform native_novelty_resume_last_progress_pose_;
  rtabmap::Transform native_novelty_resume_anchor_map_pose_;
  std::size_t native_novelty_resume_anchor_history_index_ = 0;
  std::unordered_map<std::uint64_t, NoveltyResumeObstacleEvidence>
      native_novelty_resume_obstacle_evidence_;
  bool native_novelty_resume_frame_alignment_safe_ = false;
  std::size_t native_novelty_resume_frame_alignment_supports_ = 0;
  double native_novelty_resume_frame_alignment_max_bound_m_ = 0.0;
  cv::Mat native_novelty_resume_snapshot_map_;
  float native_novelty_resume_snapshot_x_min_ = 0.0F;
  float native_novelty_resume_snapshot_y_min_ = 0.0F;
  cv::Mat native_novelty_resume_protected_free_map_;
  float native_novelty_resume_protected_free_x_min_ = 0.0F;
  float native_novelty_resume_protected_free_y_min_ = 0.0F;
  double soft_novelty_start_stamp_s_ =
      -std::numeric_limits<double>::infinity();
  rtabmap::Transform soft_novelty_anchor_pose_;
  rtabmap::Transform soft_novelty_last_view_pose_;
  float soft_novelty_max_translation_m_ = 0.0F;
  std::uint32_t soft_novelty_distinct_viewpoints_ = 0;
  std::uint32_t read_only_revisit_count_ = 0;
  bool read_only_candidate_active_ = false;
  float read_only_candidate_start_motion_m_ = 0.0F;
  std::uint32_t read_only_candidate_unknown_streak_ = 0;
  std::uint32_t read_only_candidate_no_observation_streak_ = 0;
  std::uint32_t read_only_scan_match_failures_ = 0;
  double read_only_scan_match_next_stamp_s_ =
      -std::numeric_limits<double>::infinity();
  ReadOnlyScanMatchProposal read_only_scan_match_proposal_;
  std::uint32_t mapping_frames_ = 0;
  float mapping_travel_m_ = 0.0F;
  float mapping_rotation_rad_ = 0.0F;
  float session_motion_progress_m_ = 0.0F;
  std::uint32_t usable_observation_streak_ = 0;
  // Non-zero while the convergence gate has remained true during the
  // debounce interval.  A candidate is cancelled as soon as a later metric
  // sample reports new coverage or graph motion.
  std::uint32_t freeze_candidate_start_frame_ = 0;
  float freeze_candidate_motion_progress_m_ = 0.0F;
  std::size_t freeze_candidate_known_cells_ = 0;
  std::size_t freeze_candidate_revisit_count_ = 0;
  float freeze_candidate_max_novelty_ratio_ = 1.0F;
  std::uint32_t localization_frames_ = 0;
  std::uint32_t localization_observations_since_metric_ = 0;
  double localization_last_metric_stamp_s_ =
      -std::numeric_limits<double>::infinity();
  std::uint32_t last_visual_revisit_frame_ = 0;
  float last_observation_novelty_ratio_ = 1.0F;
  float last_observation_endpoint_novelty_ratio_ = 1.0F;
  float last_observation_ray_novelty_ratio_ = 1.0F;
  std::uint32_t idle_static_streak_ = 0;
  bool idle_hold_active_ = false;
  float idle_accumulated_translation_m_ = 0.0F;
  float idle_accumulated_yaw_rad_ = 0.0F;
  std::deque<CoverageSample> coverage_history_;
  std::deque<std::uint32_t> recent_visual_revisit_frames_;
  std::set<std::array<int, 3>> seen_loop_constraints_;
  std::set<std::pair<int, int>> accepted_loop_pairs_;
  std::set<std::pair<int, int>> visual_loop_regions_;
  ConvergenceEvidence convergence_;
  double last_slam_stamp_s_ = -std::numeric_limits<double>::infinity();
  cv::Mat pending_slam_covariance_;
  ProvisionalQueryBuffer query_buffer_;
  std::map<std::uint64_t, br::QueryOutcome> completed_recovery_queries_;
  std::map<std::uint64_t, br::QueryOutcome> completed_normal_queries_;
  std::map<std::uint64_t, std::uint16_t> recovery_query_high_water_;
  std::map<std::uint64_t, std::uint16_t> normal_query_high_water_;
  std::size_t durable_query_terminal_count_ = 0U;
  std::uint32_t query_buffered_keyframes_ = 0;
  std::uint32_t query_buffer_overflows_ = 0;
  std::uint32_t query_invalid_frame_discards_ = 0;
  std::uint32_t query_integrity_failures_ = 0;
  std::uint32_t query_promotions_ = 0;
  std::uint32_t query_all_known_discards_ = 0;
  std::uint32_t query_promotion_rejections_ = 0;
  std::uint32_t query_process_failures_ = 0;
  br::QueryOutcome query_outcome_this_frame_ = br::QueryOutcome::kNone;
  QueryScope query_scope_this_frame_ = QueryScope::kNone;
  std::uint64_t query_generation_this_frame_ = 0;
  bool query_integrity_failed_ = false;
  std::set<int> query_footprint_ignored_ids_;
  cv::Mat cached_map_;
  float cached_x_min_ = 0.0F;
  float cached_y_min_ = 0.0F;
  std::vector<std::uint8_t> cached_low_;
  std::vector<std::uint8_t> cached_high_;
  cv::Mat frozen_map_;
  float frozen_x_min_ = 0.0F;
  float frozen_y_min_ = 0.0F;
  std::vector<std::uint8_t> frozen_low_;
  std::vector<std::uint8_t> frozen_high_;
};

std::vector<std::uint8_t> encode_result(const ProcessResult &result) {
  std::vector<std::uint8_t> payload;
  const std::size_t expected = sizeof(result.meta) + result.occupancy.size() + result.low.size() +
                               result.high.size() + result.poses.size() * sizeof(br::PoseRecord);
  payload.reserve(expected);
  append_bytes(payload, &result.meta, sizeof(result.meta));
  append_bytes(payload, result.occupancy.data(), result.occupancy.size());
  append_bytes(payload, result.low.data(), result.low.size());
  append_bytes(payload, result.high.data(), result.high.size());
  if (!result.poses.empty()) {
    append_bytes(payload, result.poses.data(), result.poses.size() * sizeof(br::PoseRecord));
  }
  return payload;
}

struct WorkerOptions {
  std::string database_path;
  WorkerProfile profile = WorkerProfile::kOfficial;
  FeatureBackend feature_backend = FeatureBackend::kCpu;
  std::string python_detector_path;
  std::string python_matcher_path;
};

WorkerOptions parse_options(int argc, char **argv) {
  WorkerOptions output;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--database" && index + 1 < argc) {
      output.database_path = argv[++index];
    } else if (argument == "--feature-backend" && index + 1 < argc) {
      const std::string value(argv[++index]);
      if (value == "cpu") {
        output.feature_backend = FeatureBackend::kCpu;
      } else if (value == "kornia-sift") {
        output.feature_backend = FeatureBackend::kKorniaSift;
      } else {
        throw std::invalid_argument(
            "feature backend must be cpu or kornia-sift");
      }
    } else if (argument == "--python-detector" && index + 1 < argc) {
      output.python_detector_path = argv[++index];
    } else if (argument == "--python-matcher" && index + 1 < argc) {
      output.python_matcher_path = argv[++index];
    } else if (argument == "--profile" && index + 1 < argc) {
      const std::string value(argv[++index]);
      if (value == "official") {
        output.profile = WorkerProfile::kOfficial;
      } else if (value == "sparse-rgbd") {
        output.profile = WorkerProfile::kSparseRgbd;
      } else if (value == "sparse-icp") {
        output.profile = WorkerProfile::kSparseIcp;
      } else if (value == "native-robust") {
        output.profile = WorkerProfile::kNativeRobust;
      } else if (value == "native-robust-ceres") {
        output.profile = WorkerProfile::kNativeRobustCeres;
      } else {
        throw std::invalid_argument(
            "worker profile must be official, sparse-rgbd, sparse-icp, native-robust, or native-robust-ceres");
      }
    } else {
      throw std::invalid_argument(
          "usage: behavior_rtabmap_worker [--database PATH] "
          "[--profile official|sparse-rgbd|sparse-icp|native-robust|native-robust-ceres] "
          "[--feature-backend cpu|kornia-sift] [--python-detector PATH] "
          "[--python-matcher PATH]");
    }
  }
  if (output.feature_backend == FeatureBackend::kKorniaSift &&
      (output.python_detector_path.empty() ||
       output.python_matcher_path.empty())) {
    throw std::invalid_argument(
        "kornia-sift requires --python-detector and --python-matcher paths");
  }
  if (output.feature_backend == FeatureBackend::kCpu &&
      (!output.python_detector_path.empty() ||
       !output.python_matcher_path.empty())) {
    throw std::invalid_argument(
        "python detector and matcher paths are only valid with kornia-sift");
  }
  return output;
}

}  // namespace

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::srand(0);
  cv::setRNGSeed(0);
  cv::setNumThreads(1);
  const char *upstream_log = std::getenv("BEHAVIOR_RTABMAP_UPSTREAM_LOG");
  if (upstream_log != nullptr && upstream_log[0] != '\0' &&
      upstream_log[0] != '0') {
    const std::string upstream_log_path =
        std::string(upstream_log) == "1"
            ? "/tmp/behavior_rtabmap_upstream_" +
                  std::to_string(static_cast<long long>(::getpid())) + ".log"
            : std::string(upstream_log);
    ULogger::setType(ULogger::kTypeFile, upstream_log_path, false);
    ULogger::setLevel(ULogger::kInfo);
  } else {
    ULogger::setType(ULogger::kTypeNoLog);
    ULogger::setLevel(ULogger::kFatal);
  }

  try {
    const WorkerOptions options = parse_options(argc, argv);
    SlamWorker worker(options.database_path,
                      options.profile,
                      options.feature_backend,
                      options.python_detector_path,
                      options.python_matcher_path);
    while (true) {
      br::Prefix prefix{};
      if (!read_all(std::cin, &prefix, sizeof(prefix))) {
        return 0;
      }
      if (std::memcmp(prefix.magic, "B1RQ", 4) != 0 ||
          prefix.version != br::kProtocolVersion || prefix.payload_size > br::kMaxPacketBytes) {
        send_error(br::Status::kBadRequest, "invalid request prefix");
        return 2;
      }
      std::vector<std::uint8_t> payload(prefix.payload_size);
      if (!payload.empty() && !read_all(std::cin, payload.data(), payload.size())) {
        return 2;
      }
      const auto command = static_cast<br::Command>(prefix.code);
      try {
        if (command == br::Command::kShutdown) {
          return 0;
        }
        if (command == br::Command::kReset) {
          worker.reset();
          send_packet(br::Status::kOk, encode_result(worker.snapshot(0, true, false, false)));
        } else if (command == br::Command::kPing) {
          send_packet(br::Status::kOk, encode_result(worker.snapshot(0, true, false, false)));
        } else if (command == br::Command::kFrame) {
          FrameInput frame = decode_frame(payload);
          ProcessResult result = worker.process(frame);
          send_packet(result.meta.tracking_ok ? br::Status::kOk : br::Status::kTrackingLost,
                      encode_result(result));
        } else {
          send_error(br::Status::kBadRequest, "unknown command");
        }
      } catch (const std::invalid_argument &error) {
        send_error(br::Status::kBadRequest, error.what());
        if (worker.query_integrity_failed()) {
          std::cout.flush();
          std::cerr.flush();
          std::_Exit(3);
        }
      } catch (const std::exception &error) {
        send_error(br::Status::kInternalError, error.what());
        if (worker.query_integrity_failed()) {
          std::cout.flush();
          std::cerr.flush();
          std::_Exit(3);
        }
      }
    }
  } catch (const std::exception &error) {
    send_error(br::Status::kInternalError, error.what());
    return 1;
  }
}
