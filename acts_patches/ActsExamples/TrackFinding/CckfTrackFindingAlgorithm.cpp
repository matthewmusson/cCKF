// CckfTrackFindingAlgorithm.cpp — implementation of the cCKF ACTS Algorithm.
//
// Structure mirrors TrackFindingAlgorithm.cpp closely. The key additions:
//   1. CckfMeasurementSelectorAdapter — wraps CckfMeasurementSelector, injects
//      BranchContext/SensorProps before each surface's select() call.
//   2. CckfBranchStopperWrapper — wraps CckfBranchStopper, updates per-branch
//      dynamic columns (step_k, gate log-odds, X0) before each value decision.
//   3. Dynamic column registration + initialisation for the four cCKF columns.
//   4. Per-event timing via CckfTimers.

#include "CckfTrackFindingAlgorithm.hpp"

#include "cckf/CckfBranchStopper.hpp"
#include "cckf/CckfFeatures.hpp"
#include "cckf/CckfMeasurementSelector.hpp"
#include "cckf/CckfTimers.hpp"
#include "cckf/SensorLookup.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Direction.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/ProxyAccessor.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/Types.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/SympyStepper.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/TrackFinding/TrackStateCreator.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/HashedString.hpp"
#include "Acts/Utilities/HashCombine.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/Seed.hpp"
#include "ActsExamples/EventData/SpacePoint.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// std::hash specialization for SeedIdentifier (std::array<T,N>).
// Identical to the one in TrackFindingAlgorithm.cpp — template
// specializations with identical definitions across TUs are well-defined.
template <class T, std::size_t N>
struct std::hash<std::array<T, N>> {
  std::size_t operator()(const std::array<T, N>& array) const {
    std::size_t result = 0;
    for (auto&& element : array) {
      result = Acts::hashMixAndCombine(result, element);
    }
    return result;
  }
};

namespace ActsExamples {

namespace {

// ============================================================================
// Seed deduplication helpers (same as TrackFindingAlgorithm.cpp)
// ============================================================================

using SeedIdentifier = std::array<Index, 3>;

SeedIdentifier makeSeedIdentifier(const ConstSeedProxy& seed) {
  SeedIdentifier result;
  for (const auto& [i, spIndex] : Acts::enumerate(seed.spacePointIndices())) {
    const ConstSpacePointProxy sp =
        seed.container().spacePointContainer().at(spIndex);
    const Acts::SourceLink& firstSourceLink = sp.sourceLinks().front();
    result.at(i) = firstSourceLink.get<IndexSourceLink>().index();
  }
  return result;
}

template <typename Visitor>
void visitSeedIdentifiers(const TrackProxy& track, Visitor visitor) {
  std::vector<Index> sourceLinkIndices;
  sourceLinkIndices.reserve(track.nMeasurements());
  for (const auto& trackState : track.trackStatesReversed()) {
    if (!trackState.hasUncalibratedSourceLink()) {
      continue;
    }
    const Acts::SourceLink& sourceLink = trackState.getUncalibratedSourceLink();
    sourceLinkIndices.push_back(sourceLink.get<IndexSourceLink>().index());
  }
  for (std::size_t i = 0; i < sourceLinkIndices.size(); ++i) {
    for (std::size_t j = i + 1; j < sourceLinkIndices.size(); ++j) {
      for (std::size_t k = j + 1; k < sourceLinkIndices.size(); ++k) {
        visitor({sourceLinkIndices.at(k), sourceLinkIndices.at(j),
                 sourceLinkIndices.at(i)});
      }
    }
  }
}

// ============================================================================
// CckfMeasurementSelectorAdapter
// ============================================================================
//
// Wraps cckf::CckfMeasurementSelector to:
//   1. Provide a non-template select() method for the delegate (the
//      underlying CckfMeasurementSelector::select<traj_t> is a template).
//   2. Before each surface's select(), extract the branch context from the
//      predicted state and backward trajectory walk, then call
//      setBranchContext() / setSensorProps() on the selector.
//   3. Optionally filter candidates to seed hits (stayOnSeed logic).

class CckfMeasurementSelectorAdapter {
 public:
  using Traj = Acts::VectorMultiTrajectory;

  explicit CckfMeasurementSelectorAdapter(
      cckf::CckfMeasurementSelector* selector, Traj* trajectory,
      const cckf::SensorLookup* sensorLookup)
      : m_selector(selector), m_trajectory(trajectory),
        m_sensorLookup(sensorLookup) {}

  void setSeed(const std::optional<ConstSeedProxy>& seed) { m_seed = seed; }
  void setGeoContext(const Acts::GeometryContext& gc) { m_geoContext = &gc; }

  Acts::Result<std::pair<std::vector<Traj::TrackStateProxy>::iterator,
                         std::vector<Traj::TrackStateProxy>::iterator>>
  select(std::vector<Traj::TrackStateProxy>& candidates, bool& isOutlier,
         const Acts::Logger& logger) const {
    // ---- stayOnSeed logic (same as TrackFindingAlgorithm) ----
    if (m_seed.has_value()) {
      std::vector<Traj::TrackStateProxy> newCandidates;
      for (const auto& candidate : candidates) {
        if (isSeedCandidate(candidate)) {
          newCandidates.push_back(candidate);
        }
      }
      if (!newCandidates.empty()) {
        candidates = std::move(newCandidates);
      }
    }

    // ---- Build BranchContext from the predicted state + backward walk ----
    if (!candidates.empty() && m_selector != nullptr) {
      cckf::BranchContext ctx;

      // Extract eta and q/p from the predicted bound parameters on this
      // surface. All candidates share the same predicted state (they are
      // on the same surface for the same branch).
      const auto predicted = candidates[0].predicted();
      constexpr float kThetaEps = 1e-6f;
      constexpr float kPi = 3.14159265358979323846f;
      float theta = static_cast<float>(predicted[Acts::eBoundTheta]);
      theta = std::clamp(theta, kThetaEps, kPi - kThetaEps);
      ctx.eta = -std::log(std::tan(theta / 2.0f));
      ctx.qop = static_cast<float>(predicted[Acts::eBoundQOverP]);

      // Walk backward from prevTip through the branch's existing track
      // states to count measurements, holes, sequential holes, and
      // compute step_k. This is O(n_surfaces_so_far) per surface, but
      // n_surfaces ~ 30 for typical detectors — negligible vs. MLP
      // inference. A running-accumulator approach (updated in the branch
      // stopper wrapper) could avoid the walk, but the measurement
      // selector delegate has no access to the track proxy, so the walk
      // is the correct approach here.
      Acts::TrackIndexType prevIdx = candidates[0].previous();
      uint32_t n_hits = 0;
      uint32_t n_holes = 0;
      uint32_t n_seq_holes = 0;
      uint32_t step = 0;
      bool firstNonHoleSeen = false;

      while (prevIdx != Acts::kTrackIndexInvalid) {
        auto ts = m_trajectory->getTrackState(prevIdx);
        auto flags = ts.typeFlags();

        if (flags.isMeasurement()) {
          ++n_hits;
          ++step;
          firstNonHoleSeen = true;
        } else if (flags.isHole()) {
          ++n_holes;
          ++step;
          if (!firstNonHoleSeen) {
            ++n_seq_holes;
          }
        } else if (flags.isOutlier()) {
          ++step;
          firstNonHoleSeen = true;
        }
        // Material-only states don't count toward step_k.

        prevIdx = ts.previous();
      }

      ctx.n_hits = n_hits;
      ctx.n_holes = n_holes;
      ctx.n_seq_holes = n_seq_holes;
      ctx.step_k = step;
      ctx.pathInX0 = 0.0f;  // TODO(task5): accumulate from material

      m_selector->setBranchContext(ctx);

      cckf::SensorProps sensorProps{};
      if (m_sensorLookup != nullptr) {
        auto geoId = candidates[0].referenceSurface().geometryId();
        sensorProps = m_sensorLookup->get(
            static_cast<uint32_t>(geoId.volume()));
      }
      m_selector->setSensorProps(sensorProps);
    }

    if (m_selector != nullptr && m_geoContext != nullptr) {
      m_selector->setGeoContext(*m_geoContext);
    }

    // ---- Delegate to the cCKF gate ----
    return m_selector->select<Traj>(candidates, isOutlier, logger);
  }

 private:
  cckf::CckfMeasurementSelector* m_selector;
  Traj* m_trajectory;
  const cckf::SensorLookup* m_sensorLookup;
  std::optional<ConstSeedProxy> m_seed;
  const Acts::GeometryContext* m_geoContext = nullptr;

  bool isSeedCandidate(const Traj::TrackStateProxy& candidate) const {
    assert(candidate.hasUncalibratedSourceLink());
    assert(m_seed.has_value());
    const Acts::SourceLink& sourceLink = candidate.getUncalibratedSourceLink();
    for (const ConstSpacePointProxy sp : m_seed->spacePoints()) {
      for (const Acts::SourceLink& sl : sp.sourceLinks()) {
        if (sourceLink.get<IndexSourceLink>() == sl.get<IndexSourceLink>()) {
          return true;
        }
      }
    }
    return false;
  }
};

// ============================================================================
// FallbackMeasurementSelectorAdapter
// ============================================================================
//
// Wraps Acts::MeasurementSelector (chi2-based) with a non-template select()
// for the delegate, plus stayOnSeed logic. Used when gateWeightsPath is
// empty (ablation without the gate MLP). Mirrors the MeasurementSelector
// wrapper in TrackFindingAlgorithm.cpp exactly.

class FallbackMeasurementSelectorAdapter {
 public:
  using Traj = Acts::VectorMultiTrajectory;

  explicit FallbackMeasurementSelectorAdapter(
      Acts::MeasurementSelector selector)
      : m_selector(std::move(selector)) {}

  void setSeed(const std::optional<ConstSeedProxy>& seed) { m_seed = seed; }

  Acts::Result<std::pair<std::vector<Traj::TrackStateProxy>::iterator,
                         std::vector<Traj::TrackStateProxy>::iterator>>
  select(std::vector<Traj::TrackStateProxy>& candidates, bool& isOutlier,
         const Acts::Logger& logger) const {
    // ---- stayOnSeed logic (same as TrackFindingAlgorithm) ----
    if (m_seed.has_value()) {
      std::vector<Traj::TrackStateProxy> newCandidates;
      for (const auto& candidate : candidates) {
        if (isSeedCandidate(candidate)) {
          newCandidates.push_back(candidate);
        }
      }
      if (!newCandidates.empty()) {
        candidates = std::move(newCandidates);
      }
    }

    return m_selector.select<Acts::VectorMultiTrajectory>(candidates, isOutlier,
                                                          logger);
  }

 private:
  Acts::MeasurementSelector m_selector;
  std::optional<ConstSeedProxy> m_seed;

  bool isSeedCandidate(const Traj::TrackStateProxy& candidate) const {
    assert(candidate.hasUncalibratedSourceLink());
    assert(m_seed.has_value());
    const Acts::SourceLink& sourceLink = candidate.getUncalibratedSourceLink();
    for (const ConstSpacePointProxy sp : m_seed->spacePoints()) {
      for (const Acts::SourceLink& sl : sp.sourceLinks()) {
        if (sourceLink.get<IndexSourceLink>() == sl.get<IndexSourceLink>()) {
          return true;
        }
      }
    }
    return false;
  }
};

// ============================================================================
// CckfBranchStopperWrapper
// ============================================================================
//
// Wraps cckf::CckfBranchStopper to update the four cCKF dynamic columns on
// the track proxy BEFORE the value function reads them. This mirrors the
// pattern in TrackFindingAlgorithm's BranchStopper, which uses a mutable
// ProxyAccessor to update BranchState::nPixelHoles/nStripHoles on a
// `const TrackProxy&`. (The const-ref prevents reassigning the proxy
// variable, but does not prevent mutation through the proxy because
// TrackProxy::ReadOnly is false.)
//
// Column update policy:
//   step_k:            +1 for every sensitive surface (measurement, hole,
//                      or outlier). Updated here unconditionally.
//   sum_gate_logodds:  For measurements, += raw gate logit from the
//                      CckfMeasurementSelector (looked up by source-link
//                      index via lastGateLogit()). For holes, += a
//                      pessimistic default (kHoleLogitDefault = -5.0f,
//                      matching the Python training code's treatment).
//   min_gate_logodds:  min(current, logit) using the same source as above.
//   accumulated_x0:    TODO — requires per-surface material thickness.
//                      Left at initial value (0).

class CckfBranchStopperWrapper {
 public:
  using BranchStopperResult =
      Acts::CombinatorialKalmanFilterBranchStopperResult;

  /// Pessimistic gate logit for holes (no measurement was selected).
  /// This matches the chi2_log_odds treatment in the Python training code
  /// for holes: a strongly negative logit indicating very low confidence.
  static constexpr float kHoleLogitDefault = -5.0f;

  explicit CckfBranchStopperWrapper(cckf::CckfBranchStopper* inner,
                                    cckf::CckfMeasurementSelector* gate)
      : m_inner(inner), m_gate(gate) {}

  mutable std::size_t m_nStoppedBranches = 0;

  BranchStopperResult operator()(
      const TrackContainer::TrackProxy& track,
      const TrackContainer::TrackStateProxy& trackState) const {
    // Update step_k: count every sensitive surface (measurement, hole, or
    // outlier). The CKF actor calls the branch stopper for each of these.
    kStepKWriter(track) += 1.0f;

    // Update gate log-odds columns.
    float logit = kHoleLogitDefault;
    if (trackState.typeFlags().isMeasurement() ||
        trackState.typeFlags().isOutlier()) {
      // Look up the raw gate logit from the most recent select() call.
      if (m_gate != nullptr && trackState.hasUncalibratedSourceLink()) {
        const Acts::SourceLink sl = trackState.getUncalibratedSourceLink();
        const auto* isl =
            sl.getPtr<ActsExamples::IndexSourceLink>();
        if (isl != nullptr) {
          auto maybeLogit = m_gate->lastGateLogit(isl->index());
          if (maybeLogit.has_value()) {
            logit = *maybeLogit;
          }
          // If not found (shouldn't happen for accepted candidates),
          // fall through with the hole default.
        }
      }
    }
    // For holes, logit stays at kHoleLogitDefault.

    kSumGateLogOddsWriter(track) += logit;
    float currentMin = kMinGateLogOddsWriter(track);
    if (logit < currentMin) {
      kMinGateLogOddsWriter(track) = logit;
    }

    // TODO(task5-or-later): Update accumulated X0. Requires per-surface
    // material thickness. The CKF actor's material interaction is
    // performed outside the branch stopper callback, so a separate
    // mechanism is needed.

    // Delegate to the value function branch stopper.
    auto result = (*m_inner)(track, trackState);
    if (result != BranchStopperResult::Continue) {
      ++m_nStoppedBranches;
    }
    return result;
  }

 private:
  // Mutable accessors — see class doc for why this is safe on const
  // TrackProxy&.
  static constexpr Acts::ProxyAccessor<float> kStepKWriter{
      Acts::hashString(cckf::CckfColumns::kStepK)};
  static constexpr Acts::ProxyAccessor<float> kSumGateLogOddsWriter{
      Acts::hashString(cckf::CckfColumns::kSumGateLogOdds)};
  static constexpr Acts::ProxyAccessor<float> kMinGateLogOddsWriter{
      Acts::hashString(cckf::CckfColumns::kMinGateLogOdds)};
  static constexpr Acts::ProxyAccessor<float> kAccumulatedX0Writer{
      Acts::hashString(cckf::CckfColumns::kAccumulatedX0)};

  cckf::CckfBranchStopper* m_inner;
  cckf::CckfMeasurementSelector* m_gate;
};

// ============================================================================
// Fallback branch stopper (no value function)
// ============================================================================

class PassthroughBranchStopper {
 public:
  using BranchStopperResult =
      Acts::CombinatorialKalmanFilterBranchStopperResult;

  mutable std::size_t m_nStoppedBranches = 0;

  BranchStopperResult operator()(
      const TrackContainer::TrackProxy& /*track*/,
      const TrackContainer::TrackStateProxy& /*trackState*/) const {
    return BranchStopperResult::Continue;
  }
};

// ============================================================================
// CckfTimers CSV writer
// ============================================================================

void writeTimingCsv(const std::string& path, std::size_t eventNumber,
                    const cckf::CckfTimers& timers, std::size_t nSeeds,
                    std::size_t nTracks) {
  if (path.empty()) {
    return;
  }
  // ACTS Sequencer can call execute() concurrently for different events.
  // Protect the check-header-then-append sequence from data races.
  static std::mutex csvMutex;
  std::lock_guard<std::mutex> lock(csvMutex);

  // Open in append mode. Write header if the file is empty/new.
  bool writeHeader = false;
  {
    std::ifstream probe(path);
    writeHeader = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream out(path, std::ios::app);
  if (!out) {
    return;  // silently skip if we cannot open
  }
  if (writeHeader) {
    out << "event,n_seeds,n_tracks,"
        << "gate_inference_ns,gate_inference_calls,"
        << "gate_feature_ns,gate_feature_calls,"
        << "value_inference_ns,value_inference_calls,"
        << "meas_selection_ns,meas_selection_calls\n";
  }
  out << eventNumber << "," << nSeeds << "," << nTracks << ","
      << timers.gate_inference.total_ns << ","
      << timers.gate_inference.n_calls << ","
      << timers.gate_feature_build.total_ns << ","
      << timers.gate_feature_build.n_calls << ","
      << timers.value_inference.total_ns << ","
      << timers.value_inference.n_calls << ","
      << timers.measurement_selection.total_ns << ","
      << timers.measurement_selection.n_calls << "\n";
}

// ============================================================================
// Column initialiser: sets the four cCKF columns to their starting values
// on a freshly created track (seed root or shallow copy).
// ============================================================================

void initCckfColumns(const TrackContainer::TrackProxy& track) {
  static constexpr Acts::ProxyAccessor<float> sumWriter{
      Acts::hashString(cckf::CckfColumns::kSumGateLogOdds)};
  static constexpr Acts::ProxyAccessor<float> minWriter{
      Acts::hashString(cckf::CckfColumns::kMinGateLogOdds)};
  static constexpr Acts::ProxyAccessor<float> x0Writer{
      Acts::hashString(cckf::CckfColumns::kAccumulatedX0)};
  static constexpr Acts::ProxyAccessor<float> stepWriter{
      Acts::hashString(cckf::CckfColumns::kStepK)};

  sumWriter(track) = 0.0f;
  minWriter(track) = std::numeric_limits<float>::infinity();
  x0Writer(track) = 0.0f;
  stepWriter(track) = 0.0f;
}

}  // namespace

// ============================================================================
// Constructor
// ============================================================================

CckfTrackFindingAlgorithm::CckfTrackFindingAlgorithm(
    Config config, std::unique_ptr<const Acts::Logger> lgr)
    : IAlgorithm("CckfTrackFindingAlgorithm", std::move(lgr)),
      m_cfg(std::move(config)) {
  if (m_cfg.inputMeasurements.empty()) {
    throw std::invalid_argument("Missing measurements input collection");
  }
  if (m_cfg.inputInitialTrackParameters.empty()) {
    throw std::invalid_argument(
        "Missing initial track parameters input collection");
  }
  if (m_cfg.outputTracks.empty()) {
    throw std::invalid_argument("Missing tracks output collection");
  }
  if (m_cfg.seedDeduplication && m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument(
        "Missing seeds input collection (required for seed deduplication)");
  }
  if (m_cfg.stayOnSeed && m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument(
        "Missing seeds input collection (required for staying on seed)");
  }
  if (!m_cfg.gateWeightsPath.empty() && m_cfg.inputClusters.empty()) {
    throw std::invalid_argument(
        "inputClusters must be set when gateWeightsPath is provided "
        "(the gate MLP requires cluster features)");
  }

  // Warn about config fields that are silently ignored when the cCKF value
  // function (or passthrough stopper) replaces the default branch stopper.
  // The default BranchStopper in TrackFindingAlgorithm uses these to count
  // pixel/strip holes separately, but the cCKF stopper reads hole counts
  // from the value function's feature vector instead.
  {
    bool valueActive = !m_cfg.valueWeightsPath.empty();

    {
      if (m_cfg.maxPixelHoles != std::numeric_limits<std::size_t>::max()) {
        ACTS_WARNING(
            "maxPixelHoles is set to " << m_cfg.maxPixelHoles
            << " but is ignored: the cCKF "
            << (valueActive ? "value function" : "passthrough")
            << " branch stopper does not use pixel/strip hole caps.");
      }
      if (m_cfg.maxStripHoles != std::numeric_limits<std::size_t>::max()) {
        ACTS_WARNING(
            "maxStripHoles is set to " << m_cfg.maxStripHoles
            << " but is ignored: the cCKF "
            << (valueActive ? "value function" : "passthrough")
            << " branch stopper does not use pixel/strip hole caps.");
      }
      if (!m_cfg.pixelVolumeIds.empty()) {
        ACTS_WARNING(
            "pixelVolumeIds is set but is ignored: the cCKF "
            << (valueActive ? "value function" : "passthrough")
            << " branch stopper does not distinguish pixel/strip volumes.");
      }
      if (!m_cfg.stripVolumeIds.empty()) {
        ACTS_WARNING(
            "stripVolumeIds is set but is ignored: the cCKF "
            << (valueActive ? "value function" : "passthrough")
            << " branch stopper does not distinguish pixel/strip volumes.");
      }
    }
  }

  if (m_cfg.trackSelectorCfg.has_value()) {
    m_trackSelector = std::visit(
        [](const auto& cfg) -> std::optional<Acts::TrackSelector> {
          return Acts::TrackSelector(cfg);
        },
        m_cfg.trackSelectorCfg.value());
  }

  if (!m_cfg.digiConfigPath.empty()) {
    m_sensorLookup =
        std::make_unique<cckf::SensorLookup>(m_cfg.digiConfigPath);
  }

  m_inputMeasurements.initialize(m_cfg.inputMeasurements);
  m_inputInitialTrackParameters.initialize(m_cfg.inputInitialTrackParameters);
  m_inputSeeds.maybeInitialize(m_cfg.inputSeeds);
  m_inputClusters.maybeInitialize(m_cfg.inputClusters);
  m_outputTracks.initialize(m_cfg.outputTracks);
}

// ============================================================================
// execute — per-event CKF with cCKF components
// ============================================================================

ProcessCode CckfTrackFindingAlgorithm::execute(
    const AlgorithmContext& ctx) const {
  // ---- Read inputs ----
  const auto& measurements = m_inputMeasurements(ctx);
  const auto& initialParameters = m_inputInitialTrackParameters(ctx);
  const SeedContainer* seeds = nullptr;
  if (m_inputSeeds.isInitialized()) {
    seeds = &m_inputSeeds(ctx);
    if (initialParameters.size() != seeds->size()) {
      ACTS_ERROR("Number of initial parameters and seeds do not match: "
                 << initialParameters.size() << " != " << seeds->size());
    }
  }

  const ClusterContainer* clusters = nullptr;
  if (m_inputClusters.isInitialized()) {
    clusters = &m_inputClusters(ctx);
  }

  // ---- Per-event timing ----
  cckf::CckfTimers timers;

  // ---- Perigee surface ----
  auto pSurface = Acts::Surface::makeShared<Acts::PerigeeSurface>(
      Acts::Vector3{0., 0., 0.});

  // ---- Calibrator + updater ----
  PassThroughCalibrator pcalibrator;
  MeasurementCalibratorAdapter calibrator(pcalibrator,
                                          measurements.container());
  Acts::GainMatrixUpdater kfUpdater(m_cfg.useJosephFormulation);

  // ---- Source link accessor ----
  IndexSourceLinkAccessor slAccessor;
  slAccessor.container = &measurements.orderedIndices();

  // ---- Measurement selector: gate MLP or fallback chi2 ----
  //
  // The cCKF measurement selector is constructed per-event because:
  //   - The ClusterContainer pointer changes each event
  //   - The CckfTimers instance is per-event
  //   - Not thread-safe: per-event ownership avoids cross-event races
  //
  // Weight loading per event (~100-200KB binary read) is negligible
  // compared to CKF propagation. If profiling shows otherwise, the blob
  // can be cached as a member and the selector reset each event.

  using TrackStateCreatorType =
      Acts::TrackStateCreator<IndexSourceLinkAccessor::Iterator,
                              TrackContainer>;

  // We need the measurement selector adapter and the CKF components to
  // have non-overlapping lifetimes with the CKF run. Declare them here
  // so they outlive the per-seed loop.

  // Gate MLP path
  std::unique_ptr<cckf::CckfMeasurementSelector> cckfSelector;
  // Fallback chi2 selector adapter (when gate is disabled)
  std::optional<FallbackMeasurementSelectorAdapter> fallbackAdapter;

  if (!m_cfg.gateWeightsPath.empty()) {
    cckf::CckfMeasurementSelector::Config selCfg;
    selCfg.gateWeightsPath = m_cfg.gateWeightsPath;
    selCfg.gateThreshold = m_cfg.gateThreshold;
    selCfg.maxCandidates = m_cfg.gateMaxCandidates;
    cckfSelector = std::make_unique<cckf::CckfMeasurementSelector>(
        selCfg, clusters, &timers);
  } else {
    fallbackAdapter.emplace(
        Acts::MeasurementSelector(m_cfg.measurementSelectorCfg));
  }

  // ---- Value function branch stopper ----
  std::unique_ptr<cckf::CckfBranchStopper> cckfStopper;
  if (!m_cfg.valueWeightsPath.empty()) {
    cckf::CckfBranchStopper::Config stopCfg;
    stopCfg.valueWeightsPath = m_cfg.valueWeightsPath;
    stopCfg.valueThreshold = m_cfg.valueThreshold;
    cckfStopper =
        std::make_unique<cckf::CckfBranchStopper>(stopCfg, &timers);
  }

  // ---- Track containers ----
  auto trackContainer = std::make_shared<Acts::VectorTrackContainer>();
  auto trackStateContainer = std::make_shared<Acts::VectorMultiTrajectory>();

  auto trackContainerTemp = std::make_shared<Acts::VectorTrackContainer>();
  auto trackStateContainerTemp =
      std::make_shared<Acts::VectorMultiTrajectory>();

  TrackContainer tracks(trackContainer, trackStateContainer);
  TrackContainer tracksTemp(trackContainerTemp, trackStateContainerTemp);

  // ---- Register cCKF dynamic columns ----
  // These are read by CckfBranchStopper via ConstProxyAccessor and
  // written by CckfBranchStopperWrapper via ProxyAccessor.
  tracks.addColumn<float>(std::string(cckf::CckfColumns::kSumGateLogOdds));
  tracks.addColumn<float>(std::string(cckf::CckfColumns::kMinGateLogOdds));
  tracks.addColumn<float>(std::string(cckf::CckfColumns::kAccumulatedX0));
  tracks.addColumn<float>(std::string(cckf::CckfColumns::kStepK));

  tracksTemp.addColumn<float>(
      std::string(cckf::CckfColumns::kSumGateLogOdds));
  tracksTemp.addColumn<float>(
      std::string(cckf::CckfColumns::kMinGateLogOdds));
  tracksTemp.addColumn<float>(
      std::string(cckf::CckfColumns::kAccumulatedX0));
  tracksTemp.addColumn<float>(std::string(cckf::CckfColumns::kStepK));

  // Standard columns (same as TrackFindingAlgorithm)
  tracks.addColumn<unsigned int>("trackGroup");
  tracksTemp.addColumn<unsigned int>("trackGroup");
  Acts::ProxyAccessor<unsigned int> seedNumber("trackGroup");

  // ---- Wire up the measurement selector adapter ----
  //
  // The adapter wraps CckfMeasurementSelector (or the fallback) and
  // provides the non-template select() that the TrackStateCreator
  // delegate needs.
  //
  // NOTE: The adapter holds a raw pointer to trackStateContainerTemp.
  // This pointer is valid for the duration of execute() — the shared_ptr
  // owns the object, and the adapter's lifetime is contained within this
  // function.

  CckfMeasurementSelectorAdapter cckfMeasSelAdapter(
      cckfSelector.get(), trackStateContainerTemp.get(),
      m_sensorLookup.get());
  cckfMeasSelAdapter.setGeoContext(ctx.geoContext);

  TrackStateCreatorType trackStateCreator;
  trackStateCreator.sourceLinkAccessor
      .template connect<&IndexSourceLinkAccessor::range>(&slAccessor);
  trackStateCreator.calibrator
      .template connect<&MeasurementCalibratorAdapter::calibrate>(&calibrator);

  if (cckfSelector) {
    trackStateCreator.measurementSelector
        .template connect<&CckfMeasurementSelectorAdapter::select>(
            &cckfMeasSelAdapter);
  } else {
    trackStateCreator.measurementSelector
        .template connect<&FallbackMeasurementSelectorAdapter::select>(
            &*fallbackAdapter);
  }

  // ---- Wire up branch stopper ----
  CckfBranchStopperWrapper cckfStopperWrapper(cckfStopper.get(),
                                              cckfSelector.get());
  PassthroughBranchStopper passthroughStopper;

  using Extensions = Acts::CombinatorialKalmanFilterExtensions<TrackContainer>;
  Extensions extensions;
  extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<
      typename TrackContainer::TrackStateContainerBackend>>(&kfUpdater);

  if (cckfStopper) {
    extensions.branchStopper
        .connect<&CckfBranchStopperWrapper::operator()>(&cckfStopperWrapper);
  } else {
    extensions.branchStopper
        .connect<&PassthroughBranchStopper::operator()>(&passthroughStopper);
  }

  extensions.createTrackStates
      .template connect<&TrackStateCreatorType::createTrackStates>(
          &trackStateCreator);

  // ---- Propagator options ----
  Acts::PropagatorPlainOptions firstPropOptions(ctx.geoContext,
                                                ctx.magFieldContext);
  firstPropOptions.maxSteps = m_cfg.maxSteps;
  firstPropOptions.direction = m_cfg.reverseSearch
                                   ? Acts::Direction::Backward()
                                   : Acts::Direction::Forward();
  firstPropOptions.constrainToVolumeIds = m_cfg.constrainToVolumeIds;
  firstPropOptions.endOfWorldVolumeIds = m_cfg.endOfWorldVolumeIds;

  Acts::PropagatorPlainOptions secondPropOptions(ctx.geoContext,
                                                  ctx.magFieldContext);
  secondPropOptions.maxSteps = m_cfg.maxSteps;
  secondPropOptions.direction = firstPropOptions.direction.invert();
  secondPropOptions.constrainToVolumeIds = m_cfg.constrainToVolumeIds;
  secondPropOptions.endOfWorldVolumeIds = m_cfg.endOfWorldVolumeIds;

  TrackFinderOptions firstOptions(ctx.geoContext, ctx.magFieldContext,
                                  ctx.calibContext, extensions,
                                  firstPropOptions);
  firstOptions.targetSurface = m_cfg.reverseSearch ? pSurface.get() : nullptr;

  TrackFinderOptions secondOptions(ctx.geoContext, ctx.magFieldContext,
                                   ctx.calibContext, extensions,
                                   secondPropOptions);
  secondOptions.targetSurface = m_cfg.reverseSearch ? nullptr : pSurface.get();
  secondOptions.skipPrePropagationUpdate = true;

  // ---- Extrapolator (for perigee extrapolation after CKF) ----
  using Extrapolator = Acts::Propagator<Acts::SympyStepper, Acts::Navigator>;
  using ExtrapolatorOptions = Extrapolator::template Options<
      Acts::ActorList<Acts::MaterialInteractor, Acts::EndOfWorldReached>>;

  Extrapolator extrapolator(
      Acts::SympyStepper(m_cfg.magneticField),
      Acts::Navigator({m_cfg.trackingGeometry},
                      logger().cloneWithSuffix("Navigator")),
      logger().cloneWithSuffix("Propagator"));

  ExtrapolatorOptions extrapolationOptions(ctx.geoContext, ctx.magFieldContext);
  extrapolationOptions.constrainToVolumeIds = m_cfg.constrainToVolumeIds;
  extrapolationOptions.endOfWorldVolumeIds = m_cfg.endOfWorldVolumeIds;

  // ---- Main seed loop ----
  ACTS_DEBUG("Invoke cCKF track finding with " << initialParameters.size()
                                               << " seeds.");

  unsigned int nSeed = 0;
  std::unordered_map<SeedIdentifier, bool> discoveredSeeds;

  auto addTrack = [&](const TrackProxy& track) {
    ++m_nFoundTracks;

    if (m_cfg.trimTracks) {
      Acts::trimTrack(track, true, true, true, true);
    }
    Acts::calculateTrackQuantities(track);

    if (m_trackSelector.has_value() && !m_trackSelector->isValidTrack(track)) {
      return;
    }

    visitSeedIdentifiers(track, [&](const SeedIdentifier& seedIdentifier) {
      if (auto it = discoveredSeeds.find(seedIdentifier);
          it != discoveredSeeds.end()) {
        it->second = true;
      }
    });

    ++m_nSelectedTracks;

    auto destProxy = tracks.makeTrack();
    destProxy.copyFrom(track);
  };

  if (seeds != nullptr && m_cfg.seedDeduplication) {
    for (const auto& seed : *seeds) {
      SeedIdentifier seedIdentifier = makeSeedIdentifier(seed);
      discoveredSeeds.emplace(seedIdentifier, false);
    }
  }

  for (std::size_t iSeed = 0; iSeed < initialParameters.size(); ++iSeed) {
    m_nTotalSeeds++;

    if (seeds != nullptr) {
      const ConstSeedProxy seed = seeds->at(iSeed);

      if (m_cfg.seedDeduplication) {
        SeedIdentifier seedIdentifier = makeSeedIdentifier(seed);
        if (auto it = discoveredSeeds.find(seedIdentifier);
            it != discoveredSeeds.end() && it->second) {
          m_nDeduplicatedSeeds++;
          ACTS_VERBOSE("Skipping seed " << iSeed << " due to deduplication.");
          continue;
        }
      }

      if (m_cfg.stayOnSeed) {
        if (cckfSelector) {
          cckfMeasSelAdapter.setSeed(seed);
        } else {
          fallbackAdapter->setSeed(seed);
        }
      }
    }

    tracksTemp.clear();

    const Acts::BoundTrackParameters& firstInitialParameters =
        initialParameters.at(iSeed);
    ACTS_VERBOSE("Processing seed " << iSeed << " with initial parameters "
                                    << firstInitialParameters);

    auto firstRootBranch = tracksTemp.makeTrack();
    initCckfColumns(firstRootBranch);

    auto firstResult = (*m_cfg.findTracks)(firstInitialParameters, firstOptions,
                                           tracksTemp, firstRootBranch);
    nSeed++;

    if (!firstResult.ok()) {
      m_nFailedSeeds++;
      ACTS_WARNING("Track finding failed for seed " << iSeed << " with error "
                                                    << firstResult.error());
      continue;
    }

    auto& firstTracksForSeed = firstResult.value();
    for (auto& firstTrack : firstTracksForSeed) {
      auto trackCandidate = tracksTemp.makeTrack();
      trackCandidate.copyFrom(firstTrack);

      Acts::Result<void> firstSmoothingResult{
          Acts::smoothTrack(ctx.geoContext, trackCandidate, logger())};
      if (!firstSmoothingResult.ok()) {
        m_nFailedSmoothing++;
        ACTS_ERROR("First smoothing for seed "
                   << iSeed << " and track " << firstTrack.index()
                   << " failed with error " << firstSmoothingResult.error());
        continue;
      }

      std::size_t nSecond = 0;
      seedNumber(trackCandidate) = nSeed - 1;

      if (m_cfg.twoWay) {
        std::optional<Acts::VectorMultiTrajectory::TrackStateProxy>
            firstMeasurementOpt;
        for (auto trackState : trackCandidate.trackStatesReversed()) {
          if (trackState.typeFlags().isMeasurement()) {
            firstMeasurementOpt = trackState;
          }
        }

        if (firstMeasurementOpt.has_value()) {
          TrackContainer::TrackStateProxy firstMeasurement{
              firstMeasurementOpt.value()};
          TrackContainer::ConstTrackStateProxy firstMeasurementConst{
              firstMeasurement};

          Acts::BoundTrackParameters secondInitialParameters =
              trackCandidate.createParametersFromState(firstMeasurementConst);

          if (!secondInitialParameters.referenceSurface().insideBounds(
                  secondInitialParameters.localPosition())) {
            m_nSkippedSecondPass++;
            ACTS_DEBUG(
                "Smoothing of first pass produced out-of-bounds parameters. "
                "Skipping second pass.");
            continue;
          }

          auto secondRootBranch = tracksTemp.makeTrack();
          secondRootBranch.copyFromWithoutStates(trackCandidate);
          initCckfColumns(secondRootBranch);

          auto secondResult =
              (*m_cfg.findTracks)(secondInitialParameters, secondOptions,
                                  tracksTemp, secondRootBranch);

          if (!secondResult.ok()) {
            ACTS_WARNING("Second track finding failed for seed "
                         << iSeed << " with error " << secondResult.error());
          } else {
            auto originalFirstMeasurementPrevious =
                firstMeasurement.previous();

            auto& secondTracksForSeed = secondResult.value();
            for (auto& secondTrack : secondTracksForSeed) {
              auto secondTrackCopy = tracksTemp.makeTrack();
              secondTrackCopy.copyFrom(secondTrack);

              secondTrackCopy.reverseTrackStates(true);

              firstMeasurement.previous() =
                  secondTrackCopy.outermostTrackState().index();

              auto tipIndex = trackCandidate.tipIndex();
              auto stemIndex = trackCandidate.stemIndex();
              trackCandidate.copyFromWithoutStates(secondTrackCopy);
              trackCandidate.tipIndex() = tipIndex;
              trackCandidate.stemIndex() = stemIndex;

              bool doExtrapolate = true;

              if (!m_cfg.reverseSearch) {
                doExtrapolate = !trackCandidate.hasReferenceSurface();
              } else {
                auto secondSmoothingResult =
                    Acts::smoothTrack(ctx.geoContext, trackCandidate, logger());
                if (!secondSmoothingResult.ok()) {
                  m_nFailedSmoothing++;
                  ACTS_ERROR("Second smoothing for seed "
                             << iSeed << " and track " << secondTrack.index()
                             << " failed with error "
                             << secondSmoothingResult.error());
                  continue;
                }
                trackCandidate.reverseTrackStates(true);
              }

              if (doExtrapolate) {
                auto secondExtrapolationResult =
                    Acts::extrapolateTrackToReferenceSurface(
                        trackCandidate, *pSurface, extrapolator,
                        extrapolationOptions, m_cfg.extrapolationStrategy,
                        logger());
                if (!secondExtrapolationResult.ok()) {
                  m_nFailedExtrapolation++;
                  ACTS_ERROR("Second extrapolation for seed "
                             << iSeed << " and track " << secondTrack.index()
                             << " failed with error "
                             << secondExtrapolationResult.error());
                  continue;
                }
              }

              addTrack(trackCandidate);
              ++nSecond;
            }

            firstMeasurement.previous() = originalFirstMeasurementPrevious;
          }
        }
      }

      if (nSecond == 0) {
        auto tipIndex = trackCandidate.tipIndex();
        auto stemIndex = trackCandidate.stemIndex();
        trackCandidate.copyFromWithoutStates(firstTrack);
        trackCandidate.tipIndex() = tipIndex;
        trackCandidate.stemIndex() = stemIndex;

        auto firstExtrapolationResult =
            Acts::extrapolateTrackToReferenceSurface(
                trackCandidate, *pSurface, extrapolator, extrapolationOptions,
                m_cfg.extrapolationStrategy, logger());
        if (!firstExtrapolationResult.ok()) {
          m_nFailedExtrapolation++;
          ACTS_ERROR("Extrapolation for seed "
                     << iSeed << " and track " << firstTrack.index()
                     << " failed with error "
                     << firstExtrapolationResult.error());
          continue;
        }

        addTrack(trackCandidate);
      }
    }
  }

  // ---- Shared hits ----
  if (m_cfg.computeSharedHits) {
    computeSharedHits(tracks, measurements);
  }

  ACTS_DEBUG("Finalized cCKF track finding with " << tracks.size()
                                                   << " track candidates.");

  // ---- Aggregate stopped-branch count ----
  if (cckfStopper) {
    m_nStoppedBranches += cckfStopperWrapper.m_nStoppedBranches;
  }

  // ---- Memory statistics ----
  m_memoryStatistics.local().hist +=
      tracks.trackStateContainer().statistics().hist;

  // ---- Timing output ----
  writeTimingCsv(m_cfg.outputTimingPath, ctx.eventNumber, timers,
                 initialParameters.size(), tracks.size());

  // ---- Build const output ----
  auto constTrackStateContainer =
      std::make_shared<Acts::ConstVectorMultiTrajectory>(
          std::move(*trackStateContainer));
  auto constTrackContainer = std::make_shared<Acts::ConstVectorTrackContainer>(
      std::move(*trackContainer));
  ConstTrackContainer constTracks{constTrackContainer,
                                  constTrackStateContainer};

  m_outputTracks(ctx, std::move(constTracks));
  return ProcessCode::SUCCESS;
}

// ============================================================================
// makeTrackFinderFunction
// ============================================================================

namespace {

using Stepper = Acts::SympyStepper;
using Navigator = Acts::Navigator;
using Propagator = Acts::Propagator<Stepper, Navigator>;
using CKF =
    Acts::CombinatorialKalmanFilter<Propagator, ActsExamples::TrackContainer>;

struct CckfTrackFinderFunctionImpl
    : public CckfTrackFindingAlgorithm::TrackFinderFunction {
  CKF trackFinder;

  explicit CckfTrackFinderFunctionImpl(CKF&& f)
      : trackFinder(std::move(f)) {}

  CckfTrackFindingAlgorithm::TrackFinderResult operator()(
      const ActsExamples::TrackParameters& initialParameters,
      const CckfTrackFindingAlgorithm::TrackFinderOptions& options,
      ActsExamples::TrackContainer& tracks,
      ActsExamples::TrackProxy rootBranch) const override {
    return trackFinder.findTracks(initialParameters, options, tracks,
                                  rootBranch);
  }
};

}  // namespace

std::shared_ptr<CckfTrackFindingAlgorithm::TrackFinderFunction>
CckfTrackFindingAlgorithm::makeTrackFinderFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    const Acts::Logger& logger) {
  Stepper stepper(std::move(magneticField));
  Navigator::Config cfg{std::move(trackingGeometry)};
  cfg.resolvePassive = false;
  cfg.resolveMaterial = true;
  cfg.resolveSensitive = true;
  Navigator navigator(cfg, logger.cloneWithSuffix("Navigator"));
  Propagator propagator(std::move(stepper), std::move(navigator),
                        logger.cloneWithSuffix("Propagator"));
  CKF trackFinder(std::move(propagator), logger.cloneWithSuffix("Finder"));

  return std::make_shared<CckfTrackFinderFunctionImpl>(std::move(trackFinder));
}

// ============================================================================
// finalize
// ============================================================================

ProcessCode CckfTrackFindingAlgorithm::finalize() {
  ACTS_INFO("CckfTrackFindingAlgorithm statistics:");
  ACTS_INFO("- total seeds: " << m_nTotalSeeds);
  ACTS_INFO("- deduplicated seeds: " << m_nDeduplicatedSeeds);
  ACTS_INFO("- failed seeds: " << m_nFailedSeeds);
  ACTS_INFO("- failed smoothing: " << m_nFailedSmoothing);
  ACTS_INFO("- failed extrapolation: " << m_nFailedExtrapolation);
  ACTS_INFO("- failure ratio seeds: "
            << static_cast<double>(m_nFailedSeeds) / m_nTotalSeeds);
  ACTS_INFO("- found tracks: " << m_nFoundTracks);
  ACTS_INFO("- selected tracks: " << m_nSelectedTracks);
  ACTS_INFO("- stopped branches: " << m_nStoppedBranches);
  ACTS_INFO("- skipped second pass: " << m_nSkippedSecondPass);

  auto memoryStatistics =
      m_memoryStatistics.combine([](const auto& a, const auto& b) {
        Acts::VectorMultiTrajectory::Statistics c;
        c.hist = a.hist + b.hist;
        return c;
      });
  std::stringstream ss;
  memoryStatistics.toStream(ss);
  ACTS_DEBUG("Track State memory statistics (averaged):\n" << ss.str());
  return ProcessCode::SUCCESS;
}

// ============================================================================
// computeSharedHits (identical to TrackFindingAlgorithm)
// ============================================================================

void CckfTrackFindingAlgorithm::computeSharedHits(
    TrackContainer& tracks, const MeasurementSubset& measurements) const {
  std::vector<std::size_t> firstTrackOnTheHit(
      measurements.container().size(), std::numeric_limits<std::size_t>::max());
  std::vector<std::size_t> firstStateOnTheHit(
      measurements.container().size(), std::numeric_limits<std::size_t>::max());

  for (auto track : tracks) {
    for (auto state : track.trackStatesReversed()) {
      if (!state.typeFlags().isMeasurement()) {
        continue;
      }

      std::size_t hitIndex = state.getUncalibratedSourceLink()
                                 .template get<IndexSourceLink>()
                                 .index();

      if (firstTrackOnTheHit.at(hitIndex) ==
          std::numeric_limits<std::size_t>::max()) {
        firstTrackOnTheHit.at(hitIndex) = track.index();
        firstStateOnTheHit.at(hitIndex) = state.index();
        continue;
      }

      std::size_t indexFirstTrack = firstTrackOnTheHit.at(hitIndex);
      std::size_t indexFirstState = firstStateOnTheHit.at(hitIndex);

      auto firstState = tracks.getTrack(indexFirstTrack)
                            .container()
                            .trackStateContainer()
                            .getTrackState(indexFirstState);
      firstState.typeFlags().setIsSharedHit();

      state.typeFlags().setIsSharedHit();
    }
  }
}

}  // namespace ActsExamples
