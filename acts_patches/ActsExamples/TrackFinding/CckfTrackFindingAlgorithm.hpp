// CckfTrackFindingAlgorithm.hpp — ACTS Algorithm that wires the cCKF
// components (gate MLP, value function, calibrated columns) into the
// CombinantorialKalmanFilter.
//
// This is a modified copy of TrackFindingAlgorithm: same CKF engine,
// same propagator, same track container types, but with:
//   - CckfMeasurementSelector  replacing Acts::MeasurementSelector (gate MLP)
//   - CckfBranchStopper        replacing the default BranchStopper (value MLP)
//   - Per-branch dynamic columns for gate log-odds accumulation
//   - Per-event timing instrumentation via CckfTimers
//   - ClusterContainer read from the whiteboard for gate features
//
// Part of the cCKF project (calibrated CKF).

#pragma once

#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/TrackFinding/MeasurementSelector.hpp"
#include "Acts/TrackFinding/TrackSelector.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"
#include "ActsExamples/EventData/Cluster.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/Seed.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "cckf/SensorLookup.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <tbb/combinable.h>
#pragma GCC diagnostic pop

namespace ActsExamples {

/// CKF-based track finding with learned gate and value function.
///
/// Drop-in replacement for TrackFindingAlgorithm in the ACTS Examples
/// sequencer. Reads the same inputs (measurements, initial track parameters,
/// seeds) and writes the same output (ConstTrackContainer). Additionally reads
/// ClusterContainer for the gate MLP's cluster features.
class CckfTrackFindingAlgorithm final : public IAlgorithm {
 public:
  using TrackFinderOptions =
      Acts::CombinatorialKalmanFilterOptions<TrackContainer>;
  using TrackFinderResult =
      Acts::Result<std::vector<TrackContainer::TrackProxy>>;

  /// Type-erased CKF function. Identical interface to
  /// TrackFindingAlgorithm::TrackFinderFunction — same CKF, same propagator,
  /// different extensions wired at call time via TrackFinderOptions.
  class TrackFinderFunction {
   public:
    virtual ~TrackFinderFunction() = default;
    virtual TrackFinderResult operator()(const TrackParameters&,
                                         const TrackFinderOptions&,
                                         TrackContainer&, TrackProxy) const = 0;
  };

  /// Create the track finder function. Same implementation as
  /// TrackFindingAlgorithm::makeTrackFinderFunction (SympyStepper +
  /// Navigator + CKF), but returns CckfTrackFindingAlgorithm's own
  /// TrackFinderFunction type.
  static std::shared_ptr<TrackFinderFunction> makeTrackFinderFunction(
      std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
      std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
      const Acts::Logger& logger);

  struct Config {
    // ---- Whiteboard keys (same as TrackFindingAlgorithm) ----

    /// Input measurements collection.
    std::string inputMeasurements;
    /// Input initial track parameter estimates for each seed.
    std::string inputInitialTrackParameters;
    /// Input seeds (optional; enables deduplication / stay-on-seed).
    std::string inputSeeds;
    /// Output track collection.
    std::string outputTracks;

    // ---- Geometry / field / finder ----

    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry;
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField;

    /// Type-erased CKF function (from makeTrackFinderFunction).
    std::shared_ptr<TrackFinderFunction> findTracks;
    /// Fallback chi2 measurement selector config (used only when
    /// gateWeightsPath is empty, for ablation without the gate).
    Acts::MeasurementSelector::Config measurementSelectorCfg;
    /// Track selector config (same as TrackFindingAlgorithm).
    std::optional<std::variant<Acts::TrackSelector::Config,
                               Acts::TrackSelector::EtaBinnedConfig>>
        trackSelectorCfg = std::nullopt;

    // ---- Propagation / CKF behaviour ----

    unsigned int maxSteps = 100000;
    Acts::TrackExtrapolationStrategy extrapolationStrategy =
        Acts::TrackExtrapolationStrategy::firstOrLast;
    bool twoWay = true;
    bool reverseSearch = false;
    bool seedDeduplication = false;
    bool stayOnSeed = false;
    bool computeSharedHits = false;
    bool trimTracks = true;
    bool useJosephFormulation = false;

    std::vector<std::uint32_t> pixelVolumeIds;
    std::vector<std::uint32_t> stripVolumeIds;
    std::size_t maxPixelHoles = std::numeric_limits<std::size_t>::max();
    std::size_t maxStripHoles = std::numeric_limits<std::size_t>::max();

    std::vector<std::uint32_t> constrainToVolumeIds;
    std::vector<std::uint32_t> endOfWorldVolumeIds;

    // ---- cCKF-specific ----

    /// Path to the gate MLP weight blob. If empty, falls back to the
    /// standard chi2 measurement selector (m_cfg.measurementSelectorCfg).
    std::string gateWeightsPath;
    /// Path to the value function weight blob. If empty, uses a pass-
    /// through branch stopper (always Continue).
    std::string valueWeightsPath;
    /// Gate acceptance threshold (calibrated probability). Only
    /// candidates with P(same particle) >= gateThreshold pass.
    float gateThreshold = 0.5f;
    /// Value function prune threshold (sigmoid probability). Branches
    /// with P(completion) < valueThreshold are stopped.
    float valueThreshold = 0.1f;
    /// Maximum candidates to keep per surface after the gate.
    std::size_t gateMaxCandidates = 10;
    /// Hard chi2 ceiling: reject hits above this regardless of gate score.
    float gateChi2Ceiling = 15.0f;
    /// Spatial pre-filter window in units of sigma. Only candidates within
    /// this bounding box of the predicted measurement are scored by the
    /// gate MLP. 0 disables (all candidates on the surface are scored).
    float gateWindowNSigma = 0.0f;
    /// Whiteboard key for the ClusterContainer (needed for gate
    /// features: cluster size, charge, second moments).
    std::string inputClusters;
    /// If non-empty, per-event timing data is appended to this CSV file.
    std::string outputTimingPath;
    /// Digitization configuration path (for sensor property lookup).
    /// When set, SensorLookup is loaded once and used to populate
    /// pitch_u/pitch_v/thickness gate features per volume.
    std::string digiConfigPath;
  };

  /// Constructor. Validates config and initializes whiteboard handles.
  explicit CckfTrackFindingAlgorithm(
      Config config, std::unique_ptr<const Acts::Logger> logger = nullptr);

  /// Per-event execute: run CKF with cCKF components for all seeds.
  ProcessCode execute(const AlgorithmContext& ctx) const final;

  const Config& config() const { return m_cfg; }

 private:
  void computeSharedHits(TrackContainer& tracks,
                         const MeasurementSubset& measurements) const;
  ProcessCode finalize() override;

  Config m_cfg;
  std::optional<Acts::TrackSelector> m_trackSelector;
  std::unique_ptr<cckf::SensorLookup> m_sensorLookup;

  ReadDataHandle<MeasurementSubset> m_inputMeasurements{this,
                                                        "InputMeasurements"};
  ReadDataHandle<TrackParametersContainer> m_inputInitialTrackParameters{
      this, "InputInitialTrackParameters"};
  ReadDataHandle<SeedContainer> m_inputSeeds{this, "InputSeeds"};
  ReadDataHandle<ClusterContainer> m_inputClusters{this, "InputClusters"};

  WriteDataHandle<ConstTrackContainer> m_outputTracks{this, "OutputTracks"};

  mutable std::atomic<std::size_t> m_nTotalSeeds{0};
  mutable std::atomic<std::size_t> m_nDeduplicatedSeeds{0};
  mutable std::atomic<std::size_t> m_nFailedSeeds{0};
  mutable std::atomic<std::size_t> m_nFailedSmoothing{0};
  mutable std::atomic<std::size_t> m_nFailedExtrapolation{0};
  mutable std::atomic<std::size_t> m_nFoundTracks{0};
  mutable std::atomic<std::size_t> m_nSelectedTracks{0};
  mutable std::atomic<std::size_t> m_nStoppedBranches{0};
  mutable std::atomic<std::size_t> m_nSkippedSecondPass{0};

  mutable tbb::combinable<Acts::VectorMultiTrajectory::Statistics>
      m_memoryStatistics{[]() {
        auto mtj = std::make_shared<Acts::VectorMultiTrajectory>();
        return mtj->statistics();
      }};
};

}  // namespace ActsExamples
