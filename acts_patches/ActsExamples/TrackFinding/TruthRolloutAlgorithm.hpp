// cCKF/acts_patches/ActsExamples/TrackFinding/TruthRolloutAlgorithm.hpp
//
// Tier-3 rollout executor (Tier 2 plumbing; the pi-dagger DEFINITION lives
// in cckf/tier3_walker.py and is ratified there).
//
// For each worklist row (a divergence or tip state found by the walker),
// seed the CKF at that state's filtered parameters (diagonal covariance --
// see the spec's covariance discussion) and run it with:
//   - measurementSelector = truth selector: keep only the candidate whose
//     measurement carries the branch majority particle; lowest-chi2
//     tie-break (same ratified rule as the walker);
//   - branchStopper = never stop: pi-dagger runs to detector exit.
// Output: per-rollout hit sequence CSV + the visited track states on the
// whiteboard (outputTracks) so RootTrackStatesWriter can log them for the
// DAgger-supplement ablation.
//
// See docs/superpowers/specs/2026-08-25-tier3-rollout-design.md.
#pragma once

#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/EventData/Measurement.hpp"

// Reuse the cCKF's type-erased CKF factory: same stepper, navigator and
// Kalman core; only the extensions differ, and those are wired per call.
#include "CckfTrackFindingAlgorithm.hpp"

#include "cckf/TruthRolloutSelector.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ActsExamples {

class TruthRolloutAlgorithm final : public IAlgorithm {
 public:
  /// One rollout start, parsed from the walker's worklist CSV.
  struct WorklistRow {
    std::uint64_t rolloutId = 0;
    std::uint64_t seedId = 0;
    std::uint32_t stepK = 0;
    std::uint64_t geometryId = 0;
    // Filtered bound parameters (loc0, loc1, phi, theta, q/p, t) and their
    // VARIANCES (err_*_flt squared). Off-diagonals are not logged by the
    // trackstates writer; the spec's truth-suffix comparison bounds the
    // resulting bias.
    double par[6] = {0, 0, 0, 0, 0, 0};
    double var[6] = {0, 0, 0, 0, 0, 0};
    /// Packed majority-particle barcode (expansion.py encode_particle_id).
    std::uint64_t majorityPid = 0;
  };

  struct Config {
    /// Input measurements collection (whiteboard key).
    std::string inputMeasurements;
    /// Output track collection: every rollout's visited states.
    std::string outputTracks;
    /// Directory holding per-event worklists,
    /// `event{NNNNNNNNN}-rollout-worklist.csv` (walker output).
    std::string worklistDir;
    /// Directory for per-event hit-sequence CSVs,
    /// `event{NNNNNNNNN}-rollout-hits.csv`.
    std::string outputDir;
    /// measurement-simhit-map / simhits CSVs for the truth lookup,
    /// per event in the same naming scheme as expansion.py.
    std::string csvDir;

    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry;
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField;
    std::shared_ptr<CckfTrackFindingAlgorithm::TrackFinderFunction> findTracks;

    unsigned int maxSteps = 100000;
    /// Cap on rollouts per event (0 = no cap). For smoke tests.
    std::size_t maxRollouts = 0;
  };

  TruthRolloutAlgorithm(Config cfg,
                        std::unique_ptr<const Acts::Logger> lgr);

  ProcessCode execute(const AlgorithmContext& ctx) const final;
  ProcessCode finalize() final;

  const Config& config() const { return m_cfg; }

 private:
  static std::vector<WorklistRow> readWorklist(const std::string& path);

  Config m_cfg;

  ReadDataHandle<MeasurementContainer> m_inputMeasurements{
      this, "InputMeasurements"};
  WriteDataHandle<ConstTrackContainer> m_outputTracks{this, "OutputTracks"};

  mutable std::atomic<std::size_t> m_nRollouts{0};
  mutable std::atomic<std::size_t> m_nSteps{0};
  mutable std::atomic<std::size_t> m_nHoles{0};
  mutable std::atomic<std::size_t> m_nFailed{0};
};

}  // namespace ActsExamples
