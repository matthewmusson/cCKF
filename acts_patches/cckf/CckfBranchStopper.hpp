// cCKF/acts_patches/cckf/CckfBranchStopper.hpp
#pragma once

#include "CckfTimers.hpp"
#include "MlpInference.hpp"
#include "WeightBlob.hpp"

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/ProxyAccessor.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilterExtensions.hpp"
#include "Acts/Utilities/HashedString.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace cckf {

/// Dynamic column names for the per-branch value-function accumulators.
///
/// These live on the TRACK container (i.e. one value per branch, carried
/// forward as the branch is extended/forked), not the track state
/// container: `sum`/`min` accumulate the gate's raw logit across every step
/// of the branch so far, `x0` accumulates path length in units of radiation
/// length, and `step` is the current CKF layer index. Something upstream of
/// this class (the CKF actor / CckfMeasurementSelector wiring, out of scope
/// for this header -- see Task 3 report) is responsible for registering
/// these as dynamic float columns on the track container and updating them
/// each time a new track state is appended to a branch, mirroring how
/// `nMeasurements`/`nHoles` are maintained natively by the CKF actor. This
/// header only *reads* them.
namespace CckfColumns {
constexpr std::string_view kSumGateLogOdds = "cckf_sum_gate_logodds";
constexpr std::string_view kMinGateLogOdds = "cckf_min_gate_logodds";
constexpr std::string_view kAccumulatedX0 = "cckf_accumulated_x0";
constexpr std::string_view kStepK = "cckf_step_k";
}  // namespace CckfColumns

/// Drop-in replacement for the ACTS default `BranchStopper` delegate that
/// runs the value function MLP (calibrated-free: sigmoid on the raw logit,
/// no Platt calibration -- see design note 4 in the Task 3 brief) instead of
/// hand-tuned hole/branch caps.
///
/// Not thread-safe, for the same reason as `CckfMeasurementSelector`: the
/// owned `MlpInference` keeps mutable scratch buffers. Each CKF worker
/// thread must own its own `CckfBranchStopper` instance.
class CckfBranchStopper {
 public:
  using BranchStopperResult = Acts::CombinatorialKalmanFilterBranchStopperResult;

  struct Config {
    /// Value function prune threshold (raw sigmoid probability, no Platt
    /// calibration). Branches with P(completion) < valueThreshold are
    /// stopped.
    float valueThreshold = 0.1f;
    /// Minimum measurements before the value function is even evaluated;
    /// below this, always Continue (mirrors ACTS's own reference
    /// BranchStopper, which never stops a branch before minMeasurements
    /// worth of evidence has accumulated).
    std::size_t minMeasurementsBeforePrune = 3;
    /// Minimum measurements required for a pruned branch to be kept as a
    /// track candidate (StopAndKeep) rather than discarded outright
    /// (StopAndDrop).
    std::size_t minMeasurementsForKeep = 6;
    /// Path to value function weight blob.
    std::string valueWeightsPath;
  };

  /// Default-constructs into a state with no value MLP loaded. operator()
  /// must not be called on a default-constructed instance (m_valueInference
  /// is null) -- this constructor exists only so the type can live in
  /// containers / be assigned into before configuration (same convention as
  /// CckfMeasurementSelector's default constructor).
  CckfBranchStopper() = default;

  explicit CckfBranchStopper(const Config& config, CckfTimers* timers)
      : m_config(config),
        m_timers(timers),
        // WeightBlob::load(...) is a prvalue passed straight into
        // MlpInference's by-value constructor (guaranteed elision / move) --
        // no separate WeightBlob member here, same fix as Task 1/2 applied
        // to avoid keeping a second, unused copy of the blob alive for the
        // object's lifetime.
        m_valueInference(
            std::make_unique<MlpInference>(WeightBlob::load(config.valueWeightsPath))) {}

  /// Number of branches this instance has stopped via the value function
  /// (diagnostic only). Plain counter, not atomic: BranchStopper instances
  /// are per-thread / per-event, so there is no concurrent writer -- an
  /// atomic here would only add overhead and (being non-copyable) would
  /// make the class harder to store in containers that copy/move it during
  /// setup.
  std::size_t nStoppedBranches() const { return m_nStoppedBranches; }

  template <typename TrackProxy, typename TrackStateProxy>
  BranchStopperResult operator()(const TrackProxy& track,
                                  const TrackStateProxy& trackState) const {
    // Always allow the first few measurements through -- not enough branch
    // history yet for the value function's prediction to be meaningful.
    if (track.nMeasurements() < m_config.minMeasurementsBeforePrune) {
      return BranchStopperResult::Continue;
    }

    // Build the 11-dim value feature vector. Order matches
    // cckf/features.py::VALUE_FEATURES exactly:
    //   [eta, state_qop, sigma2_l0, sigma2_l1, n_hits, n_holes, n_seq_holes,
    //    sum_gate_logodds, min_gate_logodds, step_k, x0_accumulated]
    float features[11];

    // Best available state/covariance: TrackStateProxy::parameters()/
    // covariance() already implement exactly the "filtered, falling back to
    // predicted" precedence this needs (smoothed() first if present, which
    // never is during the forward filtering pass this runs in; then
    // filtered(), set once the Kalman update has run for a measurement
    // state; then predicted(), the only thing hole states ever get). Using
    // the built-in accessor instead of hand-rolling the same fallback
    // avoids a std::hasFiltered()-ternary of two Eigen map types that must
    // otherwise be kept in lockstep by hand.
    const auto params = trackState.parameters();
    const auto cov = trackState.covariance();

    // eta = -log(tan(theta/2)), theta clipped away from the poles. Matches
    // cckf/features.py::eta_from_theta exactly (_THETA_EPS = 1e-6) rather
    // than Acts::AngleHelpers::etaFromTheta, which returns +/-infinity at
    // the same poles instead of a large finite value -- using the ACTS
    // helper here would feed inf into the (standardized) MLP input and
    // silently break inference for very forward/backward tracks, a
    // train/inference mismatch the training pipeline's clipping was
    // specifically designed to avoid.
    constexpr float kThetaEps = 1e-6f;
    constexpr float kPi = 3.14159265358979323846f;
    float theta = static_cast<float>(params[Acts::eBoundTheta]);
    theta = std::clamp(theta, kThetaEps, kPi - kThetaEps);
    float eta = -std::log(std::tan(theta / 2.0f));

    features[0] = eta;
    features[1] = static_cast<float>(params[Acts::eBoundQOverP]);
    features[2] = static_cast<float>(cov(Acts::eBoundLoc0, Acts::eBoundLoc0));
    features[3] = static_cast<float>(cov(Acts::eBoundLoc1, Acts::eBoundLoc1));
    features[4] = static_cast<float>(track.nMeasurements());
    features[5] = static_cast<float>(track.nHoles());

    // n_seq_holes: count consecutive holes walking back from the tip
    // (== trackState, since the actor sets the branch's tip index to the
    // just-added state before calling the branch stopper -- see
    // CombinatorialKalmanFilter.hpp). trackStatesReversed() is the
    // idiomatic ACTS traversal for this (used identically in
    // TrackFindingAlgorithm.cpp); previous() returns a raw
    // TrackIndexType, not a proxy, so chaining `.previous()` directly as if
    // it returned another track state proxy would not compile / would be
    // wrong.
    uint32_t nSeqHoles = 0;
    for (const auto& state : track.trackStatesReversed()) {
      if (!state.typeFlags().isHole()) {
        break;
      }
      ++nSeqHoles;
    }
    features[6] = static_cast<float>(nSeqHoles);

    // Per-branch gate log-odds accumulators and step/X0 counters, read via
    // Acts::ConstProxyAccessor -- the same dynamic-column access pattern
    // ACTS's own reference BranchStopper uses for its per-branch
    // BranchState (Examples/Algorithms/TrackFinding/src/
    // TrackFindingAlgorithm.cpp), rather than calling
    // track.template component<float, key>() directly. Functionally
    // equivalent on a const TrackProxy&, but this is the established idiom
    // in this codebase and works uniformly whether TrackProxy is itself a
    // mutable- or const-flavored proxy instantiation.
    features[7] = kSumGateLogOddsAccessor(track);
    features[8] = kMinGateLogOddsAccessor(track);
    features[9] = kStepKAccessor(track);
    features[10] = kAccumulatedX0Accessor(track);

    // Value inference: raw logit -> sigmoid. No Platt calibration -- the
    // value function is not calibrated the way the gate is (design note 4).
    auto t0 = std::chrono::steady_clock::now();
    float logit = m_valueInference->forward(features);
    float prob = 1.0f / (1.0f + std::exp(-logit));
    auto t1 = std::chrono::steady_clock::now();
    if (m_timers != nullptr) {
      m_timers->value_inference.record(t0, t1);
    }

    if (prob < m_config.valueThreshold) {
      ++m_nStoppedBranches;
      bool enoughMeasurements = track.nMeasurements() >= m_config.minMeasurementsForKeep;
      return enoughMeasurements ? BranchStopperResult::StopAndKeep
                                : BranchStopperResult::StopAndDrop;
    }

    return BranchStopperResult::Continue;
  }

 private:
  static constexpr Acts::ConstProxyAccessor<float> kSumGateLogOddsAccessor{
      Acts::hashString(CckfColumns::kSumGateLogOdds)};
  static constexpr Acts::ConstProxyAccessor<float> kMinGateLogOddsAccessor{
      Acts::hashString(CckfColumns::kMinGateLogOdds)};
  static constexpr Acts::ConstProxyAccessor<float> kAccumulatedX0Accessor{
      Acts::hashString(CckfColumns::kAccumulatedX0)};
  static constexpr Acts::ConstProxyAccessor<float> kStepKAccessor{
      Acts::hashString(CckfColumns::kStepK)};

  Config m_config;
  std::unique_ptr<MlpInference> m_valueInference;
  CckfTimers* m_timers = nullptr;
  mutable std::size_t m_nStoppedBranches = 0;
};

}  // namespace cckf
