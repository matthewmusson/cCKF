// cCKF/acts_patches/cckf/CckfMeasurementSelector.hpp
#pragma once

#include "CckfFeatures.hpp"
#include "CckfTimers.hpp"
#include "MlpInference.hpp"
#include "WeightBlob.hpp"

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/MeasurementHelpers.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/TransformationHelpers.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilterError.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "ActsExamples/EventData/Cluster.hpp"
#include "ActsExamples/EventData/ClusterFeatures.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cckf {

/// Drop-in replacement for Acts::MeasurementSelector that runs the gate MLP
/// (calibrated P(same particle)) instead of a chi2 threshold.
///
/// Not thread-safe: like MlpInference, a single instance must not be called
/// concurrently from multiple threads (mutable scratch state in the owned
/// MlpInference, and the branch-context/sensor-props setters below are
/// meant to be called from a single CKF actor thread immediately before each
/// select() call). ACTS CKF workers should each own their own instance.
class CckfMeasurementSelector {
 public:
  struct Config {
    /// Gate acceptance threshold (calibrated probability)
    float gateThreshold = 0.5f;
    /// Maximum candidates to keep per surface (same role as
    /// numMeasurementsCutOff)
    std::size_t maxCandidates = 10;
    /// Chi2 outlier cutoff (fallback when no candidate passes the gate)
    float chi2OutlierCutoff = 100.0f;
    /// Path to gate weight blob
    std::string gateWeightsPath;
  };

  /// Default-constructs into a state with no gate MLP loaded. select() must
  /// not be called on a default-constructed instance (m_gateInference is
  /// null) -- this constructor exists only so the type can live in
  /// containers / be assigned into before configuration.
  CckfMeasurementSelector() = default;

  explicit CckfMeasurementSelector(const Config& config,
                                   const ActsExamples::ClusterContainer* clusters,
                                   CckfTimers* timers)
      : m_config(config),
        m_clusters(clusters),
        m_timers(timers),
        // WeightBlob::load(...) returns a prvalue; MlpInference's by-value
        // constructor takes ownership of it directly (guaranteed elision /
        // move, no separate WeightBlob member needed here and nothing to
        // dangle -- see Task 1's fix for why storing WeightBlob by pointer
        // or keeping a second copy around is the footgun to avoid).
        m_gateInference(
            std::make_unique<MlpInference>(WeightBlob::load(config.gateWeightsPath))) {}

  /// Set the branch context before each surface's select() call.
  /// Called by the CKF actor (via CckfTrackFindingAlgorithm).
  void setBranchContext(const BranchContext& ctx) { m_branchCtx = ctx; }

  /// Set sensor properties for the current surface.
  void setSensorProps(const SensorProps& props) { m_sensorProps = props; }

  /// Look up the raw gate logit for a source-link index accepted in the most
  /// recent select() call. Returns std::nullopt if the index was not among
  /// the accepted candidates (e.g., it was pruned, or select() was not called
  /// with gateWeightsPath). The CckfBranchStopperWrapper uses this to
  /// update the sum/min gate log-odds dynamic columns on the track.
  std::optional<float> lastGateLogit(std::size_t sourceLinkIndex) const {
    auto it = m_acceptedLogits.find(sourceLinkIndex);
    if (it != m_acceptedLogits.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  /// Drop-in replacement for MeasurementSelector::select().
  /// Same signature so it can be connected to TrackStateCreator's delegate.
  template <typename traj_t>
  Acts::Result<std::pair<
      typename std::vector<typename traj_t::TrackStateProxy>::iterator,
      typename std::vector<typename traj_t::TrackStateProxy>::iterator>>
  select(std::vector<typename traj_t::TrackStateProxy>& candidates,
         bool& isOutlier, const Acts::Logger& logger) const {
    using Result = Acts::Result<std::pair<
        typename std::vector<typename traj_t::TrackStateProxy>::iterator,
        typename std::vector<typename traj_t::TrackStateProxy>::iterator>>;

    auto t0 = std::chrono::steady_clock::now();

    // Clear accepted-logit map from the previous surface's select() call.
    m_acceptedLogits.clear();

    if (candidates.empty()) {
      if (m_timers) {
        m_timers->measurement_selection.record(
            t0, std::chrono::steady_clock::now());
      }
      return Result::success(std::pair(candidates.begin(), candidates.end()));
    }

    const float n_window = static_cast<float>(candidates.size());
    const float log_n_window = std::log(std::max(n_window, 1.0f));

    float minChi2 = std::numeric_limits<float>::max();
    std::size_t minIndex = 0;

    // Score each candidate with the gate MLP.
    // Raw logits are stored alongside calibrated scores so the branch stopper
    // can read them via lastGateLogit() after measurement acceptance.
    std::vector<float> scores(candidates.size());
    std::vector<float> rawLogits(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto& ts = candidates[i];

      // Compute chi2 (same math as MeasurementSelector::calculateChi2)
      float chi2 = computeChi2(ts);
      ts.chi2() = chi2;

      if (chi2 < minChi2) {
        minChi2 = chi2;
        minIndex = i;
      }

      // Build feature vector
      float features[26];
      auto t_feat_start = std::chrono::steady_clock::now();
      buildFeatures(ts, chi2, n_window, features);
      auto t_feat_end = std::chrono::steady_clock::now();
      if (m_timers) {
        m_timers->gate_feature_build.record(t_feat_start, t_feat_end);
      }

      // Gate inference: split forward + calibrate to capture raw logit
      auto t_gate_start = std::chrono::steady_clock::now();
      float logit = m_gateInference->forward(features);
      scores[i] = m_gateInference->calibrate(logit, log_n_window);
      rawLogits[i] = logit;
      auto t_gate_end = std::chrono::steady_clock::now();
      if (m_timers) {
        m_timers->gate_inference.record(t_gate_start, t_gate_end);
      }
    }

    // Partition: candidates passing the gate threshold to the front.
    // scores and rawLogits are kept in lockstep with candidates via the
    // paired swap below, so scores[i]/rawLogits[i] always corresponds to
    // candidates[i] after this loop.
    isOutlier = false;
    std::size_t passedCandidates = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (scores[i] >= m_config.gateThreshold) {
        if (passedCandidates != i) {
          std::swap(candidates[passedCandidates], candidates[i]);
          std::swap(scores[passedCandidates], scores[i]);
          std::swap(rawLogits[passedCandidates], rawLogits[i]);
          if (minIndex == i) {
            minIndex = passedCandidates;
          } else if (minIndex == passedCandidates) {
            minIndex = i;
          }
        }
        ++passedCandidates;
      }
    }

    if (passedCandidates == 0) {
      // No candidate passed the gate -- fall back to outlier logic.
      // Store the logit for the outlier candidate so the branch stopper
      // can still update gate log-odds columns.
      if (minChi2 < m_config.chi2OutlierCutoff) {
        isOutlier = true;
        storeAcceptedLogit(candidates[minIndex], rawLogits[minIndex]);
        if (m_timers) {
          m_timers->measurement_selection.record(
              t0, std::chrono::steady_clock::now());
        }
        return Result::success(std::pair(candidates.begin() + minIndex,
                                         candidates.begin() + minIndex + 1));
      } else {
        if (m_timers) {
          m_timers->measurement_selection.record(
              t0, std::chrono::steady_clock::now());
        }
        return Result::success(
            std::pair(candidates.begin(), candidates.begin()));
      }
    }

    // Sort passed candidates by gate score descending.
    //
    // NOTE: this must NOT be done by comparing "current position of a/b in
    // candidates" against a fixed scores[] array (i.e. computing an index
    // via pointer arithmetic against &candidates[0] inside the comparator).
    // std::sort permutes the elements of `candidates` in place as it runs,
    // so a comparator that re-derives "my index" from current position and
    // then looks up scores[index] silently desyncs after the first internal
    // swap -- scores[k] was assigned to whichever candidate started at
    // position k, not to whatever the sort has since moved into position k.
    // Instead: sort a small index permutation against the untouched scores
    // array, then apply that permutation to candidates once.
    std::vector<std::size_t> order(passedCandidates);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&scores](std::size_t i, std::size_t j) {
                return scores[i] > scores[j];
              });

    std::vector<typename traj_t::TrackStateProxy> reordered;
    reordered.reserve(passedCandidates);
    for (std::size_t idx : order) {
      reordered.push_back(candidates[idx]);
    }
    std::copy(reordered.begin(), reordered.end(), candidates.begin());

    // Apply the same permutation to rawLogits for the accepted range.
    std::vector<float> reorderedLogits;
    reorderedLogits.reserve(passedCandidates);
    for (std::size_t idx : order) {
      reorderedLogits.push_back(rawLogits[idx]);
    }

    std::size_t nKeep = std::min(m_config.maxCandidates, passedCandidates);

    // Store logits for accepted candidates, keyed by source-link index.
    for (std::size_t i = 0; i < nKeep; ++i) {
      storeAcceptedLogit(candidates[i], reorderedLogits[i]);
    }

    if (m_timers) {
      m_timers->measurement_selection.record(t0,
                                             std::chrono::steady_clock::now());
    }

    return Result::success(
        std::pair(candidates.begin(), candidates.begin() + nKeep));
  }

 private:
  template <typename TrackStateProxy>
  float computeChi2(const TrackStateProxy& ts) const {
    // Reuse ACTS's chi2 computation via the calibrated residual: same
    // residual/innovation-covariance definition as
    // Acts::MeasurementSelector::calculateChi2, expressed through the dense
    // projector rather than the fixed-measdim subspace helper.
    const auto H = ts.projectorSubspaceHelper().fullProjector().topLeftCorner(
        ts.calibratedSize(), Acts::eBoundSize);
    const auto V = ts.effectiveCalibratedCovariance();
    const auto S = V + H * ts.predictedCovariance() * H.transpose();
    const auto residual = ts.effectiveCalibrated() - H * ts.predicted();
    return (residual.transpose() * S.inverse() * residual)(0, 0);
  }

  template <typename TrackStateProxy>
  void buildFeatures(const TrackStateProxy& ts, float chi2, float n_window,
                     float* out) const {
    // Residual
    const auto predicted = ts.predicted();
    const auto H = ts.projectorSubspaceHelper().fullProjector().topLeftCorner(
        ts.calibratedSize(), Acts::eBoundSize);
    const auto residual = ts.effectiveCalibrated() - H * predicted;
    float res0 = static_cast<float>(residual(0));
    float res1 =
        (ts.calibratedSize() >= 2) ? static_cast<float>(residual(1)) : 0.0f;

    // Innovation covariance S
    const auto V = ts.effectiveCalibratedCovariance();
    const auto S = V + H * ts.predictedCovariance() * H.transpose();
    float S00 = static_cast<float>(S(0, 0));
    float S01 = (ts.calibratedSize() >= 2) ? static_cast<float>(S(0, 1)) : 0.0f;
    float S11 = (ts.calibratedSize() >= 2) ? static_cast<float>(S(1, 1)) : 0.0f;

    // Cluster features
    float clus_s_u = 0, clus_s_v = 0, clus_q_tot = 0;
    float clus_sigma_uu = 0, clus_sigma_uv = 0, clus_sigma_vv = 0;
    float alpha_u = 0, alpha_v = 0;

    if (m_clusters != nullptr && ts.hasUncalibratedSourceLink()) {
      // getUncalibratedSourceLink() returns a SourceLink BY VALUE. Binding
      // it to a named local (rather than chaining .getPtr<T>() straight off
      // the temporary) keeps the SourceLink -- and the small-buffer storage
      // its getPtr<T>() points into -- alive for the rest of this scope.
      // Chaining directly off the temporary would return a pointer into
      // storage that is destroyed at the end of that one statement, making
      // every subsequent use of `isl` a dangling-pointer read (the same
      // class of bug Task 1 hit and fixed for MlpInference's WeightBlob).
      const Acts::SourceLink sl = ts.getUncalibratedSourceLink();
      const auto* isl = sl.template getPtr<ActsExamples::IndexSourceLink>();
      if (isl != nullptr && isl->index() < m_clusters->size()) {
        const auto& cluster = (*m_clusters)[isl->index()];
        clus_s_u = static_cast<float>(cluster.sizeLoc0);
        clus_s_v = static_cast<float>(cluster.sizeLoc1);
        clus_q_tot = static_cast<float>(cluster.sumActivations());
        auto moments = ActsExamples::clusterChargeMoments(cluster);
        clus_sigma_uu = static_cast<float>(moments.sigmaUU);
        clus_sigma_uv = static_cast<float>(moments.sigmaUV);
        clus_sigma_vv = static_cast<float>(moments.sigmaVV);
      }
    }

    // Incidence angles from predicted direction
    if (ts.hasPredicted()) {
      const auto& surface = ts.referenceSurface();
      auto freeParams = Acts::transformBoundToFreeParameters(
          surface, Acts::GeometryContext{}, predicted);
      Acts::Vector3 direction = freeParams.segment<3>(Acts::eFreeDir0);
      Acts::Vector3 position = freeParams.segment<3>(Acts::eFreePos0);
      auto angles = ActsExamples::incidenceAngles(
          surface.referenceFrame(Acts::GeometryContext{}, position, direction),
          direction);
      alpha_u = static_cast<float>(angles.alphaU);
      alpha_v = static_cast<float>(angles.alphaV);
    }

    cckf::buildGateFeatures(out, res0, res1, S00, S01, S11, chi2, clus_s_u,
                            clus_s_v, clus_q_tot, clus_sigma_uu, clus_sigma_uv,
                            clus_sigma_vv, alpha_u, alpha_v, n_window,
                            m_branchCtx, m_sensorProps);
  }

  /// Store the raw logit for an accepted candidate, keyed by its
  /// source-link index, so CckfBranchStopperWrapper can retrieve it.
  template <typename TrackStateProxy>
  void storeAcceptedLogit(const TrackStateProxy& ts, float logit) const {
    if (ts.hasUncalibratedSourceLink()) {
      const Acts::SourceLink sl = ts.getUncalibratedSourceLink();
      const auto* isl = sl.template getPtr<ActsExamples::IndexSourceLink>();
      if (isl != nullptr) {
        m_acceptedLogits[isl->index()] = logit;
      }
    }
  }

  Config m_config;
  const ActsExamples::ClusterContainer* m_clusters = nullptr;
  CckfTimers* m_timers = nullptr;
  std::unique_ptr<MlpInference> m_gateInference;
  BranchContext m_branchCtx;
  SensorProps m_sensorProps;

  /// Per-surface map from source-link index to the raw gate logit for
  /// each accepted candidate. Cleared at the start of each select() call.
  /// Mutable because select() is const (delegate requirement) but we need
  /// to populate this for the branch stopper to read.
  mutable std::unordered_map<std::size_t, float> m_acceptedLogits;
};

}  // namespace cckf
