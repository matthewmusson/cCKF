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
    /// Hard chi2 ceiling. Currently plumbed from Python but NOT applied in
    /// select() -- removed in 822d655 so the gate, not a chi2 cut, makes the
    /// accept decision. Kept only so the existing pybind signature still
    /// binds.
    float chi2Ceiling = 15.0f;
    /// Spatial pre-filter width, in sigma. Only candidates inside the
    /// axis-aligned box
    ///
    ///     |r0| <= n*sqrt(S00)   and   |r1| <= n*sqrt(S11)  (2D only)
    ///
    /// are scored by the gate. This is the *same* region expansion.py used to
    /// build the training set (`compute_window_bounds`, WINDOW_N_DEFAULT=10),
    /// verified against all 32 expanded Parquets: zero violations across
    /// 259.9M trainable rows (experiments/LOG.md, 2026-08-21).
    ///
    /// It is deliberately a box and not the chi2 ellipse. The ellipse is
    /// strictly inscribed in the box (chi2 <= n^2 implies |r_i| <= n*sqrt(S_ii)),
    /// so cutting on chi2 discards ~14% of the candidates the gate was trained
    /// on -- and only on 2D sensors, since in 1D the box and ellipse coincide.
    /// That would be a silent pixel-only selection bias. Set 0 to disable.
    float gateWindowNSigma = 0.0f;
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

  /// Set the geometry context for coordinate transforms in buildFeatures.
  void setGeoContext(const Acts::GeometryContext& gc) { m_geoCtx = &gc; }

  /// Set the current seed index (for diagnostic trace of first N seeds).
  void setSeedIndex(uint32_t idx) { m_seedIndex = idx; }

  /// Accessor for the loaded weight blob (for diagnostic logging of
  /// standardization params, Platt params, etc.).
  const WeightBlob& weightBlob() const {
    return m_gateInference->blob();
  }

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

    // cCKF never emits outliers -- see the passedCandidates == 0 branch below.
    isOutlier = false;

    if (candidates.empty()) {
      // No measurements on the surface at all: the third and most benign hole
      // cause (spec §7.4 ACTION_HOLE_NO_MEASUREMENTS). Nothing to do with the
      // gate or the window.
      if (m_timers) {
        ++m_timers->gate_diag.n_hole_no_measurements;
        m_timers->measurement_selection.record(
            t0, std::chrono::steady_clock::now());
      }
      return Result::success(std::pair(candidates.begin(), candidates.end()));
    }

    // Spatial pre-filter. TrackStateCreator hands us EVERY hit on the surface
    // with no spatial cut, so without this the gate is queried far outside the
    // region it was trained on: measured over a traced run, the median chi2 of
    // an unfiltered surface candidate is ~19,600, whereas the median inside the
    // training box is ~30 (experiments/LOG.md, 2026-08-21).
    //
    // DECISION: S is computed per candidate (its own measurement noise V),
    // whereas expansion.py reused one per-state S -- the CKF-selected hit's --
    // for every candidate on the surface. Per-candidate is the correct
    // innovation covariance, so this deliberately does not reproduce the
    // training window bit-for-bit. The difference is exactly the spread of V
    // across clusters on a module.
    std::vector<Innovation> inns(candidates.size());
    std::vector<bool> inWindow(candidates.size(), true);
    std::size_t nInWindow = candidates.size();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      inns[i] = computeInnovation(candidates[i]);
    }
    if (m_config.gateWindowNSigma > 0.0f) {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!passesBox(inns[i], m_config.gateWindowNSigma)) {
          inWindow[i] = false;
          --nInWindow;
        }
      }
      if (m_timers) {
        m_timers->gate_diag.n_window_prefiltered +=
            static_cast<int64_t>(candidates.size()) - static_cast<int64_t>(nInWindow);
      }
      if (nInWindow == 0) {
        // Measurements existed on this surface but none fell in the window.
        // Spec §7.4 calls this a "window failure" and it is a genuine hole:
        // returning an empty range makes the CKF call addNonSourcelinkState,
        // increment nHoles(), and run the branch stopper.
        if (m_timers) {
          ++m_timers->gate_diag.n_hole_window_failure;
          m_timers->measurement_selection.record(
              t0, std::chrono::steady_clock::now());
        }
        return Result::success(std::pair(candidates.begin(), candidates.begin()));
      }
    }

    const float n_window = static_cast<float>(nInWindow);
    const float log_n_window = std::log(std::max(n_window, 1.0f));

    // Score each candidate with the gate MLP.
    // Raw logits are stored alongside calibrated scores so the branch stopper
    // can read them via lastGateLogit() after measurement acceptance.
    std::vector<float> scores(candidates.size(), -1.0f);
    std::vector<float> rawLogits(candidates.size(), 0.0f);
    std::vector<float> chi2s(candidates.size(), 0.0f);
    // Per-candidate feature cache for diagnostic sampling of accepted hits.
    std::vector<std::array<float, 26>> featCache(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (!inWindow[i]) continue;
      auto& ts = candidates[i];

      const float chi2 = inns[i].chi2;
      ts.chi2() = chi2;

      // Build feature vector
      float features[26];
      auto t_feat_start = std::chrono::steady_clock::now();
      buildFeatures(ts, inns[i], n_window, features);
      auto t_feat_end = std::chrono::steady_clock::now();
      if (m_timers) {
        m_timers->gate_feature_build.record(t_feat_start, t_feat_end);
        // Check for NaN/inf in feature vector
        for (int j = 0; j < 26; ++j) {
          if (!std::isfinite(features[j])) {
            ++m_timers->gate_diag.n_nan_features;
            break;
          }
        }
      }

      // Cache raw features for diagnostic sampling
      std::copy(features, features + 26, featCache[i].begin());

      // Gate inference: split forward + calibrate to capture raw logit
      auto t_gate_start = std::chrono::steady_clock::now();
      float logit = m_gateInference->forward(features);
      scores[i] = m_gateInference->calibrate(logit, log_n_window);
      rawLogits[i] = logit;
      chi2s[i] = chi2;
      auto t_gate_end = std::chrono::steady_clock::now();
      if (m_timers) {
        m_timers->gate_inference.record(t_gate_start, t_gate_end);
      }
    }

    // Partition: candidates passing the gate threshold to the front.
    // scores and rawLogits are kept in lockstep with candidates via the
    // paired swap below, so scores[i]/rawLogits[i] always corresponds to
    // candidates[i] after this loop.
    std::size_t passedCandidates = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (scores[i] >= m_config.gateThreshold) {
        if (passedCandidates != i) {
          std::swap(candidates[passedCandidates], candidates[i]);
          std::swap(scores[passedCandidates], scores[i]);
          std::swap(rawLogits[passedCandidates], rawLogits[i]);
          std::swap(chi2s[passedCandidates], chi2s[i]);
          std::swap(featCache[passedCandidates], featCache[i]);
        }
        ++passedCandidates;
      }
    }

    // Accumulate chi2/score diagnostics for accepted vs rejected candidates.
    if (m_timers) {
      for (std::size_t i = 0; i < passedCandidates; ++i) {
        m_timers->gate_diag.sum_chi2_accepted += chi2s[i];
        m_timers->gate_diag.sum_score_accepted += scores[i];
        if (chi2s[i] > m_timers->gate_diag.max_chi2_accepted) {
          m_timers->gate_diag.max_chi2_accepted = chi2s[i];
        }
      }
      m_timers->gate_diag.n_accepted += passedCandidates;
      for (std::size_t i = passedCandidates; i < candidates.size(); ++i) {
        m_timers->gate_diag.sum_chi2_rejected += chi2s[i];
        m_timers->gate_diag.sum_score_rejected += scores[i];
      }
      m_timers->gate_diag.n_rejected +=
          static_cast<int64_t>(candidates.size()) - passedCandidates;
      // Per-step chi2 aggregation (all seeds)
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        m_timers->per_step.recordAll(m_branchCtx.step_k, chi2s[i]);
      }
      for (std::size_t i = 0; i < passedCandidates; ++i) {
        m_timers->per_step.recordAccepted(m_branchCtx.step_k, chi2s[i]);
      }
      // Track-level trace: log ALL candidates' features for first N seeds
      m_timers->track_trace.addStep(
          m_seedIndex, m_branchCtx.step_k,
          featCache.data(), scores.data(), chi2s.data(),
          candidates.size(), passedCandidates);
    }

    if (passedCandidates == 0) {
      // In-window candidates existed but the gate rejected all of them, so
      // this is a hole -- NOT an outlier.
      //
      // The old code kept the min-chi2 hit as an outlier whenever
      // chi2 < chi2OutlierCutoff (100). With the n=10 window that condition
      // was very nearly always true, so gate rejection could essentially never
      // produce a hole: ACTS increments nOutliers() rather than nHoles() for an
      // outlier state, which left n_holes/n_seq_holes flat in the branch context
      // fed to the gate and hid the hole from the value function entirely.
      //
      // The outlier is a chi2-era escape hatch. In cCKF the gate IS the
      // decision, so "nothing passed" means hole. Returning an empty range
      // drives CombinatorialKalmanFilter's addNonSourcelinkState path ->
      // nHoles()++ -> branchStopper.
      if (m_timers) {
        ++m_timers->gate_diag.n_hole_gate_failure;
        m_timers->measurement_selection.record(
            t0, std::chrono::steady_clock::now());
      }
      return Result::success(
          std::pair(candidates.begin(), candidates.begin()));
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
  /// Everything the gate needs about one candidate's residual, computed in a
  /// single Eigen pass. Previously chi2 was computed twice (pre-filter, then
  /// scoring loop) and S a third time inside buildFeatures -- three matrix
  /// inversions per candidate on a path that accounts for ~88% of CKF wall
  /// time at 57M calls/event.
  struct Innovation {
    float r0 = 0.0f, r1 = 0.0f;      ///< residual m - Hx
    float S00 = 0.0f, S01 = 0.0f, S11 = 0.0f;  ///< innovation covariance
    float chi2 = 0.0f;               ///< r^T S^-1 r
    unsigned int dim = 0;            ///< calibratedSize(): 2 pixel, 1 strip
  };

  template <typename TrackStateProxy>
  Innovation computeInnovation(const TrackStateProxy& ts) const {
    // .eval() forces Eigen expression templates into concrete matrices,
    // avoiding dangling references to temporaries from proxy accessors.
    const auto H = ts.projectorSubspaceHelper().fullProjector().topLeftCorner(
        ts.calibratedSize(), Acts::eBoundSize).eval();
    // V is the *measurement* noise R_k (the calibrated measurement covariance
    // of this hit), not the process noise. Process noise Q_k is multiple
    // scattering / energy loss and is folded into predictedCovariance() during
    // propagation, so it reaches S through the H C H^T term below.
    const auto V = ts.effectiveCalibratedCovariance().eval();
    const auto S = (V + H * ts.predictedCovariance() * H.transpose()).eval();
    const auto residual = (ts.effectiveCalibrated() - H * ts.predicted()).eval();

    Innovation inn;
    inn.dim = static_cast<unsigned int>(ts.calibratedSize());
    inn.r0 = static_cast<float>(residual(0));
    inn.S00 = static_cast<float>(S(0, 0));
    // A 1D strip has no l1 coordinate at all -- S is 1x1 and S(1,1) does not
    // exist. Zero is the sentinel the feature builder already expects; the box
    // test below skips the l1 leg entirely when dim < 2.
    if (inn.dim >= 2) {
      inn.r1 = static_cast<float>(residual(1));
      inn.S01 = static_cast<float>(S(0, 1));
      inn.S11 = static_cast<float>(S(1, 1));
    }
    inn.chi2 = static_cast<float>(
        (residual.transpose() * S.inverse() * residual)(0, 0));
    return inn;
  }

  /// The n-sigma axis-aligned box from expansion.py::compute_window_bounds.
  /// Dimension-aware: on a 1D strip only the l0 leg is testable, and there the
  /// box coincides exactly with the chi2 ellipse.
  bool passesBox(const Innovation& inn, float n) const {
    if (!(inn.S00 > 0.0f)) {
      return false;  // degenerate S: no defensible window, drop the candidate
    }
    if (std::abs(inn.r0) > n * std::sqrt(inn.S00)) {
      return false;
    }
    if (inn.dim >= 2) {
      if (!(inn.S11 > 0.0f) || std::abs(inn.r1) > n * std::sqrt(inn.S11)) {
        return false;
      }
    }
    return true;
  }

  template <typename TrackStateProxy>
  void buildFeatures(const TrackStateProxy& ts, const Innovation& inn,
                     float n_window, float* out) const {
    // Residual and innovation covariance arrive precomputed from
    // computeInnovation() -- see the Innovation docstring for why.
    const float res0 = inn.r0, res1 = inn.r1;
    const float S00 = inn.S00, S01 = inn.S01, S11 = inn.S11;
    const float chi2 = inn.chi2;

    // .eval() forces Eigen expression templates into concrete matrices,
    // avoiding dangling references to temporaries from proxy accessors.
    // Still needed here: the incidence-angle block below transforms the
    // predicted bound parameters to free parameters.
    const auto predicted = ts.predicted().eval();

    // Cluster features
    float clus_s_u = 0, clus_s_v = 0, clus_q_tot = 0;
    float clus_sigma_uu = 0, clus_sigma_uv = 0, clus_sigma_vv = 0;
    float alpha_u = 0, alpha_v = 0;

    if (m_clusters == nullptr) {
      if (m_timers) ++m_timers->gate_diag.n_no_clusters_ptr;
    } else if (!ts.hasUncalibratedSourceLink()) {
      if (m_timers) ++m_timers->gate_diag.n_no_uncal_sl;
    }
    if (m_clusters != nullptr && ts.hasUncalibratedSourceLink()) {
      const Acts::SourceLink sl = ts.getUncalibratedSourceLink();
      const auto* isl = sl.template getPtr<ActsExamples::IndexSourceLink>();
      if (isl == nullptr) {
        if (m_timers) ++m_timers->gate_diag.n_null_isl;
      } else if (isl->index() >= m_clusters->size()) {
        if (m_timers) ++m_timers->gate_diag.n_index_oob;
      }
      if (isl != nullptr && isl->index() < m_clusters->size()) {
        if (m_timers) ++m_timers->gate_diag.n_cluster_ok;
        const auto& cluster = (*m_clusters)[isl->index()];
        clus_s_u = static_cast<float>(cluster.sizeLoc0);
        clus_s_v = static_cast<float>(cluster.sizeLoc1);
        clus_q_tot = static_cast<float>(cluster.sumActivations());
        // Compute charge-weighted second moments in CHANNEL-INDEX units
        // (matching expansion.py::compute_cluster_features). The shared
        // clusterChargeMoments() uses cell.path2D (physical mm²); training
        // data uses cell.bin (integer channel indices). Using the wrong
        // coordinate system shifts features 9-11 by pitch² (~64-400×).
        {
          double qTot = 0, muU = 0, muV = 0;
          for (const auto& cell : cluster.channels) {
            if (!(cell.activation > 0)) continue;
            qTot += cell.activation;
            muU += cell.activation * static_cast<double>(cell.bin[0]);
            muV += cell.activation * static_cast<double>(cell.bin[1]);
          }
          if (qTot > 0) {
            muU /= qTot;
            muV /= qTot;
            double sUU = 0, sUV = 0, sVV = 0;
            for (const auto& cell : cluster.channels) {
              if (!(cell.activation > 0)) continue;
              double du = static_cast<double>(cell.bin[0]) - muU;
              double dv = static_cast<double>(cell.bin[1]) - muV;
              sUU += cell.activation * du * du;
              sUV += cell.activation * du * dv;
              sVV += cell.activation * dv * dv;
            }
            clus_sigma_uu = static_cast<float>(sUU / qTot);
            clus_sigma_uv = static_cast<float>(sUV / qTot);
            clus_sigma_vv = static_cast<float>(sVV / qTot);
          }
        }
      }
    }

    // Incidence angles from predicted direction
    if (ts.hasPredicted() && m_geoCtx != nullptr) {
      const auto& surface = ts.referenceSurface();
      auto freeParams = Acts::transformBoundToFreeParameters(
          surface, *m_geoCtx, predicted);
      Acts::Vector3 direction = freeParams.template segment<3>(Acts::eFreeDir0);
      Acts::Vector3 position = freeParams.template segment<3>(Acts::eFreePos0);
      auto angles = ActsExamples::incidenceAngles(
          surface.referenceFrame(*m_geoCtx, position, direction),
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
  const Acts::GeometryContext* m_geoCtx = nullptr;
  uint32_t m_seedIndex = 0;

  /// Per-surface map from source-link index to the raw gate logit for
  /// each accepted candidate. Cleared at the start of each select() call.
  /// Mutable because select() is const (delegate requirement) but we need
  /// to populate this for the branch stopper to read.
  mutable std::unordered_map<std::size_t, float> m_acceptedLogits;
};

}  // namespace cckf
