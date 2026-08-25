// cCKF/acts_patches/ActsExamples/TrackFinding/TruthRolloutAlgorithm.cpp
// See header for design; spec:
// docs/superpowers/specs/2026-08-25-tier3-rollout-design.md

#include "TruthRolloutAlgorithm.hpp"

#include "cckf/TruthRolloutSelector.hpp"

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/TrackFinding/TrackStateCreator.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"

#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ActsExamples {

namespace {

/// Truth-greedy measurement selection. Keeps exactly the candidate whose
/// measurement carries the rollout's majority particle; among several
/// (module overlaps, shared clusters -- 0.011% of states on event 1) the
/// lowest-chi2 one, matching the RATIFIED rule in cckf/tier3_walker.py.
/// No window, no MLP: pi-dagger selects by identity, which is what makes
/// diagonal-seeded offline rollouts viable.
class TruthSelectorAdapter {
 public:
  using Traj = Acts::VectorMultiTrajectory;

  explicit TruthSelectorAdapter(const cckf::TruthRolloutContext* ctx)
      : m_ctx(ctx) {}

  Acts::Result<std::pair<std::vector<Traj::TrackStateProxy>::iterator,
                         std::vector<Traj::TrackStateProxy>::iterator>>
  select(std::vector<Traj::TrackStateProxy>& candidates, bool& isOutlier,
         const Acts::Logger& /*logger*/) const {
    isOutlier = false;
    std::size_t best = candidates.size();
    double bestChi2 = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto& ts = candidates[i];
      if (!ts.hasUncalibratedSourceLink()) {
        continue;
      }
      const auto measIndex =
          ts.getUncalibratedSourceLink().get<IndexSourceLink>().index();
      if (!m_ctx->isTruthMeasurement(measIndex)) {
        continue;
      }
      const double chi2 = candidateChi2(ts);
      if (chi2 < bestChi2) {
        bestChi2 = chi2;
        best = i;
      }
    }

    if (best == candidates.size()) {
      // No truth candidate on this surface: hole, continue propagating.
      return std::make_pair(candidates.begin(), candidates.begin());
    }
    if (best != 0) {
      std::swap(candidates[0], candidates[best]);
    }
    candidates[0].chi2() = static_cast<float>(bestChi2);
    return std::make_pair(candidates.begin(), std::next(candidates.begin()));
  }

 private:
  /// Plain chi2 of one candidate against its predicted state. Follows the
  /// .eval() discipline (see cCKF/CLAUDE.md, Eigen expression-template UB).
  static double candidateChi2(Traj::TrackStateProxy& ts) {
    return Acts::visit_measurement(
        ts.calibratedSize(), [&](auto N) -> double {
          constexpr std::size_t kM = decltype(N)::value;
          const auto meas = ts.template calibrated<kM>().eval();
          const auto measCov = ts.template calibratedCovariance<kM>().eval();
          const auto H = ts.projectorSubspaceHelper()
                             .template projector<kM>()
                             .eval();
          const auto predicted = ts.predicted().eval();
          const auto predictedCov = ts.predictedCovariance().eval();
          const auto res = (meas - H * predicted).eval();
          const auto S = (measCov + H * predictedCov * H.transpose()).eval();
          return (res.transpose() * S.inverse() * res).eval()(0, 0);
        });
  }

  const cckf::TruthRolloutContext* m_ctx;
};

/// pi-dagger never stops: rollouts run to the detector edge.
struct NeverStopBranchStopper {
  using BranchStopperResult =
      Acts::CombinatorialKalmanFilterBranchStopperResult;
  BranchStopperResult operator()(
      const TrackContainer::TrackProxy& /*track*/,
      const TrackContainer::TrackStateProxy& /*trackState*/) const {
    return BranchStopperResult::Continue;
  }
};

std::string eventFile(const std::string& dir, const std::string& stem,
                      std::size_t eventNr) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "event%09zu", eventNr);
  return dir + "/" + buf + stem;
}

}  // namespace

TruthRolloutAlgorithm::TruthRolloutAlgorithm(Config cfg,
                                             Acts::Logging::Level lvl)
    : IAlgorithm("TruthRolloutAlgorithm", lvl), m_cfg(std::move(cfg)) {
  if (m_cfg.inputMeasurements.empty()) {
    throw std::invalid_argument("Missing input measurements");
  }
  if (m_cfg.outputTracks.empty()) {
    throw std::invalid_argument("Missing output tracks key");
  }
  if (m_cfg.findTracks == nullptr) {
    throw std::invalid_argument("Missing findTracks function");
  }
  m_inputMeasurements.initialize(m_cfg.inputMeasurements);
  m_outputTracks.initialize(m_cfg.outputTracks);
}

std::vector<TruthRolloutAlgorithm::WorklistRow>
TruthRolloutAlgorithm::readWorklist(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("TruthRolloutAlgorithm: cannot open " + path);
  }
  std::vector<WorklistRow> rows;
  std::string line;
  std::getline(f, line);  // header, fixed column order (walker writes it)
  while (std::getline(f, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> v;
    while (std::getline(ss, tok, ',')) {
      v.push_back(tok);
    }
    if (v.size() < 17) {
      throw std::runtime_error("TruthRolloutAlgorithm: short row in " + path);
    }
    WorklistRow r;
    r.rolloutId = std::stoull(v[0]);
    r.seedId = std::stoull(v[1]);
    r.stepK = static_cast<std::uint32_t>(std::stoul(v[2]));
    r.geometryId = std::stoull(v[3]);
    for (int i = 0; i < 6; ++i) {
      r.par[i] = std::stod(v[4 + static_cast<std::size_t>(i)]);
      r.var[i] = std::stod(v[10 + static_cast<std::size_t>(i)]);
    }
    r.majorityPid = std::stoull(v[16]);
    rows.push_back(r);
  }
  return rows;
}

ProcessCode TruthRolloutAlgorithm::execute(const AlgorithmContext& ctx) const {
  const auto& measurements = m_inputMeasurements(ctx);

  const auto worklist = readWorklist(
      eventFile(m_cfg.worklistDir, "-rollout-worklist.csv", ctx.eventNumber));
  auto hitMap = cckf::TruthHitMap::load(
      eventFile(m_cfg.csvDir, "-measurement-simhit-map.csv", ctx.eventNumber),
      eventFile(m_cfg.csvDir, "-simhits.csv", ctx.eventNumber));
  ACTS_INFO("TruthRollout: " << worklist.size() << " rollout starts, "
                             << hitMap.size() << " mapped measurements");

  // ---- CKF plumbing (mirrors CckfTrackFindingAlgorithm::execute) ----
  PassThroughCalibrator pcalibrator;
  MeasurementCalibratorAdapter calibrator(pcalibrator,
                                          measurements.container());
  Acts::GainMatrixUpdater kfUpdater(false);
  IndexSourceLinkAccessor slAccessor;
  slAccessor.container = &measurements.orderedIndices();

  auto trackContainer = std::make_shared<Acts::VectorTrackContainer>();
  auto trackStateContainer = std::make_shared<Acts::VectorMultiTrajectory>();
  TrackContainer tracks(trackContainer, trackStateContainer);
  tracks.addColumn<std::uint64_t>("rolloutId");
  Acts::ProxyAccessor<std::uint64_t> rolloutIdAcc("rolloutId");

  cckf::TruthRolloutContext rolloutCtx;
  rolloutCtx.hitMap = &hitMap;
  TruthSelectorAdapter selector(&rolloutCtx);
  NeverStopBranchStopper neverStop;

  using TrackStateCreatorType =
      Acts::TrackStateCreator<IndexSourceLinkAccessor::Iterator,
                              TrackContainer>;
  TrackStateCreatorType trackStateCreator;
  trackStateCreator.sourceLinkAccessor
      .template connect<&IndexSourceLinkAccessor::range>(&slAccessor);
  trackStateCreator.calibrator
      .template connect<&MeasurementCalibratorAdapter::calibrate>(&calibrator);
  trackStateCreator.measurementSelector
      .template connect<&TruthSelectorAdapter::select>(&selector);

  using Extensions = Acts::CombinatorialKalmanFilterExtensions<TrackContainer>;
  Extensions extensions;
  extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<
      typename TrackContainer::TrackStateContainerBackend>>(&kfUpdater);
  extensions.branchStopper.connect<&NeverStopBranchStopper::operator()>(
      &neverStop);
  extensions.createTrackStates
      .template connect<&TrackStateCreatorType::createTrackStates>(
          &trackStateCreator);

  Acts::PropagatorPlainOptions propOptions(ctx.geoContext, ctx.magFieldContext);
  propOptions.maxSteps = m_cfg.maxSteps;
  propOptions.direction = Acts::Direction::Forward();

  CckfTrackFindingAlgorithm::TrackFinderOptions options(
      ctx.geoContext, ctx.magFieldContext, ctx.calibContext, extensions,
      propOptions);

  // ---- Hit-sequence output ----
  const std::string outPath =
      eventFile(m_cfg.outputDir, "-rollout-hits.csv", ctx.eventNumber);
  std::ofstream out(outPath);
  out << "rollout_id,step,geometry_id,meas_id\n";

  // ---- Rollout loop ----
  std::size_t nDone = 0;
  for (const auto& row : worklist) {
    if (m_cfg.maxRollouts > 0 && nDone >= m_cfg.maxRollouts) {
      break;
    }
    const Acts::Surface* surface = m_cfg.trackingGeometry->findSurface(
        Acts::GeometryIdentifier(row.geometryId));
    if (surface == nullptr) {
      ++m_nFailed;
      continue;
    }

    Acts::BoundVector par;
    Acts::BoundSquareMatrix cov = Acts::BoundSquareMatrix::Zero();
    for (int i = 0; i < 6; ++i) {
      par[i] = row.par[i];
      // Guard degenerate logged variances (e.g. time unmeasured).
      cov(i, i) = row.var[i] > 0.0 ? row.var[i] : 1e-4;
    }
    Acts::BoundTrackParameters start(surface->getSharedPtr(), par, cov,
                                     Acts::ParticleHypothesis::pion());

    rolloutCtx.majorityPid = row.majorityPid;

    auto rootBranch = tracks.makeTrack();
    auto result = (*m_cfg.findTracks)(start, options, tracks, rootBranch);
    ++nDone;
    ++m_nRollouts;
    if (!result.ok()) {
      ++m_nFailed;
      continue;
    }

    for (const auto& track : result.value()) {
      rolloutIdAcc(track) = row.rolloutId;
      std::size_t step = 0;
      for (const auto& ts : track.trackStatesReversed()) {
        if (!ts.hasReferenceSurface()) {
          continue;
        }
        const auto geoId = ts.referenceSurface().geometryId().value();
        long long measId = -1;
        if (ts.typeFlags().isMeasurement() &&
            ts.hasUncalibratedSourceLink()) {
          measId = static_cast<long long>(
              ts.getUncalibratedSourceLink().get<IndexSourceLink>().index());
        } else {
          ++m_nHoles;
        }
        out << row.rolloutId << "," << step << "," << geoId << "," << measId
            << "\n";
        ++step;
        ++m_nSteps;
      }
    }
  }

  ConstTrackContainer constTracks{
      std::make_shared<Acts::ConstVectorTrackContainer>(
          std::move(*trackContainer)),
      std::make_shared<Acts::ConstVectorMultiTrajectory>(
          std::move(*trackStateContainer))};
  m_outputTracks(ctx, std::move(constTracks));

  ACTS_INFO("TruthRollout: " << nDone << " rollouts, " << m_nSteps
                             << " steps total, " << m_nFailed << " failed");
  return ProcessCode::SUCCESS;
}

ProcessCode TruthRolloutAlgorithm::finalize() {
  ACTS_INFO("TruthRollout totals: rollouts=" << m_nRollouts
            << " steps=" << m_nSteps << " holes=" << m_nHoles
            << " failed=" << m_nFailed);
  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
