// cCKF/acts_patches/cckf/TruthRolloutSelector.hpp
#pragma once

#include "Acts/EventData/SourceLink.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cckf {

/// Measurement -> contributing-particle lookup for truth-greedy rollouts.
///
/// Loaded once per event from the Stage 1 CSVs (same files expansion.py
/// reads):
///   - event{N:09d}-measurement-simhit-map.csv : measurement_id, hit_id
///   - event{N:09d}-simhits.csv               : per-hit particle barcode
///     fields (particle_id_pv, _sv, _part, _gen, _subpart)
///
/// The packed barcode uses expansion.py's encode_particle_id layout:
///   pid = (pv << 48) | (sv << 32) | (part << 16) | (gen << 8) | sub
/// so the worklist's majority PID (produced by the Python walker from the
/// expanded Parquet) compares directly against these values with no
/// re-encoding on either side.
class TruthHitMap {
 public:
  /// Returns true if measurement `measIndex` has `pid` among its
  /// contributing particles. Merged clusters carry several contributors;
  /// membership (not the mode) is the spec's correctness test, matching
  /// selected_correctness in patch_is_selected.py.
  bool contains(std::uint64_t measIndex, std::uint64_t pid) const {
    auto it = m_contribs.find(measIndex);
    if (it == m_contribs.end()) {
      return false;
    }
    for (std::uint64_t p : it->second) {
      if (p == pid) {
        return true;
      }
    }
    return false;
  }

  std::size_t size() const { return m_contribs.size(); }

  /// Load from the two CSVs. Header layout is taken from the files
  /// themselves (column-name lookup, not fixed positions), because the
  /// simhits CSV column order differs between ACTS versions.
  static TruthHitMap load(const std::string& measSimhitMapCsv,
                          const std::string& simhitsCsv) {
    TruthHitMap out;

    // --- simhits: row index (file order) IS the hit_id ------------------
    std::vector<std::uint64_t> hitPid;
    {
      std::ifstream f(simhitsCsv);
      if (!f) {
        throw std::runtime_error("TruthHitMap: cannot open " + simhitsCsv);
      }
      std::string line;
      std::getline(f, line);
      auto cols = splitCsv(line);
      int iPv = indexOf(cols, "particle_id_pv");
      int iSv = indexOf(cols, "particle_id_sv");
      int iPart = indexOf(cols, "particle_id_part");
      int iGen = indexOf(cols, "particle_id_gen");
      int iSub = indexOf(cols, "particle_id_subpart");
      if (iPart < 0) {
        throw std::runtime_error(
            "TruthHitMap: no particle_id_part column in " + simhitsCsv);
      }
      while (std::getline(f, line)) {
        if (line.empty()) {
          continue;
        }
        auto v = splitCsv(line);
        auto field = [&](int i) -> std::uint64_t {
          return (i >= 0 && i < static_cast<int>(v.size()))
                     ? std::stoull(v[static_cast<std::size_t>(i)])
                     : 0ull;
        };
        std::uint64_t pid = (field(iPv) << 48) | (field(iSv) << 32) |
                            (field(iPart) << 16) | (field(iGen) << 8) |
                            field(iSub);
        hitPid.push_back(pid);
      }
    }

    // --- map: measurement_id -> [hit_id...] -> [pid...] -----------------
    {
      std::ifstream f(measSimhitMapCsv);
      if (!f) {
        throw std::runtime_error("TruthHitMap: cannot open " +
                                 measSimhitMapCsv);
      }
      std::string line;
      std::getline(f, line);
      auto cols = splitCsv(line);
      int iMeas = indexOf(cols, "measurement_id");
      int iHit = indexOf(cols, "hit_id");
      if (iMeas < 0 || iHit < 0) {
        throw std::runtime_error(
            "TruthHitMap: need measurement_id and hit_id columns in " +
            measSimhitMapCsv);
      }
      while (std::getline(f, line)) {
        if (line.empty()) {
          continue;
        }
        auto v = splitCsv(line);
        std::uint64_t meas =
            std::stoull(v[static_cast<std::size_t>(iMeas)]);
        std::uint64_t hit = std::stoull(v[static_cast<std::size_t>(iHit)]);
        if (hit < hitPid.size()) {
          out.m_contribs[meas].push_back(hitPid[hit]);
        }
      }
    }
    return out;
  }

 private:
  static std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      // strip spaces the way expansion.py strips column names
      std::size_t b = tok.find_first_not_of(" \t\r");
      std::size_t e = tok.find_last_not_of(" \t\r");
      out.push_back(b == std::string::npos
                        ? std::string()
                        : tok.substr(b, e - b + 1));
    }
    return out;
  }

  static int indexOf(const std::vector<std::string>& cols,
                     const std::string& name) {
    for (std::size_t i = 0; i < cols.size(); ++i) {
      if (cols[i] == name) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> m_contribs;
};

/// Truth-greedy measurement selection for tier-3 rollouts.
///
/// Header-only policy object used by TruthRolloutAlgorithm's selector
/// adapter (mirroring how CckfMeasurementSelectorAdapter wraps
/// CckfMeasurementSelector). Given the candidate track states on a surface,
/// keep exactly the candidates whose measurement carries the rollout's
/// majority particle among its contributors; the adapter then applies the
/// pi-dagger tie-break (lowest chi2) if more than one survives and marks a
/// hole when none do.
///
/// Deliberately NO chi2 window and NO MLP on IDENTITY selection: pi-dagger
/// picks WHICH hit by identity, never by chi2. The covariance therefore
/// cannot change which hit is chosen on a reached surface -- the property
/// that makes diagonal-seeded offline rollouts viable (spec:
/// docs/superpowers/specs/2026-08-25-tier3-rollout-design.md).
///
/// `windowNsigma` (below) gates only ACCEPTANCE of the already-identified
/// true hit, so identity selection stays covariance-independent; the window
/// merely decides whether the reachable true hit counts as reached.
struct TruthRolloutContext {
  /// Packed barcode of the branch's majority particle for the CURRENT
  /// rollout. Set by the algorithm before each findTracks call. Plain
  /// member (not atomic): rollouts run sequentially within one algorithm
  /// instance; each worker thread owns its own instance, same threading
  /// contract as CckfMeasurementSelector.
  std::uint64_t majorityPid = 0;

  /// Chi2 window for pi-dagger acceptance: the true hit is taken iff
  /// chi2 < windowNsigma^2. <= 0 disables the window (unbounded pi-dagger,
  /// the pre-2026-09 behavior). Matches the deployed gate pre-filter
  /// semantics (CckfMeasurementSelector nSigma: chi2 < nsigma^2), so
  /// V(n) conditions on exactly what the deployed chain can reach.
  double windowNsigma = 0.0;

  const TruthHitMap* hitMap = nullptr;

  bool isTruthMeasurement(std::uint64_t measIndex) const {
    return hitMap != nullptr && hitMap->contains(measIndex, majorityPid);
  }
};

}  // namespace cckf
