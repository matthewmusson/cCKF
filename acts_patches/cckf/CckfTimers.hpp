// cCKF/acts_patches/cckf/CckfTimers.hpp
#pragma once
#include <array>
#include <chrono>
#include <cstdint>

namespace cckf {

/// Accumulates wall-clock time and call counts for one instrumented category.
/// All timing is std::chrono::steady_clock; aggregation is per-event (the
/// caller is expected to reset() at the start of each event).
struct TimerCounter {
  int64_t total_ns = 0;
  int64_t n_calls = 0;

  void record(std::chrono::steady_clock::time_point start,
              std::chrono::steady_clock::time_point end) {
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start)
                    .count();
    ++n_calls;
  }

  double mean_ns() const {
    return n_calls > 0 ? static_cast<double>(total_ns) / n_calls : 0.0;
  }
};

/// Aggregate chi2 and gate-score diagnostics accumulated per event.
struct GateDiagnostics {
  int64_t n_accepted = 0;
  int64_t n_rejected = 0;
  int64_t n_outlier_fallback = 0;
  double sum_chi2_accepted = 0;
  double sum_chi2_rejected = 0;
  float max_chi2_accepted = 0;
  double sum_score_accepted = 0;
  double sum_score_rejected = 0;
  int64_t n_nan_features = 0;
};

/// Per-step chi2/score aggregation across all seeds. Tells us whether
/// compound error builds over track depth: if step 0 has mean chi2 ~ 1
/// but step 5 has mean chi2 ~ 100, that's DAgger compound error.
struct PerStepStats {
  static constexpr int MAX_STEPS = 30;
  int64_t n_accepted[MAX_STEPS] = {};
  double sum_chi2_accepted[MAX_STEPS] = {};
  int64_t n_total_candidates[MAX_STEPS] = {};
  double sum_chi2_all[MAX_STEPS] = {};

  void recordAccepted(uint32_t step, float chi2) {
    if (step < MAX_STEPS) {
      ++n_accepted[step];
      sum_chi2_accepted[step] += chi2;
    }
  }
  void recordAll(uint32_t step, float chi2) {
    if (step < MAX_STEPS) {
      ++n_total_candidates[step];
      sum_chi2_all[step] += chi2;
    }
  }
};

/// Per-track feature trace: for the first N seeds, logs a representative
/// sample of ALL candidates' raw features at every step. The sample
/// preserves the distribution shape: accepted (best chi2 among passed),
/// min-chi2, max-chi2, and evenly spaced chi2 quantiles.
struct TrackTrace {
  static constexpr int N_FEAT = 26;
  static constexpr int MAX_TRACKS = 100;
  static constexpr int MAX_STEPS = 15;
  static constexpr int MAX_CANDS = 30;

  struct CandEntry {
    float features[N_FEAT] = {};
    float score = 0;
    float chi2 = 0;
    bool accepted = false;
  };

  struct StepEntry {
    uint32_t step_k = 0;
    uint32_t seed_idx = 0;
    int n_cands_stored = 0;
    int n_cands_total = 0;
    int n_accepted = 0;
    CandEntry cands[MAX_CANDS] = {};
  };

  StepEntry entries[MAX_TRACKS][MAX_STEPS] = {};
  int step_counts[MAX_TRACKS] = {};
  int n_tracks = 0;

  void addStep(uint32_t seed_idx, uint32_t step_k,
               const std::array<float, N_FEAT>* all_feats,
               const float* all_scores, const float* all_chi2s,
               std::size_t n_cands, std::size_t n_passed) {
    // Find or create track slot
    int t = -1;
    for (int i = 0; i < n_tracks; ++i) {
      if (entries[i][0].seed_idx == seed_idx) { t = i; break; }
    }
    if (t < 0) {
      if (n_tracks >= MAX_TRACKS) return;
      t = n_tracks++;
    }
    int s = step_counts[t];
    if (s >= MAX_STEPS) return;
    auto& se = entries[t][s];
    se.step_k = step_k;
    se.seed_idx = seed_idx;
    se.n_cands_total = static_cast<int>(n_cands);
    se.n_accepted = static_cast<int>(n_passed);

    if (n_cands == 0) {
      se.n_cands_stored = 0;
      step_counts[t] = s + 1;
      return;
    }

    // Build sorted-by-chi2 index array (small N, insertion sort is fine)
    int sorted_idx[512];
    int N = static_cast<int>(n_cands);
    if (N > 512) N = 512;
    for (int i = 0; i < N; ++i) sorted_idx[i] = i;
    for (int i = 1; i < N; ++i) {
      int key = sorted_idx[i];
      float keyVal = all_chi2s[key];
      int j = i - 1;
      while (j >= 0 && all_chi2s[sorted_idx[j]] > keyVal) {
        sorted_idx[j + 1] = sorted_idx[j];
        --j;
      }
      sorted_idx[j + 1] = key;
    }

    // Pick representative indices into a selection buffer
    int sel[MAX_CANDS];
    int n_sel = 0;

    // Slot 0: best accepted (lowest chi2 among passed candidates)
    int best_acc = -1;
    for (int i = 0; i < N; ++i) {
      int idx = sorted_idx[i];
      if (static_cast<std::size_t>(idx) < n_passed) {
        best_acc = idx;
        break;
      }
    }
    sel[n_sel++] = (best_acc >= 0) ? best_acc : sorted_idx[0];

    if (N <= MAX_CANDS) {
      // Store all candidates — no subsampling needed
      for (int i = 0; i < N; ++i) {
        if (sorted_idx[i] == sel[0]) continue;
        sel[n_sel++] = sorted_idx[i];
      }
    } else {
      // Slot 1: min-chi2 overall
      if (sorted_idx[0] != sel[0]) sel[n_sel++] = sorted_idx[0];
      // Slot 2: max-chi2 overall
      if (sorted_idx[N - 1] != sel[0]) sel[n_sel++] = sorted_idx[N - 1];
      // Remaining slots: evenly spaced quantiles from sorted list
      int remaining = MAX_CANDS - n_sel;
      for (int j = 0; j < remaining && n_sel < MAX_CANDS; ++j) {
        int pick = sorted_idx[(j * (N - 1)) / (remaining - 1)];
        // Skip duplicates
        bool dup = false;
        for (int k = 0; k < n_sel; ++k) {
          if (sel[k] == pick) { dup = true; break; }
        }
        if (!dup) sel[n_sel++] = pick;
      }
    }

    se.n_cands_stored = n_sel;
    for (int c = 0; c < n_sel; ++c) {
      auto& ce = se.cands[c];
      int idx = sel[c];
      for (int j = 0; j < N_FEAT; ++j) ce.features[j] = all_feats[idx][j];
      ce.score = all_scores[idx];
      ce.chi2 = all_chi2s[idx];
      ce.accepted = (static_cast<std::size_t>(idx) < n_passed);
    }
    step_counts[t] = s + 1;
  }
};

/// Per-event timing accumulator for the cCKF instrumentation. One instance is
/// expected per worker thread (mirrors MlpInference's per-thread ownership
/// requirement), reset() at the start of every event.
struct CckfTimers {
  /// Time spent inside the gate MLP forward + calibrate call.
  TimerCounter gate_inference;
  /// Time spent building the 26-dim gate feature vector (cluster lookups,
  /// Cholesky, normalized cluster features) ahead of each gate_inference call.
  TimerCounter gate_feature_build;
  /// Time spent inside the value function MLP forward + calibrate call.
  /// Populated starting in Task 3; present here so a single CckfTimers
  /// instance can be threaded through both the gate and value integrations.
  TimerCounter value_inference;
  /// Total wall-clock time inside CckfMeasurementSelector::select(), i.e. the
  /// full replacement for MeasurementSelector::select() including chi2,
  /// feature building, and gate inference for every candidate on a surface.
  TimerCounter measurement_selection;

  GateDiagnostics gate_diag;
  PerStepStats per_step;
  TrackTrace track_trace;

  void reset() {
    gate_inference = {};
    gate_feature_build = {};
    value_inference = {};
    measurement_selection = {};
    gate_diag = {};
    per_step = {};
    track_trace = {};
  }
};

}  // namespace cckf
