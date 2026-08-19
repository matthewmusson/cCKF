// cCKF/acts_patches/cckf/CckfTimers.hpp
#pragma once
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

  void reset() {
    gate_inference = {};
    gate_feature_build = {};
    value_inference = {};
    measurement_selection = {};
  }
};

}  // namespace cckf
