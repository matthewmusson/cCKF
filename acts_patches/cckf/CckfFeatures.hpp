// cCKF/acts_patches/cckf/CckfFeatures.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cckf {

/// Branch history context, set by the CKF actor before each select() call.
///
/// This is the injection point the design notes call for: the
/// MeasurementSelector::select() signature does not receive the track proxy,
/// so branch-local state (hit/hole counts, kinematics, path-in-X0) has to be
/// pushed into the selector from outside immediately before each surface's
/// selection, the same way BranchStopper receives its state via
/// ProxyAccessor.
struct BranchContext {
  uint32_t n_hits = 0;
  uint32_t n_holes = 0;
  uint32_t n_seq_holes = 0;
  float eta = 0;
  float qop = 0;
  uint32_t step_k = 0;
  float pathInX0 = 0;
};

/// Sensor properties looked up from geometry for the current surface.
struct SensorProps {
  float pitch_u = 0;
  float pitch_v = 0;
  float thickness = 0;
};

/// Build the 26-dim gate feature vector.
///
/// Feature order matches cckf/features.py GATE_FEATURES exactly:
///   [residual_l0, residual_l1, chol_S_00, chol_S_10, chol_S_11, chi2_inc,
///    clus_s_u, clus_s_v, clus_q_tot, clus_sigma_uu, clus_sigma_uv,
///    clus_sigma_vv, kappa_u, kappa_v, q_tilde,
///    n_window, eta, state_qop, step_k, pathInX0_interval,
///    pitch_u, pitch_v, thickness,
///    n_hits, n_holes, n_seq_holes]
///
/// @param out 26-element float array to fill
/// @param res0 residual in local 0
/// @param res1 residual in local 1
/// @param S00 innovation covariance (0,0)
/// @param S01 innovation covariance (0,1)
/// @param S11 innovation covariance (1,1)
/// @param chi2 incremental chi2
/// @param clus_s_u cluster size in u (channels)
/// @param clus_s_v cluster size in v (channels)
/// @param clus_q_tot total cluster charge
/// @param clus_sigma_uu charge-weighted second moment uu
/// @param clus_sigma_uv charge-weighted second moment uv
/// @param clus_sigma_vv charge-weighted second moment vv
/// @param alpha_u incidence angle in u
/// @param alpha_v incidence angle in v
/// @param n_window number of candidates on this surface
/// @param ctx branch history context
/// @param sensor sensor geometry properties
inline void buildGateFeatures(float* out, float res0, float res1, float S00,
                              float S01, float S11, float chi2,
                              float clus_s_u, float clus_s_v,
                              float clus_q_tot, float clus_sigma_uu,
                              float clus_sigma_uv, float clus_sigma_vv,
                              float alpha_u, float alpha_v, float n_window,
                              const BranchContext& ctx,
                              const SensorProps& sensor) {
  // Cholesky of 2x2 S. Mirrors cckf/features.py::cholesky_S: L00 = sqrt(S00),
  // L10 = S01 / L00, L11 = sqrt(S11 - L10^2), with variances floored and the
  // Schur complement clamped at zero so a degenerate S yields finite output.
  float l00 = std::sqrt(std::max(S00, 1e-30f));
  float l10 = (l00 > 1e-15f) ? S01 / l00 : 0.0f;
  float diag = S11 - l10 * l10;
  float l11 = std::sqrt(std::max(diag, 1e-30f));

  // Normalized cluster features (kappa_u, kappa_v, q_tilde). Mirrors
  // cckf/features.py::kappa_u / kappa_v / q_tilde exactly.
  float tan_alpha_u = std::tan(alpha_u);
  float tan_alpha_v = std::tan(alpha_v);
  float expected_s_u =
      (sensor.pitch_u > 1e-10f)
          ? 1.0f + sensor.thickness * std::abs(tan_alpha_u) / sensor.pitch_u
          : 1.0f;
  float expected_s_v =
      (sensor.pitch_v > 1e-10f)
          ? 1.0f + sensor.thickness * std::abs(tan_alpha_v) / sensor.pitch_v
          : 1.0f;
  float kappa_u = (expected_s_u > 1e-10f) ? clus_s_u / expected_s_u : 0.0f;
  float kappa_v = (expected_s_v > 1e-10f) ? clus_s_v / expected_s_v : 0.0f;

  float expected_q = sensor.thickness *
                      std::sqrt(1.0f + tan_alpha_u * tan_alpha_u +
                               tan_alpha_v * tan_alpha_v);
  float q_tilde = (expected_q > 1e-10f) ? clus_q_tot / expected_q : 0.0f;

  // Fill in exact GATE_FEATURES order
  out[0] = res0;
  out[1] = res1;
  out[2] = l00;
  out[3] = l10;
  out[4] = l11;
  out[5] = chi2;
  out[6] = clus_s_u;
  out[7] = clus_s_v;
  out[8] = clus_q_tot;
  out[9] = clus_sigma_uu;
  out[10] = clus_sigma_uv;
  out[11] = clus_sigma_vv;
  out[12] = kappa_u;
  out[13] = kappa_v;
  out[14] = q_tilde;
  out[15] = n_window;
  out[16] = ctx.eta;
  out[17] = ctx.qop;
  out[18] = static_cast<float>(ctx.step_k);
  out[19] = ctx.pathInX0;
  out[20] = sensor.pitch_u;
  out[21] = sensor.pitch_v;
  out[22] = sensor.thickness;
  out[23] = static_cast<float>(ctx.n_hits);
  out[24] = static_cast<float>(ctx.n_holes);
  out[25] = static_cast<float>(ctx.n_seq_holes);
}

}  // namespace cckf
