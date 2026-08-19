// cCKF/acts_patches/cckf/MlpInference.hpp
#pragma once
#include "WeightBlob.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cckf {

class MlpInference {
 public:
  /// Takes ownership of the blob (by value; pass an rvalue or std::move an
  /// lvalue to avoid a copy). Storing by value rather than by pointer/ref
  /// makes it structurally impossible to dangle off a temporary, e.g.
  /// `MlpInference mlp(WeightBlob::load(path));` — the previous
  /// pointer-to-const-ref design bound to the temporary WeightBlob, which
  /// was destroyed at the end of that statement, leaving every subsequent
  /// forward() call as undefined behavior.
  explicit MlpInference(WeightBlob blob)
      : m_blob(std::move(blob)),
        m_buf_a(m_blob.n_hidden),
        m_buf_b(m_blob.n_hidden),
        m_input_buf(m_blob.n_features) {}

  /// Run forward pass on raw (un-standardized) input features.
  /// Returns raw logit (pre-sigmoid).
  float forward(const float* raw_input) const {
    const uint32_t n_feat = m_blob.n_features;
    const uint32_t n_hid = m_blob.n_hidden;
    const uint32_t n_layers = m_blob.n_layers;

    // Standardize input
    for (uint32_t j = 0; j < n_feat; ++j) {
      float s = m_blob.standardization_std[j];
      m_input_buf[j] =
          (s > 1e-30f)
              ? (raw_input[j] - m_blob.standardization_mean[j]) / s
              : 0.0f;
    }

    // First hidden layer: n_feat -> n_hid
    linear_silu(m_blob.weights[0].data(), m_blob.biases[0].data(),
                m_input_buf.data(), m_buf_a.data(), n_feat, n_hid);

    // Remaining hidden layers: n_hid -> n_hid
    float* src = m_buf_a.data();
    float* dst = m_buf_b.data();
    for (uint32_t i = 1; i < n_layers; ++i) {
      linear_silu(m_blob.weights[i].data(), m_blob.biases[i].data(), src,
                  dst, n_hid, n_hid);
      std::swap(src, dst);
    }

    // Final layer: n_hid -> 1 (no activation)
    float logit = m_blob.biases[n_layers][0];
    const float* w_final = m_blob.weights[n_layers].data();
    for (uint32_t j = 0; j < n_hid; ++j) {
      logit += w_final[j] * src[j];
    }

    return logit;
  }

  /// Apply occupancy-conditional Platt calibration to raw logit.
  /// Returns calibrated probability in [0, 1].
  float calibrate(float logit, float log_n_window) const {
    float a = m_blob.platt_a0 + m_blob.platt_a1 * log_n_window;
    float b = m_blob.platt_b0 + m_blob.platt_b1 * log_n_window;
    float z = a * logit + b;
    return 1.0f / (1.0f + std::exp(-z));
  }

  /// Forward pass + calibration. Returns calibrated probability.
  float predict(const float* raw_input, float log_n_window) const {
    return calibrate(forward(raw_input), log_n_window);
  }

 private:
  static void linear_silu(const float* W, const float* b, const float* x,
                          float* y, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t i = 0; i < out_dim; ++i) {
      float acc = b[i];
      const float* row = W + i * in_dim;
      for (uint32_t j = 0; j < in_dim; ++j) {
        acc += row[j] * x[j];
      }
      // SiLU: z * sigmoid(z)
      y[i] = acc / (1.0f + std::exp(-acc));
    }
  }

  WeightBlob m_blob;
  // Scratch buffers reused across forward() calls to avoid per-call
  // allocation. NOT thread-safe: two threads calling forward()/predict() on
  // the *same* MlpInference instance concurrently will race on these
  // buffers. ACTS CKF can be multi-threaded, so callers must give each
  // thread its own MlpInference instance (cheap to construct — copy or
  // reload the WeightBlob per thread).
  mutable std::vector<float> m_buf_a;
  mutable std::vector<float> m_buf_b;
  mutable std::vector<float> m_input_buf;
};

}  // namespace cckf
