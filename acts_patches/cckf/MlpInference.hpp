// cCKF/acts_patches/cckf/MlpInference.hpp
#pragma once
#include "WeightBlob.hpp"

#include <Eigen/Core>

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
  ///
  /// KERNEL NOTE (2026-08-25). The original hand-rolled layer loop ran one
  /// scalar multiply-add per iteration: the accumulator is a float
  /// *reduction*, IEEE addition is not associative, and ACTS builds without
  /// -ffast-math, so the compiler was forbidden from reordering it into
  /// SIMD partial sums. Measured cost: 24.2 us/call for ~37K MACs
  /// (~1.5 GFLOP/s, an order of magnitude under one vectorized core).
  /// Eigen's own product kernels carry explicit SIMD reductions and need no
  /// fast-math, so every layer is now an Eigen product. Consequence for
  /// tests: vectorized summation legitimately reorders additions, so
  /// fixture comparisons (gate_test*.bin) must use a small tolerance
  /// (1e-5 relative) instead of bit equality.
  explicit MlpInference(WeightBlob blob)
      : m_blob(std::move(blob)),
        m_buf_a(m_blob.n_hidden),
        m_buf_b(m_blob.n_hidden),
        m_input_buf(m_blob.n_features) {}

  using MatX = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>;
  using MapMat = Eigen::Map<const MatX>;
  using MapVec = Eigen::Map<const Eigen::VectorXf>;

  /// Run forward pass on raw (un-standardized) input features.
  /// Returns raw logit (pre-sigmoid).
  float forward(const float* raw_input) const {
    const uint32_t n_feat = m_blob.n_features;
    const uint32_t n_hid = m_blob.n_hidden;
    const uint32_t n_layers = m_blob.n_layers;

    standardizeRow(raw_input, m_input_buf.data());

    // First hidden layer: n_feat -> n_hid.  W is stored row-major
    // (out x in), exactly Eigen's RowMajor layout, so the blob is mapped in
    // place — no copy, no transpose.
    Eigen::Map<Eigen::VectorXf> a(m_buf_a.data(), n_hid);
    Eigen::Map<Eigen::VectorXf> b(m_buf_b.data(), n_hid);
    a.noalias() = MapMat(m_blob.weights[0].data(), n_hid, n_feat) *
                      MapVec(m_input_buf.data(), n_feat) +
                  MapVec(m_blob.biases[0].data(), n_hid);
    silu(a);

    // Remaining hidden layers: n_hid -> n_hid
    bool srcIsA = true;
    for (uint32_t i = 1; i < n_layers; ++i) {
      auto& src = srcIsA ? a : b;
      auto& dst = srcIsA ? b : a;
      dst.noalias() = MapMat(m_blob.weights[i].data(), n_hid, n_hid) * src +
                      MapVec(m_blob.biases[i].data(), n_hid);
      silu(dst);
      srcIsA = !srcIsA;
    }

    // Final layer: n_hid -> 1 (no activation)
    const auto& last = srcIsA ? a : b;
    return m_blob.biases[n_layers][0] +
           MapVec(m_blob.weights[n_layers].data(), n_hid).dot(last);
  }

  /// Batched forward pass: `n` feature rows, contiguous row-major
  /// (`raw_inputs[r * n_features + j]`), logits written to `logits[0..n)`.
  ///
  /// One matrix-matrix product per layer instead of n matrix-vector
  /// products: the ~37K weights stream from memory once per SURFACE rather
  /// than once per candidate, and matrix-matrix products have the
  /// arithmetic intensity the vector units want. Intended call site:
  /// CckfMeasurementSelector::select(), scoring every candidate that
  /// survives the box prefilter (~7/surface on event 4) in one call.
  ///
  /// Scratch matrices grow to the largest n seen and are then reused, so
  /// steady-state calls allocate nothing. Same thread contract as
  /// forward(): one instance per thread.
  void forwardBatch(const float* raw_inputs, uint32_t n,
                    float* logits) const {
    if (n == 0) {
      return;
    }
    const uint32_t n_feat = m_blob.n_features;
    const uint32_t n_hid = m_blob.n_hidden;
    const uint32_t n_layers = m_blob.n_layers;

    if (m_batch_x.rows() < static_cast<Eigen::Index>(n)) {
      m_batch_x.resize(n, n_feat);
      m_batch_a.resize(n, n_hid);
      m_batch_b.resize(n, n_hid);
    }

    for (uint32_t r = 0; r < n; ++r) {
      standardizeRow(raw_inputs + static_cast<std::size_t>(r) * n_feat,
                     m_batch_x.row(r).data());
    }

    auto X = m_batch_x.topRows(n);
    auto A = m_batch_a.topRows(n);
    auto B = m_batch_b.topRows(n);

    // X (n x feat) * W0^T (feat x hid) -> n x hid, bias broadcast per row.
    A.noalias() =
        X * MapMat(m_blob.weights[0].data(), n_hid, n_feat).transpose();
    A.rowwise() += MapVec(m_blob.biases[0].data(), n_hid).transpose();
    silu(A);

    bool srcIsA = true;
    for (uint32_t i = 1; i < n_layers; ++i) {
      auto& src = srcIsA ? A : B;
      auto& dst = srcIsA ? B : A;
      dst.noalias() =
          src * MapMat(m_blob.weights[i].data(), n_hid, n_hid).transpose();
      dst.rowwise() += MapVec(m_blob.biases[i].data(), n_hid).transpose();
      silu(dst);
      srcIsA = !srcIsA;
    }

    const auto& last = srcIsA ? A : B;
    Eigen::Map<Eigen::VectorXf>(logits, n).noalias() =
        last * MapVec(m_blob.weights[n_layers].data(), n_hid);
    Eigen::Map<Eigen::VectorXf>(logits, n).array() +=
        m_blob.biases[n_layers][0];
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

  const WeightBlob& blob() const { return m_blob; }

 private:
  /// Standardize one raw feature row into `out` (both length n_features).
  void standardizeRow(const float* raw, float* out) const {
    const uint32_t n_feat = m_blob.n_features;
    for (uint32_t j = 0; j < n_feat; ++j) {
      float s = m_blob.standardization_std[j];
      out[j] = (s > 1e-30f)
                   ? (raw[j] - m_blob.standardization_mean[j]) / s
                   : 0.0f;
    }
  }

  /// SiLU z * sigmoid(z), elementwise on any Eigen dense expression.
  template <typename Derived>
  static void silu(Eigen::DenseBase<Derived>& z) {
    z.derived().array() =
        z.derived().array() / (1.0f + (-z.derived().array()).exp());
  }

  WeightBlob m_blob;
  // Scratch buffers reused across forward()/forwardBatch() calls to avoid
  // per-call allocation. NOT thread-safe: two threads calling into the
  // *same* MlpInference instance concurrently will race on these buffers.
  // ACTS CKF can be multi-threaded, so callers must give each thread its
  // own MlpInference instance (cheap to construct — copy or reload the
  // WeightBlob per thread).
  mutable std::vector<float> m_buf_a;
  mutable std::vector<float> m_buf_b;
  mutable std::vector<float> m_input_buf;
  mutable MatX m_batch_x;
  mutable MatX m_batch_a;
  mutable MatX m_batch_b;
};

}  // namespace cckf
