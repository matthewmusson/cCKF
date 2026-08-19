// cCKF/tests/test_mlp_inference.cpp
//
// Standalone test (no ACTS dependency) for cckf::WeightBlob / cckf::MlpInference.
//
// Loads the fixture blob produced by tests/test_export_weights.py
// (tests/fixtures/gate_test.bin + gate_test_input.bin + gate_test_expected.bin),
// runs MlpInference::forward on the fixed input, and checks the result matches
// the PyTorch reference logit to within tolerance.
//
// Build (from cCKF/):
//   clang++ -std=c++17 -O2 -Iacts_patches/cckf tests/test_mlp_inference.cpp \
//       -o /tmp/test_mlp_inference
// Run:
//   /tmp/test_mlp_inference [path/to/tests/fixtures]
//
// Regenerate fixtures with: python3 tests/test_export_weights.py

#include "MlpInference.hpp"
#include "WeightBlob.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_floats(const std::string& path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open fixture file: " + path);
  }
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()),
         static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) {
    throw std::runtime_error("Truncated fixture file: " + path);
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string fixtures_dir =
      (argc > 1) ? argv[1] : "tests/fixtures";

  const std::string blob_path = fixtures_dir + "/gate_test.bin";
  const std::string input_path = fixtures_dir + "/gate_test_input.bin";
  const std::string expected_path = fixtures_dir + "/gate_test_expected.bin";

  int failures = 0;

  // --- Test 1: header fields load correctly ---
  cckf::WeightBlob blob;
  try {
    blob = cckf::WeightBlob::load(blob_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: could not load blob %s: %s\n",
                 blob_path.c_str(), e.what());
    return 1;
  }

  if (blob.n_features != 26) {
    std::fprintf(stderr, "FAIL: expected n_features=26, got %u\n",
                 blob.n_features);
    ++failures;
  }
  if (blob.n_hidden != 128) {
    std::fprintf(stderr, "FAIL: expected n_hidden=128, got %u\n",
                 blob.n_hidden);
    ++failures;
  }
  if (blob.n_layers != 3) {
    std::fprintf(stderr, "FAIL: expected n_layers=3, got %u\n",
                 blob.n_layers);
    ++failures;
  }
  if (blob.weights.size() != blob.n_layers + 1 ||
      blob.biases.size() != blob.n_layers + 1) {
    std::fprintf(stderr, "FAIL: weights/biases vector count mismatch\n");
    ++failures;
  }
  std::printf("PASS: header fields (n_features=%u, n_hidden=%u, n_layers=%u)\n",
              blob.n_features, blob.n_hidden, blob.n_layers);

  // --- Test 2: forward() matches the PyTorch reference logit ---
  std::vector<float> input = read_floats(input_path, blob.n_features);
  std::vector<float> expected = read_floats(expected_path, 1);

  cckf::MlpInference mlp(blob);
  float logit = mlp.forward(input.data());

  const float tol = 1e-5f;  // per brief; measured diff is ~1e-8 in practice
  float diff = std::fabs(logit - expected[0]);
  if (diff > tol) {
    std::fprintf(stderr,
                 "FAIL: forward() = %.6f, expected %.6f (diff %.6f > tol %.6f)\n",
                 logit, expected[0], diff, tol);
    ++failures;
  } else {
    std::printf("PASS: forward() = %.6f, expected %.6f (diff %.6f <= tol %.6f)\n",
                logit, expected[0], diff, tol);
  }

  // --- Test 3: calibrate() sigmoid sanity check ---
  cckf::WeightBlob identity_blob = blob;
  identity_blob.platt_a0 = 1.0f;
  identity_blob.platt_a1 = 0.0f;
  identity_blob.platt_b0 = 0.0f;
  identity_blob.platt_b1 = 0.0f;
  cckf::MlpInference mlp_identity(identity_blob);
  float prob = mlp_identity.calibrate(0.0f, /*log_n_window=*/0.0f);
  if (std::fabs(prob - 0.5f) > 1e-6f) {
    std::fprintf(stderr, "FAIL: calibrate(0, identity Platt) = %.6f, expected 0.5\n",
                 prob);
    ++failures;
  } else {
    std::printf("PASS: calibrate(0, identity Platt) = %.6f\n", prob);
  }

  if (failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "%d TEST(S) FAILED\n", failures);
  return 1;
}
