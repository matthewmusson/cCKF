// cCKF/acts_patches/cckf/WeightBlob.hpp
#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cckf {

struct WeightBlob {
  uint32_t n_features;
  uint32_t n_hidden;
  uint32_t n_layers;

  std::vector<float> standardization_mean;
  std::vector<float> standardization_std;
  float platt_a0, platt_a1, platt_b0, platt_b1;

  // weights[i] has shape (out_dim, in_dim), row-major
  // biases[i] has shape (out_dim,)
  std::vector<std::vector<float>> weights;
  std::vector<std::vector<float>> biases;

  static WeightBlob load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      throw std::runtime_error("Cannot open weight blob: " + path);
    }

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "CCKF") {
      throw std::runtime_error("Bad magic in weight blob: " + path);
    }

    auto read_u32 = [&]() -> uint32_t {
      uint32_t v;
      f.read(reinterpret_cast<char*>(&v), 4);
      return v;
    };
    auto read_f32 = [&]() -> float {
      float v;
      f.read(reinterpret_cast<char*>(&v), 4);
      return v;
    };
    auto read_vec = [&](size_t n) -> std::vector<float> {
      std::vector<float> v(n);
      f.read(reinterpret_cast<char*>(v.data()),
             static_cast<std::streamsize>(n * 4));
      return v;
    };

    WeightBlob blob;
    uint32_t version = read_u32();
    if (version != 1) {
      throw std::runtime_error("Unsupported blob version: " +
                               std::to_string(version));
    }

    blob.n_features = read_u32();
    blob.n_hidden = read_u32();
    blob.n_layers = read_u32();

    blob.standardization_mean = read_vec(blob.n_features);
    blob.standardization_std = read_vec(blob.n_features);

    blob.platt_a0 = read_f32();
    blob.platt_a1 = read_f32();
    blob.platt_b0 = read_f32();
    blob.platt_b1 = read_f32();

    blob.weights.resize(blob.n_layers + 1);
    blob.biases.resize(blob.n_layers + 1);

    uint32_t in_dim = blob.n_features;
    for (uint32_t i = 0; i < blob.n_layers; ++i) {
      uint32_t out_dim = blob.n_hidden;
      blob.weights[i] = read_vec(out_dim * in_dim);
      blob.biases[i] = read_vec(out_dim);
      in_dim = out_dim;
    }
    // final layer: n_hidden -> 1
    blob.weights[blob.n_layers] = read_vec(1 * blob.n_hidden);
    blob.biases[blob.n_layers] = read_vec(1);

    if (!f) {
      throw std::runtime_error("Truncated weight blob: " + path);
    }

    return blob;
  }
};

}  // namespace cckf
