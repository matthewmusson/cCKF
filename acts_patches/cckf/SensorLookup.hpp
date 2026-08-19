// cCKF/acts_patches/cckf/SensorLookup.hpp
#pragma once

#include "CckfFeatures.hpp"

#include <fstream>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>  // ACTS already links nlohmann_json

namespace cckf {

class SensorLookup {
 public:
  SensorLookup() = default;

  explicit SensorLookup(const std::string& digiConfigPath) {
    std::ifstream f(digiConfigPath);
    if (!f) {
      throw std::runtime_error("Cannot open digi config: " + digiConfigPath);
    }
    nlohmann::json cfg = nlohmann::json::parse(f);

    for (const auto& entry : cfg["entries"]) {
      uint32_t volume = entry.value("volume", 0u);
      auto geo = entry.value("value", nlohmann::json{})
                     .value("geometric", nlohmann::json{});
      auto bins = geo.value("segmentation", nlohmann::json{})
                      .value("binningdata", nlohmann::json::array());

      SensorProps props;
      if (bins.size() > 0) {
        auto& b0 = bins[0];
        int nbins = b0.value("bins", 0);
        if (nbins > 0) {
          props.pitch_u = static_cast<float>(
              (b0.value("max", 0.0) - b0.value("min", 0.0)) / nbins);
        }
      }
      if (bins.size() > 1) {
        auto& b1 = bins[1];
        int nbins = b1.value("bins", 0);
        if (nbins > 0) {
          props.pitch_v = static_cast<float>(
              (b1.value("max", 0.0) - b1.value("min", 0.0)) / nbins);
        }
      }
      props.thickness = static_cast<float>(
          geo.value("thickness", 0.0));

      m_lookup[volume] = props;
    }
  }

  SensorProps get(uint32_t volumeId) const {
    auto it = m_lookup.find(volumeId);
    return (it != m_lookup.end()) ? it->second : SensorProps{};
  }

 private:
  std::unordered_map<uint32_t, SensorProps> m_lookup;
};

}  // namespace cckf
