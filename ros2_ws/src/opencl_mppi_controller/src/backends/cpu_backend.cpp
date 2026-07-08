// CPU reference backend — 3-step MPPI pipeline.
//
// Used for validation, debugging, and GPU-failure fallback.
// OpenMP parallelises the K trajectories.

#include "opencl_mppi_controller/mppi_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mppi_controller
{
namespace
{

inline float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

class CpuBackend : public IMppiBackend
{
public:
  GpuBackend type() const override { return GpuBackend::Cpu; }
  const char * name() const override { return "CPU"; }

  bool initialize(const DeviceParams & dp, uint64_t seed) override
  {
    dp_ = dp; K_ = dp.K; T_ = dp.T;

    nominal_.assign(T_ * 3, 0.0f);
    perturbed_.assign(static_cast<size_t>(K_) * T_ * 3, 0.0f);
    traj_x_.assign(static_cast<size_t>(K_) * T_, 0.0f);
    traj_y_.assign(static_cast<size_t>(K_) * T_, 0.0f);
    traj_yaw_.assign(static_cast<size_t>(K_) * T_, 0.0f);
    costs_.assign(K_, 0.0f);
    costmap_.assign(static_cast<size_t>(dp.size_x) * dp.size_y, 0);
    path_.resize(static_cast<size_t>(dp.path_len) * 2, 0.0f);

    rngs_.reserve(K_);
    for (int k = 0; k < K_; ++k) {
      uint64_t z = seed + static_cast<uint64_t>(k);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      z = z ^ (z >> 31);
      rngs_.emplace_back(static_cast<unsigned int>(z & 0xFFFFFFFFu));
    }
    nd_ = std::normal_distribution<float>(0.0f, 1.0f);

    return true;
  }

  void shutdown() override {}
  void reset() override { std::fill(nominal_.begin(), nominal_.end(), 0.0f); }

  void uploadCostmap(const uint8_t * d, int w, int h) override
  {
    dp_.size_x = w; dp_.size_y = h;
    if (!d) return;
    size_t n = static_cast<size_t>(w) * h;
    if (n > costmap_.size()) costmap_.resize(n);
    std::memcpy(costmap_.data(), d, n);
  }

  void uploadPath(const float * p, int n) override
  {
    dp_.path_len = n;
    size_t f = static_cast<size_t>(n) * 2;
    if (f > path_.size()) path_.resize(f);
    std::memcpy(path_.data(), p, f * sizeof(float));
  }

  void uploadNominal(const float * n) override
  {
    std::memcpy(nominal_.data(), n, nominal_.size() * sizeof(float));
  }

  void uploadState(float x, float y, float yaw,
                   float gx, float gy, float gyaw, bool gf) override
  {
    dp_.start_x = x; dp_.start_y = y; dp_.start_yaw = yaw;
    dp_.goal_x = gx; dp_.goal_y = gy; dp_.goal_yaw = gyaw;
    dp_.goal_is_final = gf ? 1 : 0;
  }

  // ---- Step 1: rollout ----
  void rollout() override
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < K_; ++k) {
      auto & rng = rngs_[k];
      float x = dp_.start_x, y = dp_.start_y, yaw = dp_.start_yaw;

      for (int t = 0; t < T_; ++t) {
        const bool perturb = (k != 0);
        float nv = perturb ? nd_(rng) * dp_.v_std : 0.0f;
        float nw = perturb ? nd_(rng) * dp_.w_std : 0.0f;

        const float v_raw = nominal_[t*3+0] + nv;
        float v = std::min(std::max(v_raw, dp_.v_min), dp_.v_max);
        float vy_raw = 0.0f, vy = 0.0f;
        if (dp_.motion_model == 2) {
          float nvy = perturb ? nd_(rng) * dp_.vy_std : 0.0f;
          vy_raw = nominal_[t*3+1] + nvy;
          vy = std::min(std::max(vy_raw, -dp_.vy_max), dp_.vy_max);
        }
        const float w_raw = nominal_[t*3+2] + nw;
        float w = std::min(std::max(w_raw, -dp_.w_max), dp_.w_max);
        if (dp_.motion_model == 1) {
          const float wd = std::fabs(v) / dp_.min_turning_r;
          w = std::min(std::max(w, -wd), wd);
        }

        const size_t ci = (static_cast<size_t>(k) * T_ + t) * 3;
        perturbed_[ci+0] = v_raw;
        perturbed_[ci+1] = vy_raw;
        perturbed_[ci+2] = w_raw;

        const float cy = std::cos(yaw), sy = std::sin(yaw);
        x   += dp_.dt * (v*cy - vy*sy);
        y   += dp_.dt * (v*sy + vy*cy);
        yaw  = wrap(yaw + dp_.dt * w);

        const size_t ti = static_cast<size_t>(k) * T_ + t;
        traj_x_[ti] = x; traj_y_[ti] = y; traj_yaw_[ti] = yaw;
      }
    }
  }

  // ---- Step 2: calculateCost ----
  void calculateCost() override
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < K_; ++k) {
      float cost = 0.0f;

      for (int t = 0; t < T_; ++t) {
        const size_t ti = static_cast<size_t>(k) * T_ + t;
        const float x = traj_x_[ti], y = traj_y_[ti], yaw = traj_yaw_[ti];

        // costmap
        if (dp_.size_x > 0 && !costmap_.empty()) {
          int mx = static_cast<int>((x - dp_.origin_x) / dp_.resolution);
          int my = static_cast<int>((y - dp_.origin_y) / dp_.resolution);
          if (mx >= 0 && mx < dp_.size_x && my >= 0 && my < dp_.size_y) {
            uint8_t c = costmap_[my * dp_.size_x + mx];
            if (c != 255) {
              if (c >= dp_.lethal_threshold) cost += dp_.collision_cost;
              else { float cn = c / 252.0f; cost += dp_.costmap_w * cn*cn * dp_.dt; }
            }
          }
        }

        // path cost
        if (dp_.path_len > 0) {
          float best = 1.0e18f; int best_i = 0;
          for (int i = 0; i < dp_.path_len; ++i) {
            float dx = x - path_[i*2+0], dy = y - path_[i*2+1];
            float d2 = dx*dx + dy*dy;
            if (d2 < best) { best = d2; best_i = i; }
          }
          cost += dp_.path_w * best * dp_.dt;
          int fi = std::min(best_i + dp_.follow_offset, dp_.path_len - 1);
          float fdx = x - path_[fi*2+0], fdy = y - path_[fi*2+1];
          cost += dp_.follow_w * std::sqrt(fdx*fdx + fdy*fdy) * dp_.dt;
        }

        // smoothness/backward/speed/angular
        const size_t ci = (static_cast<size_t>(k) * T_ + t) * 3;
        float cv = perturbed_[ci+0], cw = perturbed_[ci+2];
        cost += dp_.speed_w * (dp_.v_max - cv) * dp_.dt;
        cost += dp_.angular_w * cw * cw * dp_.dt;
        cost += dp_.backward_w * std::max(-cv, 0.0f) * dp_.dt;
      }

      // terminal
      const size_t li = static_cast<size_t>(k) * T_ + (T_ - 1);
      float gdx = traj_x_[li] - dp_.goal_x, gdy = traj_y_[li] - dp_.goal_y;
      cost += dp_.goal_w * std::sqrt(gdx*gdx + gdy*gdy);

      costs_[k] = cost;
    }
  }

  // ---- Step 3: reduce ----
  void reduce(const float * weights,
              float * ov, float * ovy, float * ow) override
  {
    std::fill(nominal_.begin(), nominal_.end(), 0.0f);
    const int CC = T_ * 3;
    for (int k = 0; k < K_; ++k) {
      const float w = weights[k];
      const int off = k * CC;
      for (int i = 0; i < CC; ++i) {
        nominal_[i] += w * perturbed_[off + i];
      }
    }
    *ov  = nominal_[0];
    *ovy = nominal_[1];
    *ow  = nominal_[2];
  }

  // ---- download ----
  void downloadCosts(float * h) override
  {
    std::memcpy(h, costs_.data(), K_ * sizeof(float));
  }

  void downloadNominal(float * h) override
  {
    std::memcpy(h, nominal_.data(), nominal_.size() * sizeof(float));
  }

  void buildDistanceField() override {}
  void updateVMax(float v) override { dp_.v_max = v; }

private:
  DeviceParams dp_;
  int K_ = 0, T_ = 0;

  std::vector<float> nominal_, perturbed_;
  std::vector<float> traj_x_, traj_y_, traj_yaw_, costs_;
  std::vector<uint8_t> costmap_;
  std::vector<float> path_;

  std::vector<std::mt19937> rngs_;
  std::normal_distribution<float> nd_;
};

}  // namespace

std::unique_ptr<IMppiBackend> createCpuBackend()
{
  return std::make_unique<CpuBackend>();
}

}  // namespace mppi_controller
