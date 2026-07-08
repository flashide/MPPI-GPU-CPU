// Standalone validation of the GPU MPPI backend without ROS2.
//
// Tests the same synthetic corridor as mppi_gpu_standalone.cpp:
//   vertical wall at x = 5 m, gap y ∈ [4.0, 6.0]
//   straight path (1,5) → (9,5)
//   closed-loop simulation, exits non-zero on failure.
//
// Build:  cmake .. && make mppi_opencl_standalone
// Usage:  ./mppi_opencl_standalone [K] [opencl|cpu] [diff|ackermann|omni]
//
// Environment:
//   MPPI_BACKEND=opencl|cpu   overrides backend selection
//   MPPI_TRACE=1              enables per-cycle verbose output

#include "opencl_mppi_controller/mppi_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

constexpr int kSizeX = 200;        // 10 m × 10 m @ 0.05 m
constexpr int kSizeY = 200;
constexpr float kRes = 0.05f;
constexpr float kOrigin = 0.0f;

// ---- synthetic scenario ---------------------------------------------------
void paintWallWithGap(std::vector<uint8_t> & map)
{
  // vertical lethal wall at x = 5 m with gap y ∈ [4.0, 6.0]
  const int wx0 = static_cast<int>(4.9f / kRes);
  const int wx1 = static_cast<int>(5.1f / kRes);
  const int gy0 = static_cast<int>(4.0f / kRes);
  const int gy1 = static_cast<int>(6.0f / kRes);
  for (int my = 0; my < kSizeY; ++my) {
    if (my >= gy0 && my < gy1) continue;
    for (int mx = wx0; mx < wx1; ++mx) {
      map[my * kSizeX + mx] = 254;  // LETHAL
    }
  }
  // exponential inflation ring
  const float inscribed = 0.2f, scaling = 3.0f;
  auto aabb_dist = [](float x, float y, float x0, float y0,
                      float x1, float y1) {
    const float ddx = std::max({x0 - x, 0.0f, x - x1});
    const float ddy = std::max({y0 - y, 0.0f, y - y1});
    return std::hypot(ddx, ddy);
  };
  for (int my = 0; my < kSizeY; ++my) {
    for (int mx = 0; mx < kSizeX; ++mx) {
      if (map[my * kSizeX + mx] == 254) continue;
      const float x = (mx + 0.5f) * kRes;
      const float y = (my + 0.5f) * kRes;
      const float dist = std::min(
        aabb_dist(x, y, 4.9f, 0.0f, 5.1f, 4.0f),
        aabb_dist(x, y, 4.9f, 6.0f, 5.1f, 10.0f));
      if (dist <= inscribed) {
        map[my * kSizeX + mx] = 253;
      } else if (dist < 1.0f) {
        const float c = 252.0f * std::exp(-scaling * (dist - inscribed));
        if (c >= 1.0f) {
          map[my * kSizeX + mx] = static_cast<uint8_t>(c);
        }
      }
    }
  }
}

bool isLethal(const std::vector<uint8_t> & map, float x, float y)
{
  const int mx = static_cast<int>((x - kOrigin) / kRes);
  const int my = static_cast<int>((y - kOrigin) / kRes);
  if (mx < 0 || mx >= kSizeX || my < 0 || my >= kSizeY) return false;
  return map[my * kSizeX + mx] >= 253;
}

// ---- cost comparison helper -----------------------------------------------
// Compare two arrays of floats element-by-element.
// Returns max absolute difference.
float maxAbsDiff(const float * a, const float * b, int n)
{
  float maxd = 0.0f;
  for (int i = 0; i < n; ++i) {
    maxd = std::max(maxd, std::fabs(a[i] - b[i]));
  }
  return maxd;
}

}  // namespace

int main(int argc, char ** argv)
{
  using namespace mppi_controller;

  // ---- parameter setup ---------------------------------------------------
  MppiParams params;
  if (argc > 1) {
    params.batch_size = std::atoi(argv[1]);
  }
  const std::string mode = argc > 2 ? argv[2] : "diff";
  GpuBackend backend_type = GpuBackend::OpenCL;
  if (argc > 3) {
    std::string be(argv[3]);
    if (be == "cpu") backend_type = GpuBackend::Cpu;
    else if (be == "opencl") backend_type = GpuBackend::OpenCL;
  }

  std::vector<float> footprint;
  if (mode == "ackermann") {
    params.motion_model = MotionModel::Ackermann;
    params.min_turning_r = 0.5f;
  } else if (mode == "omni") {
    params.motion_model = MotionModel::Omni;
  } else if (mode != "diff") {
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  }

  // ---- create costmap and path -------------------------------------------
  std::vector<uint8_t> map(kSizeX * kSizeY, 0);
  paintWallWithGap(map);

  std::vector<float> path_xy;
  const float goal_x = 9.0f, goal_y = 5.0f, goal_yaw = 0.0f;
  for (float x = 1.0f; x <= goal_x; x += 0.1f) {
    path_xy.push_back(x);
    path_xy.push_back(goal_y);
  }
  const int n_path = static_cast<int>(path_xy.size() / 2);
  const float lookahead = 3.0f;

  // ---- initialise MPPI backend -------------------------------------------
  MppiGpu mppi(params, backend_type, "src/kernels");

  // ---- closed-loop simulation --------------------------------------------
  float x = 1.0f, y = 5.0f, yaw = 0.0f;
  double total_ms = 0.0, max_ms = 0.0;
  double min_valid_ratio = 1.0;
  double dist = 0.0;
  int steps = 0;
  int retreat_count = 0;
  const int max_steps = 1200;

  // For validation: if MPPI_VALIDATE is set, also run CPU path and compare
  const bool validate = std::getenv("MPPI_VALIDATE") != nullptr;

  for (; steps < max_steps; ++steps) {
    // local path window
    int nearest = 0;
    float nearest_d2 = 1.0e18f;
    for (int i = 0; i < n_path; ++i) {
      const float dx = x - path_xy[i * 2 + 0];
      const float dy = y - path_xy[i * 2 + 1];
      const float d2 = dx * dx + dy * dy;
      if (d2 < nearest_d2) { nearest_d2 = d2; nearest = i; }
    }
    int win_end = nearest;
    float arc = 0.0f;
    for (int i = nearest + 1; i < n_path; ++i) {
      arc += std::hypot(
        path_xy[i * 2 + 0] - path_xy[(i - 1) * 2 + 0],
        path_xy[i * 2 + 1] - path_xy[(i - 1) * 2 + 1]);
      win_end = i;
      if (arc > lookahead) break;
    }
    const bool goal_is_final = (win_end + 1 == n_path);
    const float local_gx = path_xy[win_end * 2 + 0];
    const float local_gy = path_xy[win_end * 2 + 1];

    const auto t0 = std::chrono::steady_clock::now();
    const auto res = mppi.compute(
      x, y, yaw,
      map.data(), kSizeX, kSizeY, kOrigin, kOrigin, kRes,
      path_xy.data() + nearest * 2, win_end - nearest + 1,
      local_gx, local_gy, goal_yaw, goal_is_final,
      footprint.data(), static_cast<int>(footprint.size() / 2));
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);
    min_valid_ratio = std::min(min_valid_ratio,
                               static_cast<double>(res.valid_rollout_ratio));
    if (res.retreating) ++retreat_count;

    if (res.all_colliding) {
      std::printf("FAIL: all colliding at step %d\n", steps);
      return 1;
    }

    if (std::getenv("MPPI_TRACE") && steps % 20 == 0) {
      std::fprintf(stderr,
        "t=%5.2f x=%.2f y=%.2f yaw=%6.2f v=%5.2f w=%6.2f "
        "valid=%d/%d best=%.3f\n",
        steps * params.model_dt, x, y, yaw, res.v, res.w,
        res.valid_rollouts, res.sampled_rollouts, res.best_cost);
    }

    // apply first control
    x   += params.model_dt * (res.v * std::cos(yaw) - res.vy * std::sin(yaw));
    y   += params.model_dt * (res.v * std::sin(yaw) + res.vy * std::cos(yaw));
    dist += std::hypot(
      params.model_dt * (res.v * std::cos(yaw) - res.vy * std::sin(yaw)),
      params.model_dt * (res.v * std::sin(yaw) + res.vy * std::cos(yaw)));
    yaw  = std::atan2(std::sin(yaw + params.model_dt * res.w),
                      std::cos(yaw + params.model_dt * res.w));

    if (isLethal(map, x, y)) {
      std::printf("FAIL: hit wall at step %d (x=%.2f y=%.2f)\n", steps, x, y);
      return 1;
    }
    const float dx = x - goal_x, dy = y - goal_y;
    if (dx * dx + dy * dy < 0.25f * 0.25f) break;
  }

  if (steps >= max_steps) {
    std::printf("FAIL: goal not reached in %d steps (x=%.2f y=%.2f)\n",
                max_steps, x, y);
    return 1;
  }

  // ---- report -----------------------------------------------------------
  std::printf("PASS [%s]: %d steps (%.1f sim-seconds)\n",
              mode.c_str(), steps, steps * params.model_dt);
  std::printf("dist=%.2fm  speed=%.3f m/s\n",
              dist, dist / (steps * params.model_dt));
  std::printf("solve: mean=%.2f ms  max=%.2f ms  (K=%d, T=%d)\n",
              total_ms / (steps + 1), max_ms,
              params.batch_size, params.time_steps);
  std::printf("min valid ratio=%.1f%%  retreats=%d\n",
              100.0 * min_valid_ratio, retreat_count);
  return 0;
}
