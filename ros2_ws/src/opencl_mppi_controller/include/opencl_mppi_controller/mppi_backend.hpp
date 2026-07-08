// GPU backend abstraction for MPPI — per Mppi-solver分步计算架构.md.
//
// The backend exposes the three core GPU-accelerated steps:
//   1. rollout()       — control sampling + state propagation
//   2. calculateCost() — per-trajectory cost evaluation
//   3. reduce()        — softmin-weighted control update
//
// Pure C++ interface — no GPU headers exposed.

#ifndef opencl_mppi_controller__MPPI_BACKEND_HPP_
#define opencl_mppi_controller__MPPI_BACKEND_HPP_

#include <cstddef>
#include <cstdint>

namespace mppi_controller
{

// ---- enums & constants ----------------------------------------------------
enum class GpuBackend
{
  Cpu,
  OpenCL,
  CUDA,
  Vulkan
};

constexpr int kCtrlDim       = 3;    // (vx, vy, wz)
constexpr int kMaxPathPoints  = 256;
constexpr int kMaxFootprint   = 16;

// Parameter block mirrored on device constant memory.
struct DeviceParams
{
  // dimensions
  int K, T;
  float dt;
  int motion_model;                   // 0 DiffDrive, 1 Ackermann, 2 Omni

  // control limits
  float v_max, v_min, vy_max, w_max;
  float min_turning_r;

  // sampling noise std
  float v_std, vy_std, w_std;

  // costmap
  int size_x, size_y;
  float origin_x, origin_y, resolution;

  // cost weights
  float goal_w, goal_yaw_w;
  float path_w, follow_w, path_angle_w;
  float curvature_speed_w, curvature_speed_min;
  float costmap_w;
  float distance_field_w, distance_field_cutoff;
  float smooth_w, backward_w;
  float speed_w, angular_w;
  int   follow_offset;
  float collision_cost;
  float yaw_activation_dist;
  uint8_t lethal_threshold;

  // path
  int   path_len;
  float goal_x, goal_y, goal_yaw;
  int   goal_is_final;

  // footprint
  int   footprint_len;
  float fp[kMaxFootprint * 2];

  // start state
  float start_x, start_y, start_yaw;
};

// ---- backend interface ----------------------------------------------------
class IMppiBackend
{
public:
  virtual ~IMppiBackend() = default;

  // -- lifecycle --
  virtual bool initialize(const DeviceParams & dp, uint64_t rng_seed) = 0;
  virtual void shutdown() = 0;
  virtual void reset() = 0;          // zero nominal, re-seed RNG

  // -- data upload (Host → Device) --
  virtual void uploadCostmap(const uint8_t * data, int w, int h) = 0;
  virtual void uploadPath(const float * path_xy, int n) = 0;
  virtual void uploadNominal(const float * nominal) = 0;    // [T * 3]
  virtual void uploadState(float x, float y, float yaw,
                           float gx, float gy, float gyaw, bool goal_final) = 0;

  // -- MPPI pipeline steps (each is a GPU kernel launch sequence) --
  //
  // Step 1 — rollout:
  //   Sample K perturbed control sequences (Gaussian noise around nominal),
  //   propagate the kinematic model for T steps, store trajectories.
  //   k==0 always carries the unperturbed nominal.
  virtual void rollout() = 0;

  // Step 2 — calculateCost:
  //   For each stored trajectory, evaluate obstacle / path / goal /
  //   smoothness / speed / angular costs.  Write per-trajectory total
  //   cost to device cost buffer.
  virtual void calculateCost() = 0;

  // Step 3 — reduce:
  //   weights_host[0..K-1] are pre-computed softmin-normalised weights
  //   (caller computes them on CPU from costs).  The backend uploads
  //   weights, then runs the weighted-average reduction kernel:
  //     nominal[t][d] = Σ_k weight[k] · perturbed[k][t][d]
  //   Returns the first-step optimal control (vx, vy, wz) via host pointers.
  virtual void reduce(const float * weights_host,
                      float * opt_v, float * opt_vy, float * opt_w) = 0;

  // -- data download (Device → Host) --
  // Useful for diagnostics, retreat logic, and warm-start management.
  virtual void downloadCosts(float * costs_host) = 0;       // [K]
  virtual void downloadNominal(float * nominal_host) = 0;   // [T * 3]

  // -- optional --
  virtual void buildDistanceField() = 0;
  virtual void updateVMax(float v_max) = 0;

  // -- queries --
  virtual GpuBackend type() const = 0;
  virtual const char * name() const = 0;
};

}  // namespace mppi_controller

#endif  // opencl_mppi_controller__MPPI_BACKEND_HPP_
