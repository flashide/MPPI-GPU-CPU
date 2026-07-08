// MPPI Solver — algorithm orchestration layer.
//
// Owns the 4-step MPPI pipeline:
//   1. Control sampling  (via backend.rollout)
//   2. Trajectory prediction  (via backend.rollout)
//   3. Cost evaluation  (via backend.calculateCost)
//   4. Optimal control update  (via backend.reduce)
//
// The solver is pure C++ and does NOT depend on any GPU API.
// It holds an IMppiBackend pointer and coordinates the pipeline,
// warm-start horizon shifting, retreat/recovery, and statistics.

#ifndef mppi_controller__MPPI_SOLVER_HPP_
#define mppi_controller__MPPI_SOLVER_HPP_

#include "mppi_controller/mppi_backend.hpp"
#include "mppi_controller/mppi_gpu.hpp"

#include <memory>
#include <vector>

namespace mppi_controller
{

class MppiSolver
{
public:
  // solver takes ownership of the backend
  explicit MppiSolver(std::unique_ptr<IMppiBackend> backend,
                      const MppiParams & params);
  ~MppiSolver();

  MppiSolver(const MppiSolver &) = delete;
  MppiSolver & operator=(const MppiSolver &) = delete;

  // ---- lifecycle ----
  void reset();
  void setSpeedLimit(float v_max);

  // ---- main entry point ----
  // Returns the optimal control (vx, vy, wz) and diagnostic info.
  // All poses are in the costmap global frame.
  MppiResult solve(
    float robot_x, float robot_y, float robot_yaw,
    const uint8_t * costmap, int size_x, int size_y,
    float origin_x, float origin_y, float resolution,
    const float * path_xy, int path_len,
    float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
    const float * footprint_xy, int footprint_len);

private:
  // Build the device parameter block from params + current cycle data.
  DeviceParams buildDeviceParams(
    float robot_x, float robot_y, float robot_yaw,
    int size_x, int size_y,
    float origin_x, float origin_y, float resolution,
    const float * path_xy, int path_len,
    float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
    const float * footprint_xy, int footprint_len) const;

  // Compute follow_offset from path spacing.
  static int computeFollowOffset(const float * path_xy, int path_len,
                                 float follow_lookahead);

  // Softmin weights on CPU.
  static void softminWeights(const float * costs, int K, float lambda,
                             float * weights,
                             float & min_cost, float & mean_cost);

  std::unique_ptr<IMppiBackend> backend_;
  MppiParams params_;
  float v_max_limit_;

  // host-side buffers
  std::vector<float> h_costs_;
  std::vector<float> h_weights_;
  std::vector<float> h_nominal_;
  std::vector<float> h_last_valid_nominal_;

  // state
  bool   has_last_valid_nominal_ = false;
  int    consecutive_all_colliding_ = 0;
};

}  // namespace mppi_controller

#endif  // mppi_controller__MPPI_SOLVER_HPP_
