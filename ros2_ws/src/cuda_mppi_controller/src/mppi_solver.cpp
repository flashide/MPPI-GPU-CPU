// MppiSolver — MPPI pipeline orchestrator.
//
// Coordinates the 4-step pipeline across the GPU backend:
//   Step 1+2  rollout()         control sampling + state propagation
//   Step 3    calculateCost()   per-trajectory cost evaluation
//   Step 4    reduce()          softmin-weighted control update
//
// CPU-side responsibilities:
//   - warm-start horizon shifting
//   - retreat / recovery logic
//   - statistics aggregation

#include "cuda_mppi_controller/mppi_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cuda_mppi_controller
{

// ---- helpers --------------------------------------------------------------
namespace
{

int computeFollowOffsetImpl(const float * path_xy, int path_len,
                             float follow_lookahead)
{
  if (path_len <= 1) return 1;
  float arc = 0.0f;
  for (int i = 1; i < path_len; ++i) {
    arc += std::hypot(
      path_xy[i * 2 + 0] - path_xy[(i - 1) * 2 + 0],
      path_xy[i * 2 + 1] - path_xy[(i - 1) * 2 + 1]);
  }
  const float spacing = arc / static_cast<float>(path_len - 1);
  if (spacing <= 1.0e-6f) return 1;
  return std::max(1, std::min(path_len - 1,
    static_cast<int>(std::lround(follow_lookahead / spacing))));
}

}  // namespace

// ---- MppiSolver -----------------------------------------------------------
MppiSolver::MppiSolver(std::unique_ptr<IMppiBackend> backend,
                       const MppiParams & params)
  : backend_(std::move(backend))
  , params_(params)
  , v_max_limit_(params.v_max)
{
  const int K = params_.batch_size;
  const int T = params_.time_steps;

  h_costs_.resize(K);
  h_weights_.resize(K);
  h_nominal_.assign(T * kCtrlDim, 0.0f);
  h_last_valid_nominal_.assign(T * kCtrlDim, 0.0f);

  // Build initial DeviceParams and initialise the backend
  DeviceParams dp_init = {};
  dp_init.K = K;
  dp_init.T = T;
  dp_init.dt = params_.model_dt;
  dp_init.motion_model = static_cast<int>(params_.motion_model);
  dp_init.v_max = std::min(params_.v_max, v_max_limit_);
  dp_init.v_min = params_.v_min;
  dp_init.vy_max = params_.vy_max;
  dp_init.w_max = params_.w_max;
  dp_init.min_turning_r = std::max(params_.min_turning_r, 1.0e-3f);
  dp_init.v_std  = params_.v_std;
  dp_init.vy_std = params_.vy_std;
  dp_init.w_std  = params_.w_std;

  if (!backend_->initialize(dp_init, 42ULL)) {
    throw std::runtime_error("MppiSolver: backend initialization failed");
  }
  backend_->uploadNominal(h_nominal_.data());
}

MppiSolver::~MppiSolver()
{
  if (backend_) backend_->shutdown();
}

void MppiSolver::reset()
{
  std::fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
  std::fill(h_last_valid_nominal_.begin(), h_last_valid_nominal_.end(), 0.0f);
  has_last_valid_nominal_ = false;
  consecutive_all_colliding_ = 0;
  backend_->reset();
}

void MppiSolver::setSpeedLimit(float v_max)
{
  v_max_limit_ = v_max;
  backend_->updateVMax(v_max);
}

// ---- softmin (CPU) --------------------------------------------------------
void MppiSolver::softminWeights(const float * costs, int K, float lambda,
                                float * weights,
                                float & min_cost, float & mean_cost)
{
  min_cost = *std::min_element(costs, costs + K);

  double wsum = 0.0;
  for (int k = 0; k < K; ++k) {
    weights[k] = std::exp(-(costs[k] - min_cost) / lambda);
    wsum += weights[k];
  }
  const float inv = static_cast<float>(1.0 / wsum);
  double csum = 0.0;
  for (int k = 0; k < K; ++k) {
    weights[k] *= inv;
    csum += costs[k];
  }
  mean_cost = static_cast<float>(csum / static_cast<double>(K));
}

// ---- solve ----------------------------------------------------------------
MppiResult MppiSolver::solve(
  float robot_x, float robot_y, float robot_yaw,
  const uint8_t * costmap, int size_x, int size_y,
  float origin_x, float origin_y, float resolution,
  const float * path_xy, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len)
{
  const int K = params_.batch_size;
  const int T = params_.time_steps;

  // ---- compute follow offset ----
  path_len = std::min(path_len, kMaxPathPoints);
  int follow_offset = computeFollowOffsetImpl(
    path_xy, path_len, params_.follow_lookahead);

  // ---- build device params ----
  DeviceParams dp = buildDeviceParams(
    robot_x, robot_y, robot_yaw,
    size_x, size_y, origin_x, origin_y, resolution,
    path_xy, path_len,
    goal_x, goal_y, goal_yaw, goal_is_final,
    footprint_xy, footprint_len);
  dp.follow_offset = follow_offset;

  // ---- upload data to GPU ----
  if (dp.size_x > 0 && costmap) {
    backend_->uploadCostmap(costmap, size_x, size_y);
  } else {
    backend_->uploadCostmap(nullptr, 0, 0);
  }

  if (path_len > 0) {
    backend_->uploadPath(path_xy, path_len);
  }

  backend_->uploadState(robot_x, robot_y, robot_yaw,
                        goal_x, goal_y, goal_yaw, goal_is_final);

  backend_->updateVMax(dp.v_max);

  // Build distance field if enabled
  if (dp.size_x > 0 &&
      params_.distance_field_weight > 0.0f &&
      params_.distance_field_cutoff > 1.0e-6f) {
    backend_->buildDistanceField();
  }

  // ---- MPPI iteration loop -----------------------------------------------
  float min_cost = 0.0f;
  float opt_v = 0.0f, opt_vy = 0.0f, opt_w = 0.0f;

  for (int iter = 0; iter < params_.iteration_count; ++iter) {
    // Step 1+2: rollout (control sampling + state propagation)
    backend_->rollout();

    // Step 3: cost evaluation
    backend_->calculateCost();

    // Download costs for softmin on CPU
    backend_->downloadCosts(h_costs_.data());

    // CPU softmin weights
    float mean_cost_unused;
    softminWeights(h_costs_.data(), K, params_.lambda,
                   h_weights_.data(), min_cost, mean_cost_unused);

    // Step 4: reduction → update nominal + return optimal control
    backend_->reduce(h_weights_.data(), &opt_v, &opt_vy, &opt_w);
  }

  // ---- download final nominal ----
  backend_->downloadNominal(h_nominal_.data());

  // ---- anti-windup clamp ----
  const float v_max_eff = std::min(params_.v_max, v_max_limit_);
  for (int t = 0; t < T; ++t) {
    float & nv  = h_nominal_[t * kCtrlDim + 0];
    float & nvy = h_nominal_[t * kCtrlDim + 1];
    float & nw  = h_nominal_[t * kCtrlDim + 2];
    nv  = std::min(std::max(nv, dp.v_min), v_max_eff + params_.v_std);
    nvy = std::min(std::max(nvy, -params_.vy_max - params_.vy_std),
                   params_.vy_max + params_.vy_std);
    nw  = std::min(std::max(nw, -params_.w_max), params_.w_max);
  }

  // ---- statistics ----
  MppiResult res;
  res.sampled_rollouts = K;
  double cost_sum = 0.0;
  int valid_rollouts = 0;
  for (const float c : h_costs_) {
    cost_sum += c;
    if (c < params_.collision_cost) ++valid_rollouts;
  }
  res.best_cost   = min_cost;
  res.mean_cost   = static_cast<float>(cost_sum / static_cast<double>(K));
  res.valid_rollouts = valid_rollouts;
  res.valid_rollout_ratio = static_cast<float>(valid_rollouts) /
                            static_cast<float>(K);
  res.all_colliding = (min_cost >= params_.collision_cost);

  // ---- retreat / recovery ----
  if (res.all_colliding) {
    ++consecutive_all_colliding_;
    if (params_.enable_retreat && has_last_valid_nominal_) {
      const int retreat_step = std::min(consecutive_all_colliding_ - 1, T - 1);
      const int offset = retreat_step * kCtrlDim;
      const float rs = std::max(0.0f, params_.retreat_scale);
      res.v = std::min(std::max(-rs * h_last_valid_nominal_[offset + 0],
                                dp.v_min), v_max_eff);
      res.vy = std::min(std::max(-rs * h_last_valid_nominal_[offset + 1],
                                 -params_.vy_max), params_.vy_max);
      res.w = std::min(std::max(-rs * h_last_valid_nominal_[offset + 2],
                                -params_.w_max), params_.w_max);
      if (params_.motion_model == MotionModel::Ackermann) {
        const float w_dyn = std::fabs(res.v) /
                            std::max(params_.min_turning_r, 1.0e-3f);
        res.w = std::min(std::max(res.w, -w_dyn), w_dyn);
      }
      res.retreating = true;
      backend_->uploadNominal(h_last_valid_nominal_.data());
    } else {
      std::fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
      backend_->uploadNominal(h_nominal_.data());
    }
    return res;
  }

  // ---- normal exit ----
  consecutive_all_colliding_ = 0;
  h_last_valid_nominal_ = h_nominal_;
  has_last_valid_nominal_ = true;

  res.v  = std::min(std::max(opt_v,  dp.v_min), v_max_eff);
  res.vy = std::min(std::max(opt_vy, -params_.vy_max), params_.vy_max);
  res.w  = std::min(std::max(opt_w,  -params_.w_max), params_.w_max);
  if (params_.motion_model == MotionModel::Ackermann) {
    const float w_dyn = std::fabs(res.v) /
                        std::max(params_.min_turning_r, 1.0e-3f);
    res.w = std::min(std::max(res.w, -w_dyn), w_dyn);
  }

  // ---- warm-start: shift horizon + repeat last control ----
  for (int t = 0; t < T - 1; ++t) {
    for (int d = 0; d < kCtrlDim; ++d) {
      h_nominal_[t * kCtrlDim + d] = h_nominal_[(t + 1) * kCtrlDim + d];
    }
  }
  backend_->uploadNominal(h_nominal_.data());

  return res;
}

// ---- buildDeviceParams ----------------------------------------------------
DeviceParams MppiSolver::buildDeviceParams(
  float robot_x, float robot_y, float robot_yaw,
  int size_x, int size_y,
  float origin_x, float origin_y, float resolution,
  const float * /*path_xy*/, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len) const
{
  DeviceParams dp = {};
  dp.K  = params_.batch_size;
  dp.T  = params_.time_steps;
  dp.dt = params_.model_dt;
  dp.motion_model = static_cast<int>(params_.motion_model);

  dp.v_max = std::min(params_.v_max, v_max_limit_);
  dp.v_min = params_.v_min;
  dp.vy_max = params_.vy_max;
  dp.w_max  = params_.w_max;
  dp.min_turning_r = std::max(params_.min_turning_r, 1.0e-3f);

  dp.v_std  = params_.v_std;
  dp.vy_std = params_.vy_std;
  dp.w_std  = params_.w_std;

  dp.size_x = (size_x > 0) ? size_x : 0;
  dp.size_y = (size_y > 0) ? size_y : 0;
  dp.origin_x  = origin_x;
  dp.origin_y  = origin_y;
  dp.resolution = resolution;

  dp.goal_w     = params_.goal_weight;
  dp.goal_yaw_w = params_.goal_yaw_weight;
  dp.path_w     = params_.path_weight;
  dp.follow_w   = params_.path_follow_weight;
  dp.path_angle_w      = params_.path_angle_weight;
  dp.curvature_speed_w = params_.curvature_speed_weight;
  dp.curvature_speed_min = params_.curvature_speed_min;
  dp.costmap_w         = params_.costmap_weight;
  dp.distance_field_w  = params_.distance_field_weight;
  dp.distance_field_cutoff = params_.distance_field_cutoff;
  dp.smooth_w   = params_.smoothness_weight;
  dp.backward_w = params_.backward_weight;
  dp.speed_w    = params_.speed_weight;
  dp.angular_w  = params_.angular_weight;
  dp.collision_cost     = params_.collision_cost;
  dp.yaw_activation_dist = params_.yaw_goal_activation_dist;
  dp.lethal_threshold   = params_.lethal_threshold;

  dp.path_len  = path_len;
  dp.goal_x    = goal_x;
  dp.goal_y    = goal_y;
  dp.goal_yaw  = goal_yaw;
  dp.goal_is_final = goal_is_final ? 1 : 0;

  dp.footprint_len = 0;
  if (params_.consider_footprint && footprint_xy && footprint_len >= 3) {
    dp.footprint_len = std::min(footprint_len, kMaxFootprint);
    for (int i = 0; i < dp.footprint_len * 2; ++i) {
      dp.fp[i] = footprint_xy[i];
    }
  }

  dp.start_x   = robot_x;
  dp.start_y   = robot_y;
  dp.start_yaw = robot_yaw;

  return dp;
}

}  // namespace cuda_mppi_controller
