// Step 1 — Rollout Kernel: control sampling + state propagation.
//
// One work-item = one trajectory (K parallel threads).
// Each thread:
//   1. Loads private RNG state
//   2. For t = 0..T-1:
//      a. Samples perturbed controls (k==0 unperturbed = nominal)
//      b. Integrates kinematic model:  (x,y,yaw)_{t+1} = f((x,y,yaw)_t, u_t)
//   3. Stores full trajectory and perturbed controls to device buffers
//
// No cost evaluation happens here — that's Step 2 (mppi_cost.cl).

#ifndef MPPI_ROLLOUT_CL
#define MPPI_ROLLOUT_CL

#include "mppi_rng.cl"

// ---- helpers --------------------------------------------------------------
inline float wrap_angle_f(float a) { return atan2(sin(a), cos(a)); }

// ---- constant param block (identical to the one in mppi_cost.cl) ----------
// Mali-G610: __constant memory has 64 KB dedicated cache.
struct RolloutParams {
  int K, T;
  float dt;
  int motion_model;
  float v_max, v_min, vy_max, w_max;
  float min_turning_r;
  float v_std, vy_std, w_std;
  float start_x, start_y, start_yaw;
};

// ---- main kernel ----------------------------------------------------------
__kernel void rollout(__global const float * nominal,       // [T * 3]
                      __global float * perturbed,           // [K * T * 3] out
                      __global float * traj_x,              // [K * T] out
                      __global float * traj_y,              // [K * T] out
                      __global float * traj_yaw,            // [K * T] out
                      __global rng_state_t * rng_states,
                      __constant struct RolloutParams * p)
{
  const int k = get_global_id(0);
  if (k >= p->K) return;

  // -- private RNG --
  rng_state_t rng = rng_states[k];

  // -- initial state --
  float x = p->start_x, y = p->start_y, yaw = p->start_yaw;

  // -- time-step loop --
  for (int t = 0; t < p->T; ++t) {
    // control sampling
    const int perturb = (k != 0);
    float nv = perturb ? rng_normal(&rng) * p->v_std : 0.0f;
    float nw = perturb ? rng_normal(&rng) * p->w_std : 0.0f;

    const float v_raw = nominal[t * 3 + 0] + nv;
    float v = clamp(v_raw, p->v_min, p->v_max);

    float vy_raw = 0.0f, vy = 0.0f;
    if (p->motion_model == 2) {   // Omni
      float nvy = perturb ? rng_normal(&rng) * p->vy_std : 0.0f;
      vy_raw = nominal[t * 3 + 1] + nvy;
      vy = clamp(vy_raw, -p->vy_max, p->vy_max);
    }

    const float w_raw = nominal[t * 3 + 2] + nw;
    float w = clamp(w_raw, -p->w_max, p->w_max);
    if (p->motion_model == 1) {   // Ackermann
      const float w_dyn = fabs(v) / p->min_turning_r;
      w = clamp(w, -w_dyn, w_dyn);
    }

    // store perturbed controls (unclamped, for weighted update)
    const int ctrl_idx = (k * p->T + t) * 3;
    perturbed[ctrl_idx + 0] = v_raw;
    perturbed[ctrl_idx + 1] = vy_raw;
    perturbed[ctrl_idx + 2] = w_raw;

    // state propagation (kinematic bicycle / diff-drive / omni)
    const float cy = cos(yaw), sy = sin(yaw);
    x   += p->dt * (v * cy - vy * sy);
    y   += p->dt * (v * sy + vy * cy);
    yaw  = wrap_angle_f(yaw + p->dt * w);

    // store trajectory
    const int traj_idx = k * p->T + t;
    traj_x[traj_idx]   = x;
    traj_y[traj_idx]   = y;
    traj_yaw[traj_idx] = yaw;
  }

  // write-back RNG state
  rng_states[k] = rng;
}

#endif  // MPPI_ROLLOUT_CL
