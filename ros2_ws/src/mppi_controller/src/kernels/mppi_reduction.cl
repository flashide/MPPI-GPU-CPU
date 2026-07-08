// Step 3 — Reduction Kernel: softmin-weighted control update.
//
// Computes:
//   nominal[t][d] = Σ_k  weight[k] · perturbed[k][t][d]
//
// Also returns the first-step optimal control (vx, vy, wz) via a small
// 3-element output buffer.
//
// The softmin weights are computed on CPU (K floats, negligible cost for
// K ≤ 4096) and uploaded before this kernel launches.

#ifndef MPPI_REDUCTION_CL
#define MPPI_REDUCTION_CL

__kernel void reduce_controls(__global const float * perturbed,  // [K * T * 3]
                              __global const float * weights,    // [K]
                              __global float * nominal,          // [T * 3]  in/out
                              __global float * optimal,          // [3]      out
                              int K,
                              int T)
{
  const int idx = get_global_id(0);   // 0 .. T*3-1
  if (idx >= T * 3) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += weights[k] * perturbed[k * T * 3 + idx];
  }
  nominal[idx] = acc;

  // First work-item writes the first-step optimal control
  if (idx == 0) {
    float v_opt = 0.0f, vy_opt = 0.0f, w_opt = 0.0f;
    for (int k = 0; k < K; ++k) {
      v_opt  += weights[k] * perturbed[k * T * 3 + 0];
      vy_opt += weights[k] * perturbed[k * T * 3 + 1];
      w_opt  += weights[k] * perturbed[k * T * 3 + 2];
    }
    optimal[0] = v_opt;
    optimal[1] = vy_opt;
    optimal[2] = w_opt;
  }
}

#endif  // MPPI_REDUCTION_CL
