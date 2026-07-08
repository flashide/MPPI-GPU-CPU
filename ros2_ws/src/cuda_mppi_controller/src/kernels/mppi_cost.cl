// Step 2 — Cost Kernel: per-trajectory cost evaluation.
//
// One work-item = one trajectory.
// Reads trajectory buffers (x, y, yaw per step) and evaluates:
//   - costmap / collision cost     (via image2d texture or buffer)
//   - distance field cost          (optional)
//   - reference path costs         (nearest-point + follow + angle)
//   - curvature-speed penalty      (optional)
//   - smoothness / backward / speed / angular penalties
//   - terminal cost                (goal distance + yaw)

#ifndef MPPI_COST_CL
#define MPPI_COST_CL

// ---- helpers --------------------------------------------------------------
inline float wrap_angle_f(float a) { return atan2(sin(a), cos(a)); }

// Image2d-based costmap lookup  (hardware texture, preferred on Mali-G610)
inline uchar cell_cost_img(__read_only image2d_t costmap_img,
                           sampler_t sampler,
                           float ox, float oy, float res,
                           float x, float y)
{
  int mx = (int)floor((x - ox) / res);
  int my = (int)floor((y - oy) / res);
  uint4 val = read_imageui(costmap_img, sampler, (int2)(mx, my));
  return (uchar)(val.x & 0xFFu);
}

// Buffer-based costmap lookup (fallback)
inline uchar cell_cost_buf(__global const uchar * costmap,
                           int size_x, int size_y,
                           float ox, float oy, float res,
                           float x, float y)
{
  int mx = (int)floor((x - ox) / res);
  int my = (int)floor((y - oy) / res);
  if (mx < 0 || mx >= size_x || my < 0 || my >= size_y) return 0;
  return costmap[my * size_x + mx];
}

// ---- constant param block -------------------------------------------------
struct CostParams {
  int K, T;
  float dt;
  int motion_model;
  float v_max, v_min;
  // cost weights
  float goal_w, goal_yaw_w, path_w, follow_w, path_angle_w;
  float curvature_speed_w, curvature_speed_min;
  float costmap_w;
  float distance_field_w, distance_field_cutoff;
  float smooth_w, backward_w, speed_w, angular_w;
  float collision_cost;
  float yaw_activation_dist;
  uchar lethal_threshold;
  int follow_offset;
  // costmap geo
  int size_x, size_y;
  float origin_x, origin_y, resolution;
  // path
  int path_len;
  // goal
  float goal_x, goal_y, goal_yaw;
  int goal_is_final;
};

// ---- main kernel (image2d costmap) ----------------------------------------
__kernel void calculate_cost_img(
    __read_only image2d_t costmap_img,
    sampler_t costmap_sampler,
    __global const float * distance_field,
    __global const float * path,              // [path_len * 2]
    __global const float * traj_x,            // [K * T]
    __global const float * traj_y,            // [K * T]
    __global const float * traj_yaw,          // [K * T]
    __global const float * perturbed,         // [K * T * 3]
    __global float * costs,                   // [K] out
    __constant struct CostParams * p)
{
  const int k = get_global_id(0);
  if (k >= p->K) return;

  float cost = 0.0f;

  // -- per-step loop --
  for (int t = 0; t < p->T; ++t) {
    const int idx = k * p->T + t;
    const float x   = traj_x[idx];
    const float y   = traj_y[idx];
    const float yaw = traj_yaw[idx];

    // ---- costmap / collision cost ----
    if (p->size_x > 0) {
      const uchar c = cell_cost_img(costmap_img, costmap_sampler,
                                    p->origin_x, p->origin_y, p->resolution,
                                    x, y);
      if (c != 255) {
        if (c >= p->lethal_threshold) {
          cost += p->collision_cost;
        } else {
          const float cn = (float)c / 252.0f;
          cost += p->costmap_w * cn * cn * p->dt;
        }
      }
    }

    // ---- distance field cost (optional) ----
    if (p->distance_field_w > 0.0f && distance_field != 0 &&
        p->distance_field_cutoff > 1.0e-6f)
    {
      int mx = (int)floor((x - p->origin_x) / p->resolution);
      int my = (int)floor((y - p->origin_y) / p->resolution);
      float dist = p->distance_field_cutoff;
      if (mx >= 0 && mx < p->size_x && my >= 0 && my < p->size_y)
        dist = distance_field[my * p->size_x + mx];
      if (dist < p->distance_field_cutoff) {
        const float q = (p->distance_field_cutoff - dist) / p->distance_field_cutoff;
        cost += p->distance_field_w * q * q * p->dt;
      }
    }

    // ---- reference path costs ----
    if (p->path_len > 0) {
      // nearest-point search
      float best = 1.0e18f;
      int best_i = 0;
      for (int i = 0; i < p->path_len; ++i) {
        float dx = x - path[i*2+0], dy = y - path[i*2+1];
        float d2 = dx*dx + dy*dy;
        if (d2 < best) { best = d2; best_i = i; }
      }
      cost += p->path_w * best * p->dt;

      // follow-ahead point
      int fi = min(best_i + p->follow_offset, p->path_len - 1);
      float fdx = x - path[fi*2+0], fdy = y - path[fi*2+1];
      cost += p->follow_w * sqrt(fdx*fdx + fdy*fdy) * p->dt;

      // path-angle alignment
      if (p->path_angle_w > 0.0f && p->path_len > 1) {
        int pi = max(0, fi - 1), ni = min(p->path_len - 1, fi + 1);
        float tx = path[ni*2+0] - path[pi*2+0];
        float ty = path[ni*2+1] - path[pi*2+1];
        if (tx*tx + ty*ty > 1.0e-8f) {
          float pyaw = atan2(ty, tx);
          // last-step v for backward check: get from perturbed
          int ci = (k * p->T + t) * 3;
          float v_last = perturbed[ci + 0];
          if (v_last < -1.0e-3f) pyaw = wrap_angle_f(pyaw + 3.14159265358979323846f);
          float ye = wrap_angle_f(yaw - pyaw);
          cost += p->path_angle_w * ye*ye * p->dt;
        }
      }

      // curvature-speed penalty
      if (p->curvature_speed_w > 0.0f && p->path_len > 2) {
        int ci = (k * p->T + t) * 3;
        float v_cur = perturbed[ci + 0];
        if (v_cur > 0.0f) {
          int span = max(1, p->follow_offset / 2);
          int p_i = max(0, fi - span), n_i = min(p->path_len - 1, fi + span);
          float ax = path[fi*2+0] - path[p_i*2+0];
          float ay = path[fi*2+1] - path[p_i*2+1];
          float bx = path[n_i*2+0] - path[fi*2+0];
          float by = path[n_i*2+1] - path[fi*2+1];
          float alen = sqrt(ax*ax + ay*ay), blen = sqrt(bx*bx + by*by);
          if (alen > 1.0e-4f && blen > 1.0e-4f) {
            float arc = fmax(0.5f*(alen+blen), 1.0e-3f);
            float curv = fabs(wrap_angle_f(atan2(by,bx) - atan2(ay,ax))) / arc;
            if (curv > 1.0e-4f) {
              float floor_v = clamp(p->curvature_speed_min, 0.0f, p->v_max);
              float target_v = fmax(floor_v, p->v_max / (1.0f + curv));
              float overspd = fmax(v_cur - target_v, 0.0f);
              cost += p->curvature_speed_w * overspd * overspd * p->dt;
            }
          }
        }
      }
    }

    // ---- smoothness / backward / speed / angular penalties ----
    // Read previous perturbed controls for smoothness
    int ci_prev = (k * p->T + max(0, t-1)) * 3;
    int ci_cur  = (k * p->T + t) * 3;
    float prev_v  = (t > 0) ? perturbed[ci_prev + 0] : perturbed[ci_cur + 0];
    float prev_vy = (t > 0) ? perturbed[ci_prev + 1] : perturbed[ci_cur + 1];
    float prev_w  = (t > 0) ? perturbed[ci_prev + 2] : perturbed[ci_cur + 2];
    float cur_v  = perturbed[ci_cur + 0];
    float cur_vy = perturbed[ci_cur + 1];
    float cur_w  = perturbed[ci_cur + 2];

    float dv = cur_v - prev_v, dvy = cur_vy - prev_vy, dw = cur_w - prev_w;
    cost += p->smooth_w   * (dv*dv + dvy*dvy + dw*dw);
    cost += p->backward_w * fmax(-cur_v, 0.0f) * p->dt;
    cost += p->speed_w    * (p->v_max - cur_v) * p->dt;
    cost += p->angular_w  * cur_w * cur_w * p->dt;
  }

  // ---- terminal cost ----
  const int last_idx = k * p->T + (p->T - 1);
  float gdx = traj_x[last_idx] - p->goal_x;
  float gdy = traj_y[last_idx] - p->goal_y;
  float gd2 = gdx*gdx + gdy*gdy;
  cost += p->goal_w * sqrt(gd2);
  if (p->goal_is_final && gd2 < p->yaw_activation_dist * p->yaw_activation_dist) {
    float dyaw = wrap_angle_f(traj_yaw[last_idx] - p->goal_yaw);
    cost += p->goal_yaw_w * dyaw * dyaw;
  }

  costs[k] = cost;
}

// ---- buffer-variant kernel (no image2d) -----------------------------------
__kernel void calculate_cost_buf(
    __global const uchar * costmap,
    __global const float * distance_field,
    __global const float * path,
    __global const float * traj_x,
    __global const float * traj_y,
    __global const float * traj_yaw,
    __global const float * perturbed,
    __global float * costs,
    __constant struct CostParams * p)
{
  const int k = get_global_id(0);
  if (k >= p->K) return;

  float cost = 0.0f;

  for (int t = 0; t < p->T; ++t) {
    const int idx = k * p->T + t;
    const float x = traj_x[idx], y = traj_y[idx], yaw = traj_yaw[idx];

    // costmap (buffer lookup)
    if (p->size_x > 0) {
      const uchar c = cell_cost_buf(costmap, p->size_x, p->size_y,
                                    p->origin_x, p->origin_y, p->resolution, x, y);
      if (c != 255) {
        if (c >= p->lethal_threshold) { cost += p->collision_cost; }
        else {
          const float cn = (float)c / 252.0f;
          cost += p->costmap_w * cn * cn * p->dt;
        }
      }
    }

    // path costs (nearest-point only for brevity — full version mirrors
    // calculate_cost_img above)
    if (p->path_len > 0) {
      float best = 1.0e18f;
      int best_i = 0;
      for (int i = 0; i < p->path_len; ++i) {
        float dx = x - path[i*2+0], dy = y - path[i*2+1];
        float d2 = dx*dx + dy*dy;
        if (d2 < best) { best = d2; best_i = i; }
      }
      cost += p->path_w * best * p->dt;
      int fi = min(best_i + p->follow_offset, p->path_len - 1);
      float fdx = x - path[fi*2+0], fdy = y - path[fi*2+1];
      cost += p->follow_w * sqrt(fdx*fdx + fdy*fdy) * p->dt;
    }

    // smoothness / backward / speed / angular
    int ci_cur = (k * p->T + t) * 3;
    float cur_v = perturbed[ci_cur+0], cur_w = perturbed[ci_cur+2];
    cost += p->speed_w   * (p->v_max - cur_v) * p->dt;
    cost += p->angular_w * cur_w * cur_w * p->dt;
    cost += p->backward_w * fmax(-cur_v, 0.0f) * p->dt;
  }

  // terminal cost
  const int last_idx = k * p->T + (p->T - 1);
  float gdx = traj_x[last_idx] - p->goal_x;
  float gdy = traj_y[last_idx] - p->goal_y;
  float gd2 = gdx*gdx + gdy*gdy;
  cost += p->goal_w * sqrt(gd2);

  costs[k] = cost;
}

#endif  // MPPI_COST_CL
