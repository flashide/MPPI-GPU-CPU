// Build a truncated Euclidean Signed Distance Field (ESDF) from a costmap.
//
// For each free/unknown cell, search a window around it for obstacle cells
// and record the minimum distance.  Cutoff parameter limits the search radius
// so the kernel stays cheap.
//
// This is the OpenCL equivalent of build_distance_field_kernel in mppi_gpu.cu.

#ifndef MPPI_DISTANCE_FIELD_CL
#define MPPI_DISTANCE_FIELD_CL

// Replicates the GPU-side is_obstacle_distance_cell inline
inline int is_obstacle_cell(uchar cost, uchar lethal_threshold)
{
  // cost != 255 (NO_INFORMATION)  AND  cost >= lethal_threshold
  return (cost != 255) && (cost >= lethal_threshold);
}

__kernel void build_distance_field(__global const uchar * costmap,
                                   __global float * distance_field,
                                   int size_x,
                                   int size_y,
                                   float resolution,
                                   float cutoff,
                                   uchar lethal_threshold)
{
  const int idx = get_global_id(0);
  const int cell_count = size_x * size_y;
  if (idx >= cell_count) return;

  const uchar center = costmap[idx];

  // Obstacle cell → distance = 0
  if (is_obstacle_cell(center, lethal_threshold)) {
    distance_field[idx] = 0.0f;
    return;
  }

  const int mx = idx % size_x;
  const int my = idx / size_x;
  const int radius_cells = max(1, (int)ceil(cutoff / resolution));
  const float cutoff_cells = cutoff / resolution;
  float best_cells2 = cutoff_cells * cutoff_cells;

  for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
    const int yy = my + dy;
    if (yy < 0 || yy >= size_y) continue;
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
      const int xx = mx + dx;
      if (xx < 0 || xx >= size_x) continue;
      if (!is_obstacle_cell(costmap[yy * size_x + xx], lethal_threshold)) continue;
      const float d2 = (float)(dx * dx + dy * dy);
      best_cells2 = fmin(best_cells2, d2);
    }
  }

  distance_field[idx] = fmin(sqrt(best_cells2) * resolution, cutoff);
}

#endif  // MPPI_DISTANCE_FIELD_CL
