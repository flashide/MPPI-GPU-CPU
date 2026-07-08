// MppiGpu PIMPL — delegates to MppiSolver (which orchestrates the 3-step
// GPU pipeline via an IMppiBackend).
//
// This file replaces the original monolithic mppi_gpu.cu when the OpenCL
// or CPU backend is selected (MPPI_USE_CUDA=OFF).

#include "opencl_mppi_controller/mppi_gpu.hpp"
#include "opencl_mppi_controller/mppi_solver.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace mppi_controller
{

// Forward-declared factories (defined in backend .cpp files)
extern std::unique_ptr<IMppiBackend> createCpuBackend();
extern std::unique_ptr<IMppiBackend> createOpenCLBackend(
    const std::string & kernel_dir);

namespace
{

std::unique_ptr<IMppiBackend> makeBackend(GpuBackend type,
                                          const std::string & kdir)
{
  switch (type) {
    case GpuBackend::Cpu:    return createCpuBackend();
    case GpuBackend::OpenCL: return createOpenCLBackend(kdir);
    default:
      throw std::runtime_error("Unsupported backend type");
  }
}

GpuBackend resolveBackend(GpuBackend preferred)
{
  const char * env = std::getenv("MPPI_BACKEND");
  if (env) {
    std::string s(env);
    if (s == "cpu" || s == "CPU")    return GpuBackend::Cpu;
    if (s == "opencl" || s == "ocl") return GpuBackend::OpenCL;
  }
  return preferred;
}

}  // namespace

// ---- MppiGpu::Impl --------------------------------------------------------
struct MppiGpu::Impl
{
  std::unique_ptr<MppiSolver> solver;

  explicit Impl(const MppiParams & p,
                GpuBackend preferred = GpuBackend::OpenCL,
                const std::string & kdir = "src/kernels")
  {
    GpuBackend type = resolveBackend(preferred);
    auto backend = makeBackend(type, kdir);
    solver = std::make_unique<MppiSolver>(std::move(backend), p);
  }
};

// ---- MppiGpu public API ---------------------------------------------------
MppiGpu::MppiGpu(const MppiParams & params)
  : impl_(new Impl(params, GpuBackend::OpenCL, "src/kernels"))
{}

MppiGpu::MppiGpu(const MppiParams & params,
                 GpuBackend preferred_backend,
                 const std::string & kernel_dir)
  : impl_(new Impl(params, preferred_backend, kernel_dir))
{}

MppiGpu::~MppiGpu() = default;

void MppiGpu::reset()
{
  if (impl_->solver) impl_->solver->reset();
}

void MppiGpu::setSpeedLimit(float v_max)
{
  if (impl_->solver) impl_->solver->setSpeedLimit(v_max);
}

MppiResult MppiGpu::compute(
  float robot_x, float robot_y, float robot_yaw,
  const uint8_t * costmap, int size_x, int size_y,
  float origin_x, float origin_y, float resolution,
  const float * path_xy, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len)
{
  return impl_->solver->solve(
    robot_x, robot_y, robot_yaw,
    costmap, size_x, size_y,
    origin_x, origin_y, resolution,
    path_xy, path_len,
    goal_x, goal_y, goal_yaw, goal_is_final,
    footprint_xy, footprint_len);
}

MppiResult MppiGpu::computeWithDeviceCostmap(
  float robot_x, float robot_y, float robot_yaw,
  const uint8_t * device_costmap, int size_x, int size_y,
  float origin_x, float origin_y, float resolution,
  const float * path_xy, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len)
{
  // "Device" costmap means already resident; backend handles this internally
  return impl_->solver->solve(
    robot_x, robot_y, robot_yaw,
    device_costmap, size_x, size_y,
    origin_x, origin_y, resolution,
    path_xy, path_len,
    goal_x, goal_y, goal_yaw, goal_is_final,
    footprint_xy, footprint_len);
}

}  // namespace mppi_controller
