// OpenCL 3.0 backend — 3-step MPPI pipeline for Mali-G610 (RK3588).
//
// Pipeline per iteration:
//   rollout()       → sample controls + propagate states → traj buffers
//   calculateCost() → evaluate costs from traj buffers   → cost buffer
//   reduce()        → weighted-average control update    → nominal + optimal
//
// Mali-G610 optimisations:
//   - image2d_t for costmap (hardware texture cache)
//   - __constant param blocks (Valhall 64 KB constant cache)
//   - XOROSHIRO128+ RNG (low registers, no lookup table)
//   - fp32 throughout (Mali fp32 is 1:1 rate)

#include "opencl_mppi_controller/mppi_backend.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace mppi_controller
{
namespace
{

// ---- helpers --------------------------------------------------------------
#define CL_CHECK(expr)                                      \
  do { cl_int e__ = (expr);                                 \
    if (e__ != CL_SUCCESS) {                                \
      throw std::runtime_error(std::string("OpenCL err ") + \
        std::to_string(e__) + " at " __FILE__ ":" +         \
        std::to_string(__LINE__));                          \
    }                                                       \
  } while (0)

std::string readFile(const std::string & path)
{
  std::ifstream f(path);
  if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
  std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

static size_t roundUp(size_t n, size_t m) { return ((n+m-1)/m)*m; }

// Work-group sizes for Mali-G610 (Valhall, MAX_WG_SIZE=256)
constexpr size_t kWgRollout   = 256;
constexpr size_t kWgCost      = 256;
constexpr size_t kWgReduce    = 128;
constexpr size_t kWgRng       = 256;
constexpr size_t kWgDistField = 64;

// ---- constant blocks (must match kernel structs exactly) ------------------
#pragma pack(push, 16)
struct alignas(16) CLRolloutParams {
  int K, T; float dt; int motion_model;
  float v_max, v_min, vy_max, w_max, min_turning_r;
  float v_std, vy_std, w_std;
  float start_x, start_y, start_yaw;
};

struct alignas(16) CLCostParams {
  int K, T; float dt; int motion_model;
  float v_max, v_min;
  float goal_w, goal_yaw_w, path_w, follow_w, path_angle_w;
  float curvature_speed_w, curvature_speed_min, costmap_w;
  float distance_field_w, distance_field_cutoff;
  float smooth_w, backward_w, speed_w, angular_w;
  float collision_cost, yaw_activation_dist;
  uchar lethal_threshold; uchar _pad[3];
  int follow_offset;
  int size_x, size_y;
  float origin_x, origin_y, resolution;
  int path_len;
  float goal_x, goal_y, goal_yaw;
  int goal_is_final;
};
#pragma pack(pop)

// ---- OpenCL backend class -------------------------------------------------
class OpenCLBackend : public IMppiBackend
{
public:
  GpuBackend type() const override { return GpuBackend::OpenCL; }
  const char * name() const override { return "OpenCL"; }

  explicit OpenCLBackend(const std::string & kdir = "src/kernels")
    : kernel_dir_(kdir) {}

  // ==================================================================
  //  Lifecycle
  // ==================================================================
  bool initialize(const DeviceParams & dp, uint64_t rng_seed) override
  {
    dp_ = dp; K_ = dp.K; T_ = dp.T;
    initPlatform(); initDevice(); compileKernels(); allocateBuffers();
    uploadRolloutParams(); uploadCostParams(); initRNG(rng_seed);
    return true;
  }

  void shutdown() override
  {
    releaseBuf(costmap_img_); releaseBuf(costmap_buf_);
    releaseBuf(path_buf_); releaseBuf(nominal_buf_);
    releaseBuf(perturbed_buf_);
    releaseBuf(traj_x_); releaseBuf(traj_y_); releaseBuf(traj_yaw_);
    releaseBuf(costs_buf_); releaseBuf(weights_buf_);
    releaseBuf(rng_buf_); releaseBuf(optimal_buf_);
    releaseBuf(distance_field_buf_);

    if (krn_init_rng_)    clReleaseKernel(krn_init_rng_);
    if (krn_rollout_)     clReleaseKernel(krn_rollout_);
    if (krn_cost_img_)    clReleaseKernel(krn_cost_img_);
    if (krn_cost_buf_)    clReleaseKernel(krn_cost_buf_);
    if (krn_reduce_)      clReleaseKernel(krn_reduce_);
    if (krn_distfield_)   clReleaseKernel(krn_distfield_);

    if (prog_rng_)        clReleaseProgram(prog_rng_);
    if (prog_rollout_)    clReleaseProgram(prog_rollout_);
    if (prog_cost_)       clReleaseProgram(prog_cost_);
    if (prog_reduce_)     clReleaseProgram(prog_reduce_);
    if (prog_distfield_)  clReleaseProgram(prog_distfield_);

    if (sampler_)         clReleaseSampler(sampler_);
    if (cmd_queue_)       clReleaseCommandQueue(cmd_queue_);
    if (context_)         clReleaseContext(context_);
  }

  void reset() override
  {
    nominal_host_.assign(T_ * kCtrlDim, 0.0f);
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, nominal_buf_, CL_TRUE, 0,
                                   nominal_host_.size() * sizeof(float),
                                   nominal_host_.data(), 0, nullptr, nullptr));
  }

  // ==================================================================
  //  Data upload
  // ==================================================================
  void uploadCostmap(const uint8_t * data, int w, int h) override
  {
    dp_.size_x = w; dp_.size_y = h;
    if (!data) return;
    const size_t bytes = static_cast<size_t>(w) * h;
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, costmap_buf_, CL_TRUE, 0,
                                   bytes, data, 0, nullptr, nullptr));
    if (use_image2d_ && costmap_img_) {
      const size_t org[3] = {0,0,0};
      const size_t reg[3] = {static_cast<size_t>(w), static_cast<size_t>(h), 1};
      CL_CHECK(clEnqueueWriteImage(cmd_queue_, costmap_img_, CL_TRUE,
                                    org, reg, 0, 0, data, 0, nullptr, nullptr));
    }
    need_params_upload_ = true;
    need_cost_params_upload_ = true;
  }

  void uploadPath(const float * path_xy, int n) override
  {
    dp_.path_len = n;
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, path_buf_, CL_TRUE, 0,
                                   n * 2 * sizeof(float), path_xy,
                                   0, nullptr, nullptr));
    need_cost_params_upload_ = true;
  }

  void uploadNominal(const float * nominal) override
  {
    nominal_host_.assign(nominal, nominal + T_ * kCtrlDim);
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, nominal_buf_, CL_TRUE, 0,
                                   nominal_host_.size() * sizeof(float),
                                   nominal_host_.data(), 0, nullptr, nullptr));
  }

  void uploadState(float x, float y, float yaw,
                   float gx, float gy, float gyaw, bool goal_final) override
  {
    dp_.start_x = x;  dp_.start_y = y;  dp_.start_yaw = yaw;
    dp_.goal_x = gx;  dp_.goal_y = gy;  dp_.goal_yaw = gyaw;
    dp_.goal_is_final = goal_final ? 1 : 0;
    need_params_upload_ = true;
    need_cost_params_upload_ = true;
  }

  // ==================================================================
  //  Pipeline steps
  // ==================================================================
  void rollout() override
  {
    if (need_params_upload_) { uploadRolloutParams(); need_params_upload_ = false; }

    // kernel args: nominal, perturbed, traj_x, traj_y, traj_yaw, rng, params
    cl_int e = 0;
    e  = clSetKernelArg(krn_rollout_, 0, sizeof(cl_mem), &nominal_buf_);
    e |= clSetKernelArg(krn_rollout_, 1, sizeof(cl_mem), &perturbed_buf_);
    e |= clSetKernelArg(krn_rollout_, 2, sizeof(cl_mem), &traj_x_);
    e |= clSetKernelArg(krn_rollout_, 3, sizeof(cl_mem), &traj_y_);
    e |= clSetKernelArg(krn_rollout_, 4, sizeof(cl_mem), &traj_yaw_);
    e |= clSetKernelArg(krn_rollout_, 5, sizeof(cl_mem), &rng_buf_);
    e |= clSetKernelArg(krn_rollout_, 6, sizeof(cl_mem), &rollout_params_buf_);
    CL_CHECK(e);

    size_t g = roundUp(static_cast<size_t>(K_), kWgRollout);
    CL_CHECK(clEnqueueNDRangeKernel(cmd_queue_, krn_rollout_, 1, nullptr,
                                     &g, &kWgRollout, 0, nullptr, nullptr));
  }

  void calculateCost() override
  {
    if (need_cost_params_upload_) { uploadCostParams(); need_cost_params_upload_ = false; }

    cl_int e = 0;
    if (use_image2d_) {
      e  = clSetKernelArg(krn_cost_img_, 0, sizeof(cl_mem), &costmap_img_);
      e |= clSetKernelArg(krn_cost_img_, 1, sizeof(cl_sampler), &sampler_);
      e |= clSetKernelArg(krn_cost_img_, 2, sizeof(cl_mem), &distance_field_buf_);
      e |= clSetKernelArg(krn_cost_img_, 3, sizeof(cl_mem), &path_buf_);
      e |= clSetKernelArg(krn_cost_img_, 4, sizeof(cl_mem), &traj_x_);
      e |= clSetKernelArg(krn_cost_img_, 5, sizeof(cl_mem), &traj_y_);
      e |= clSetKernelArg(krn_cost_img_, 6, sizeof(cl_mem), &traj_yaw_);
      e |= clSetKernelArg(krn_cost_img_, 7, sizeof(cl_mem), &perturbed_buf_);
      e |= clSetKernelArg(krn_cost_img_, 8, sizeof(cl_mem), &costs_buf_);
      e |= clSetKernelArg(krn_cost_img_, 9, sizeof(cl_mem), &cost_params_buf_);
    } else {
      e  = clSetKernelArg(krn_cost_buf_, 0, sizeof(cl_mem), &costmap_buf_);
      e |= clSetKernelArg(krn_cost_buf_, 1, sizeof(cl_mem), &distance_field_buf_);
      e |= clSetKernelArg(krn_cost_buf_, 2, sizeof(cl_mem), &path_buf_);
      e |= clSetKernelArg(krn_cost_buf_, 3, sizeof(cl_mem), &traj_x_);
      e |= clSetKernelArg(krn_cost_buf_, 4, sizeof(cl_mem), &traj_y_);
      e |= clSetKernelArg(krn_cost_buf_, 5, sizeof(cl_mem), &traj_yaw_);
      e |= clSetKernelArg(krn_cost_buf_, 6, sizeof(cl_mem), &perturbed_buf_);
      e |= clSetKernelArg(krn_cost_buf_, 7, sizeof(cl_mem), &costs_buf_);
      e |= clSetKernelArg(krn_cost_buf_, 8, sizeof(cl_mem), &cost_params_buf_);
    }
    CL_CHECK(e);

    size_t g = roundUp(static_cast<size_t>(K_), kWgCost);
    cl_kernel k = use_image2d_ ? krn_cost_img_ : krn_cost_buf_;
    CL_CHECK(clEnqueueNDRangeKernel(cmd_queue_, k, 1, nullptr,
                                     &g, &kWgCost, 0, nullptr, nullptr));
  }

  void reduce(const float * weights_host,
              float * opt_v, float * opt_vy, float * opt_w) override
  {
    // Upload softmin weights from host
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, weights_buf_, CL_TRUE, 0,
                                   K_ * sizeof(float), weights_host,
                                   0, nullptr, nullptr));

    cl_int e = 0;
    e  = clSetKernelArg(krn_reduce_, 0, sizeof(cl_mem), &perturbed_buf_);
    e |= clSetKernelArg(krn_reduce_, 1, sizeof(cl_mem), &weights_buf_);
    e |= clSetKernelArg(krn_reduce_, 2, sizeof(cl_mem), &nominal_buf_);
    e |= clSetKernelArg(krn_reduce_, 3, sizeof(cl_mem), &optimal_buf_);
    e |= clSetKernelArg(krn_reduce_, 4, sizeof(int), &K_);
    e |= clSetKernelArg(krn_reduce_, 5, sizeof(int), &T_);
    CL_CHECK(e);

    size_t g = roundUp(static_cast<size_t>(T_ * 3), kWgReduce);
    CL_CHECK(clEnqueueNDRangeKernel(cmd_queue_, krn_reduce_, 1, nullptr,
                                     &g, &kWgReduce, 0, nullptr, nullptr));

    // Read optimal control
    float opt[3] = {};
    CL_CHECK(clEnqueueReadBuffer(cmd_queue_, optimal_buf_, CL_TRUE, 0,
                                  3 * sizeof(float), opt,
                                  0, nullptr, nullptr));
    *opt_v  = opt[0];
    *opt_vy = opt[1];
    *opt_w  = opt[2];
  }

  // ==================================================================
  //  Data download
  // ==================================================================
  void downloadCosts(float * costs_host) override
  {
    CL_CHECK(clEnqueueReadBuffer(cmd_queue_, costs_buf_, CL_TRUE, 0,
                                  K_ * sizeof(float), costs_host,
                                  0, nullptr, nullptr));
  }

  void downloadNominal(float * nominal_host) override
  {
    CL_CHECK(clEnqueueReadBuffer(cmd_queue_, nominal_buf_, CL_TRUE, 0,
                                  T_ * kCtrlDim * sizeof(float),
                                  nominal_host, 0, nullptr, nullptr));
  }

  // ==================================================================
  //  Optional
  // ==================================================================
  void buildDistanceField() override
  {
    if (dp_.distance_field_w <= 0.0f || dp_.distance_field_cutoff <= 1.0e-6f)
      return;
    int nc = dp_.size_x * dp_.size_y;
    if (nc <= 0) return;
    ensureDistFieldBuf(nc);

    cl_int e = 0;
    e  = clSetKernelArg(krn_distfield_, 0, sizeof(cl_mem), &costmap_buf_);
    e |= clSetKernelArg(krn_distfield_, 1, sizeof(cl_mem), &distance_field_buf_);
    e |= clSetKernelArg(krn_distfield_, 2, sizeof(int), &dp_.size_x);
    e |= clSetKernelArg(krn_distfield_, 3, sizeof(int), &dp_.size_y);
    e |= clSetKernelArg(krn_distfield_, 4, sizeof(float), &dp_.resolution);
    e |= clSetKernelArg(krn_distfield_, 5, sizeof(float), &dp_.distance_field_cutoff);
    uchar lt = dp_.lethal_threshold;
    e |= clSetKernelArg(krn_distfield_, 6, sizeof(uchar), &lt);
    CL_CHECK(e);

    size_t g = roundUp(static_cast<size_t>(nc), kWgDistField);
    CL_CHECK(clEnqueueNDRangeKernel(cmd_queue_, krn_distfield_, 1, nullptr,
                                     &g, &kWgDistField, 0, nullptr, nullptr));
  }

  void updateVMax(float v_max) override
  {
    dp_.v_max = v_max;
    need_params_upload_ = true;
    need_cost_params_upload_ = true;
  }

private:
  // ---- OpenCL init --------------------------------------------------------
  void initPlatform()
  {
    cl_uint n = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &n));
    std::vector<cl_platform_id> plats(n);
    CL_CHECK(clGetPlatformIDs(n, plats.data(), nullptr));

    int sel = 0;
    for (cl_uint i = 0; i < n; ++i) {
      char nm[256];
      clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(nm), nm, nullptr);
      if (std::string(nm).find("Mali") != std::string::npos ||
          std::string(nm).find("ARM")  != std::string::npos) { sel = static_cast<int>(i); break; }
    }
    platform_ = plats[sel];
    char pn[256];
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(pn), pn, nullptr);
    std::fprintf(stderr, "[OpenCL] Platform: %s\n", pn);
  }

  void initDevice()
  {
    cl_uint n = 0;
    CL_CHECK(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &n));
    if (n == 0) CL_CHECK(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &n));
    std::vector<cl_device_id> devs(n);
    CL_CHECK(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, n, devs.data(), nullptr));
    device_ = devs[0];

    char dn[256];
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
    std::fprintf(stderr, "[OpenCL] Device: %s\n", dn);

    cl_bool img = CL_FALSE;
    clGetDeviceInfo(device_, CL_DEVICE_IMAGE_SUPPORT, sizeof(img), &img, nullptr);
    use_image2d_ = (img == CL_TRUE);

    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);
    cmd_queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    std::fprintf(stderr, "[OpenCL] image2d: %s\n", use_image2d_ ? "yes" : "no");
  }

  void compileKernels()
  {
    const std::string rng_src  = readFile(kernel_dir_ + "/mppi_rng.cl");
    const std::string rol_src  = "#include \"mppi_rng.cl\"\n" +
                                  readFile(kernel_dir_ + "/mppi_rollout.cl");
    const std::string cost_src = readFile(kernel_dir_ + "/mppi_cost.cl");
    const std::string red_src  = readFile(kernel_dir_ + "/mppi_reduction.cl");
    const std::string df_src   = readFile(kernel_dir_ + "/mppi_distance_field.cl");

    const char * opts = "-cl-mad-enable -cl-fast-relaxed-math "
                        "-cl-single-precision-constant";

    auto build = [&](cl_program & pg, const char * src, const char * nm) {
      cl_int err;
      pg = clCreateProgramWithSource(context_, 1, &src, nullptr, &err);
      CL_CHECK(err);
      cl_int be = clBuildProgram(pg, 1, &device_, opts, nullptr, nullptr);
      if (be != CL_SUCCESS) {
        size_t sz = 0;
        clGetProgramBuildInfo(pg, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
        std::vector<char> log(sz+1);
        clGetProgramBuildInfo(pg, device_, CL_PROGRAM_BUILD_LOG, sz, log.data(), nullptr);
        std::fprintf(stderr, "[OpenCL] Build log (%s):\n%s\n", nm, log.data());
        throw std::runtime_error(std::string("Build failed: ") + nm);
      }
    };

    build(prog_rng_,       rng_src.c_str(),  "mppi_rng");
    build(prog_rollout_,   rol_src.c_str(),  "mppi_rollout");
    build(prog_cost_,      cost_src.c_str(), "mppi_cost");
    build(prog_reduce_,    red_src.c_str(),  "mppi_reduction");
    build(prog_distfield_, df_src.c_str(),   "mppi_distance_field");

    cl_int err;
    krn_init_rng_   = clCreateKernel(prog_rng_,       "init_rng", &err); CL_CHECK(err);
    krn_rollout_    = clCreateKernel(prog_rollout_,   "rollout", &err); CL_CHECK(err);
    krn_cost_img_   = clCreateKernel(prog_cost_,      "calculate_cost_img", &err); CL_CHECK(err);
    krn_cost_buf_   = clCreateKernel(prog_cost_,      "calculate_cost_buf", &err); CL_CHECK(err);
    krn_reduce_     = clCreateKernel(prog_reduce_,    "reduce_controls", &err); CL_CHECK(err);
    krn_distfield_  = clCreateKernel(prog_distfield_, "build_distance_field", &err); CL_CHECK(err);

    if (use_image2d_) {
      sampler_ = clCreateSampler(context_, CL_FALSE,
                                  CL_ADDRESS_CLAMP_TO_EDGE,
                                  CL_FILTER_NEAREST, &err);
      CL_CHECK(err);
    }

    std::fprintf(stderr, "[OpenCL] Kernels compiled OK\n");
  }

  // ---- buffers ------------------------------------------------------------
  void allocateBuffers()
  {
    cl_int err;
    const size_t Kb  = static_cast<size_t>(K_);
    const size_t Tb  = static_cast<size_t>(T_);

    // RNG
    rng_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                               Kb * sizeof(uint64_t) * 2, nullptr, &err);
    CL_CHECK(err);

    // Nominal + perturbed
    nominal_host_.assign(Tb * kCtrlDim, 0.0f);
    nominal_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                   Tb * kCtrlDim * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    perturbed_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                     Kb * Tb * kCtrlDim * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // Trajectory buffers (intermediate: [K * T] each)
    const size_t traj_bytes = Kb * Tb * sizeof(float);
    traj_x_   = clCreateBuffer(context_, CL_MEM_READ_WRITE, traj_bytes, nullptr, &err); CL_CHECK(err);
    traj_y_   = clCreateBuffer(context_, CL_MEM_READ_WRITE, traj_bytes, nullptr, &err); CL_CHECK(err);
    traj_yaw_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, traj_bytes, nullptr, &err); CL_CHECK(err);

    // Costs + Weights
    costs_buf_   = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                   Kb * sizeof(float), nullptr, &err); CL_CHECK(err);
    weights_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                   Kb * sizeof(float), nullptr, &err); CL_CHECK(err);

    // Optimal control output [3]
    optimal_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                   3 * sizeof(float), nullptr, &err); CL_CHECK(err);

    // Path
    path_buf_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                kMaxPathPoints * 2 * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // Costmap (raw buffer always; optional image2d)
    const size_t max_cells = 1024 * 1024;
    costmap_buf_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, max_cells, nullptr, &err);
    CL_CHECK(err);
    if (use_image2d_) {
      cl_image_format fmt = {CL_R, CL_UNSIGNED_INT8};
      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = 1024; desc.image_height = 1024;
      costmap_img_ = clCreateImage(context_,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    &fmt, &desc, nullptr, &err);
      CL_CHECK(err);
    }

    // Constant param blocks
    rollout_params_buf_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                          sizeof(CLRolloutParams), nullptr, &err);
    CL_CHECK(err);
    cost_params_buf_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                       sizeof(CLCostParams), nullptr, &err);
    CL_CHECK(err);

    // Minimal distance field (to avoid NULL ptr)
    distance_field_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                          sizeof(float), nullptr, &err);
    CL_CHECK(err);

    std::fprintf(stderr, "[OpenCL] Buffers ready (K=%d, T=%d)\n", K_, T_);
  }

  void uploadRolloutParams()
  {
    CLRolloutParams rp = {};
    rp.K = K_; rp.T = T_; rp.dt = dp_.dt;
    rp.motion_model = dp_.motion_model;
    rp.v_max = dp_.v_max; rp.v_min = dp_.v_min;
    rp.vy_max = dp_.vy_max; rp.w_max = dp_.w_max;
    rp.min_turning_r = dp_.min_turning_r;
    rp.v_std = dp_.v_std; rp.vy_std = dp_.vy_std; rp.w_std = dp_.w_std;
    rp.start_x = dp_.start_x; rp.start_y = dp_.start_y; rp.start_yaw = dp_.start_yaw;
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, rollout_params_buf_, CL_TRUE, 0,
                                   sizeof(rp), &rp, 0, nullptr, nullptr));
  }

  void uploadCostParams()
  {
    CLCostParams cp = {};
    cp.K = K_; cp.T = T_; cp.dt = dp_.dt;
    cp.motion_model = dp_.motion_model;
    cp.v_max = dp_.v_max; cp.v_min = dp_.v_min;
    cp.goal_w = dp_.goal_w; cp.goal_yaw_w = dp_.goal_yaw_w;
    cp.path_w = dp_.path_w; cp.follow_w = dp_.follow_w;
    cp.path_angle_w = dp_.path_angle_w;
    cp.curvature_speed_w = dp_.curvature_speed_w;
    cp.curvature_speed_min = dp_.curvature_speed_min;
    cp.costmap_w = dp_.costmap_w;
    cp.distance_field_w = dp_.distance_field_w;
    cp.distance_field_cutoff = dp_.distance_field_cutoff;
    cp.smooth_w = dp_.smooth_w; cp.backward_w = dp_.backward_w;
    cp.speed_w = dp_.speed_w; cp.angular_w = dp_.angular_w;
    cp.collision_cost = dp_.collision_cost;
    cp.yaw_activation_dist = dp_.yaw_activation_dist;
    cp.lethal_threshold = dp_.lethal_threshold;
    cp.follow_offset = dp_.follow_offset;
    cp.size_x = dp_.size_x; cp.size_y = dp_.size_y;
    cp.origin_x = dp_.origin_x; cp.origin_y = dp_.origin_y;
    cp.resolution = dp_.resolution;
    cp.path_len = dp_.path_len;
    cp.goal_x = dp_.goal_x; cp.goal_y = dp_.goal_y;
    cp.goal_yaw = dp_.goal_yaw;
    cp.goal_is_final = dp_.goal_is_final;
    CL_CHECK(clEnqueueWriteBuffer(cmd_queue_, cost_params_buf_, CL_TRUE, 0,
                                   sizeof(cp), &cp, 0, nullptr, nullptr));
  }

  void initRNG(uint64_t seed)
  {
    cl_int e = 0;
    e  = clSetKernelArg(krn_init_rng_, 0, sizeof(cl_mem), &rng_buf_);
    cl_ulong s = static_cast<cl_ulong>(seed);
    e |= clSetKernelArg(krn_init_rng_, 1, sizeof(cl_ulong), &s);
    e |= clSetKernelArg(krn_init_rng_, 2, sizeof(int), &K_);
    CL_CHECK(e);
    size_t g = roundUp(static_cast<size_t>(K_), kWgRng);
    CL_CHECK(clEnqueueNDRangeKernel(cmd_queue_, krn_init_rng_, 1, nullptr,
                                     &g, &kWgRng, 0, nullptr, nullptr));
  }

  void ensureDistFieldBuf(int cells)
  {
    if (distance_field_buf_) {
      size_t cur = 0;
      clGetMemObjectInfo(distance_field_buf_, CL_MEM_SIZE, sizeof(cur), &cur, nullptr);
      if (cur >= static_cast<size_t>(cells) * sizeof(float)) return;
      clReleaseMemObject(distance_field_buf_);
    }
    cl_int err;
    distance_field_buf_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                          static_cast<size_t>(cells) * sizeof(float),
                                          nullptr, &err);
    CL_CHECK(err);
  }

  static void releaseBuf(cl_mem & b) { if (b) { clReleaseMemObject(b); b = nullptr; } }

  // ---- state --------------------------------------------------------------
  std::string kernel_dir_;
  DeviceParams dp_;
  int K_ = 0, T_ = 0;
  bool use_image2d_ = true;
  bool need_params_upload_ = true;
  bool need_cost_params_upload_ = true;

  // OpenCL objects
  cl_platform_id   platform_ = nullptr;
  cl_device_id     device_   = nullptr;
  cl_context       context_  = nullptr;
  cl_command_queue cmd_queue_ = nullptr;
  cl_sampler       sampler_  = nullptr;

  // programs
  cl_program prog_rng_ = nullptr, prog_rollout_ = nullptr;
  cl_program prog_cost_ = nullptr, prog_reduce_ = nullptr;
  cl_program prog_distfield_ = nullptr;

  // kernels
  cl_kernel krn_init_rng_ = nullptr, krn_rollout_ = nullptr;
  cl_kernel krn_cost_img_ = nullptr, krn_cost_buf_ = nullptr;
  cl_kernel krn_reduce_ = nullptr, krn_distfield_ = nullptr;

  // buffers
  cl_mem rng_buf_ = nullptr, nominal_buf_ = nullptr, perturbed_buf_ = nullptr;
  cl_mem traj_x_ = nullptr, traj_y_ = nullptr, traj_yaw_ = nullptr;
  cl_mem costs_buf_ = nullptr, weights_buf_ = nullptr, optimal_buf_ = nullptr;
  cl_mem path_buf_ = nullptr;
  cl_mem costmap_buf_ = nullptr, costmap_img_ = nullptr;
  cl_mem distance_field_buf_ = nullptr;
  cl_mem rollout_params_buf_ = nullptr, cost_params_buf_ = nullptr;

  std::vector<float> nominal_host_;
};

}  // namespace

// ---- factory --------------------------------------------------------------
std::unique_ptr<IMppiBackend> createOpenCLBackend(const std::string & kdir)
{
  return std::make_unique<OpenCLBackend>(kdir);
}

}  // namespace mppi_controller
