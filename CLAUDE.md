# RK3588 MPPI Controller

基于 OpenCL/CPU 后端的 Nav2 MPPI 控制器。目标平台：RK3588 Mali-G610 GPU，机器狗自主导航。

## 架构

```
Nav2 controller_server → MppiController (不改)
  → MppiGpu (PIMPL) → MppiSolver (4-step 编排)
    → IMppiBackend → CPU Backend (OpenMP) / OpenCL Backend (Mali-G610)
```

## 构建 (RK3588)

```bash
cd ros2_ws
./build_rk3588.sh          # OpenCL + CPU 双后端
./build_rk3588.sh cpu      # 仅 CPU (回退/调试)
```

### 依赖
- CMake >= 3.18
- ROS2 Humble
- OpenCL 运行时 (ocl-icd-libopencl1)
- Eigen3, OpenCV >= 4.5 (通过 Nav2)
- 可选: OpenMP

## 项目结构

```
ros2_ws/src/mppi_controller/
  include/mppi_controller/
    mppi_controller.hpp   # Nav2 Controller 插件 (不改)
    nav2_compat.hpp            # Nav2 Humble 兼容
    mppi_gpu.hpp               # MppiParams / MppiResult / MppiGpu
    mppi_backend.hpp           # IMppiBackend 抽象接口
    mppi_solver.hpp            # 4-step pipeline 编排
  src/
    mppi_controller.cpp   # Nav2 生命周期
    mppi_optimizer.cpp         # PIMPL + 后端工厂
    mppi_solver.cpp            # softmin / warm-start / retreat
    backends/
      cpu_backend.cpp          # OpenMP CPU 参照
      opencl_backend.cpp       # Mali-G610 GPU (image2d + __constant)
    kernels/
      mppi_rollout.cl          # Step 1: 控制采样 + 状态传播
      mppi_cost.cl             # Step 2: 代价评估
      mppi_reduction.cl        # Step 3: 加权控制更新
      mppi_rng.cl              # XOROSHIRO128+ PRNG
      mppi_distance_field.cl   # ESDF 距离场
  config/
    mppi_params.example.yaml
    nav2_loopback_demo*.yaml (×3)
  test/
    mppi_opencl_standalone.cpp
    plugin_load_test.cpp
    parameter_validation_test.cpp
    controller_benchmark.cpp
```

## 后端切换

```bash
MPPI_BACKEND=opencl ./install/.../mppi_opencl_standalone
MPPI_BACKEND=cpu    ./install/.../mppi_opencl_standalone
```

## 每个控制周期的调用链

```
Solver::solve()
  ├─ backend.uploadCostmap()
  ├─ backend.uploadPath()
  ├─ backend.uploadState()
  ├─ for iter = 1..N:
  │    ├─ backend.rollout()        // GPU: K×T 轨迹
  │    ├─ backend.calculateCost()  // GPU: K costs
  │    ├─ backend.downloadCosts()  // D→H (8KB)
  │    ├─ softminWeights()         // CPU: exp(-cost/λ)
  │    └─ backend.reduce()         // GPU: 加权平均 → nominal
  ├─ warm-start: horizon shift
  └─ return MppiResult
```

## 并行模式

- 1 work-item = 1 条采样轨迹 (rollout/cost)
- 1 work-item = 1 个控制维度×时间步 (reduction)
- 控制采样: Gaussian noise around nominal, k==0 = unperturbed
- softmin weights 在 CPU 上计算 (K ≤ 4096)

## 编码约定

- namespace: `mppi_controller`
- IMppiBackend: 纯虚接口，仅 C++，无 GPU 头文件暴露
- Kernel: OpenCL C，`__constant` 参数块，`image2d_t` 代价地图
- MppiSolver: 纯 C++，不依赖 GPU API
- MppiGpu: PIMPL，调用方只需包含 `mppi_gpu.hpp`
