# MPPI-GPU-CPU

RK3588 机器狗 MPPI 控制器，OpenCL Mali-G610 GPU 加速 + CPU 回退。

```
Nav2 → MppiGpu → MppiSolver → IMppiBackend
                               ├── OpenCL Backend (Mali-G610)
                               └── CPU Backend (OpenMP)
```

## 构建

```bash
cd ros2_ws
./build_rk3588.sh          # OpenCL + CPU
./build_rk3588.sh cpu      # 仅 CPU
```

### 依赖

| 组件 | 版本 |
|------|------|
| CMake | >= 3.18 |
| ROS2 | Humble |
| OpenCL | ocl-icd-libopencl1 |
| OpenMP | 可选 |

## 验证

```bash
# CPU 先跑通
MPPI_BACKEND=cpu ./install/cuda_mppi_controller/lib/cuda_mppi_controller/mppi_opencl_standalone 2048

# GPU 验证
MPPI_BACKEND=opencl ./install/cuda_mppi_controller/lib/cuda_mppi_controller/mppi_opencl_standalone 2048
```

## Nav2 集成

```yaml
# nav2_params.yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "cuda_mppi_controller::CudaMppiController"
      batch_size: 2048
      time_steps: 56
      model_dt: 0.05
      v_max: 0.5
      w_max: 1.9
      temperature: 0.12
```

## 架构

见 [Mppi-solver分步计算架构.md](Mppi-solver分步计算架构.md)

## License

MIT
