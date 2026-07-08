#!/bin/bash
# Build script for cuda_mppi_controller with OpenCL backend on RK3588.
#
# Prerequisites:
#   sudo apt install -y opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev
#   # Verify Mali GPU is visible:
#   clinfo | grep -i mali
#
# Usage:
#   cd ros2_ws
#   ./build_rk3588.sh          # Release build
#   ./build_rk3588.sh debug    # Debug build (with profiling)
#   ./build_rk3588.sh cpu      # CPU-only fallback

set -euo pipefail

BUILD_TYPE="${1:-release}"
BACKEND_MODE="${2:-opencl}"

# Normalise build type
case "${BUILD_TYPE,,}" in
  release|rel)
    BUILD_TYPE=Release
    ;;
  debug|dbg)
    BUILD_TYPE=Debug
    ;;
  cpu)
    BUILD_TYPE=Release
    BACKEND_MODE=cpu
    ;;
esac

echo "=== Building cuda_mppi_controller for RK3588 ==="
echo "    Build type:  ${BUILD_TYPE}"
echo "    Backend:     ${BACKEND_MODE}"

# Detect OpenCL on RK3588 (Mali GPU userspace driver)
OPENCL_LIB=""
OPENCL_INC=""
if [ "${BACKEND_MODE}" = "opencl" ]; then
  # Common locations on RK3588 Ubuntu 22.04
  for libpath in /usr/lib/aarch64-linux-gnu/libOpenCL.so \
                 /usr/lib/libOpenCL.so; do
    if [ -f "${libpath}" ]; then
      OPENCL_LIB="${libpath}"
      break
    fi
  done
  for incpath in /usr/include/CL/cl.h \
                 /usr/local/include/CL/cl.h; do
    if [ -f "${incpath}" ]; then
      OPENCL_INC="$(dirname "$(dirname "${incpath}")")"
      break
    fi
  done

  if [ -z "${OPENCL_LIB}" ]; then
    echo "WARNING: libOpenCL.so not found. Trying to locate..."
    OPENCL_LIB=$(find /usr -name "libOpenCL.so" 2>/dev/null | head -1)
  fi

  if [ -n "${OPENCL_LIB}" ]; then
    echo "    OpenCL lib:  ${OPENCL_LIB}"
  else
    echo "ERROR: OpenCL library not found."
    echo "Install with: sudo apt install -y ocl-icd-libopencl1"
    exit 1
  fi
fi

# Configure
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

case "${BACKEND_MODE}" in
  opencl)
    CMAKE_ARGS+=(-DMPPI_USE_OPENCL=ON -DMPPI_USE_CPU=ON)
    if [ -n "${OPENCL_LIB}" ]; then
      CMAKE_ARGS+=(-DOPENCL_LIB="${OPENCL_LIB}")
    fi
    if [ -n "${OPENCL_INC}" ]; then
      CMAKE_ARGS+=(-DOPENCL_INC="${OPENCL_INC}")
    fi
    ;;
  cpu)
    CMAKE_ARGS+=(-DMPPI_USE_OPENCL=OFF -DMPPI_USE_CPU=ON)
    ;;
esac

# Build
cd "$(dirname "$0")"
colcon build \
  --packages-select cuda_mppi_controller \
  --cmake-args "${CMAKE_ARGS[@]}" \
  --event-handlers console_direct+

echo ""
echo "=== Build complete ==="
echo ""
echo "Run standalone validation:"
echo "  source install/setup.bash"
echo "  MPPI_BACKEND=${BACKEND_MODE} ./install/cuda_mppi_controller/lib/cuda_mppi_controller/mppi_opencl_standalone"
echo ""
echo "Run with ROS2 (add to nav2_params.yaml):"
echo "  FollowPath:"
echo "    plugin: \"cuda_mppi_controller::CudaMppiController\""
echo "    gpu_backend: \"${BACKEND_MODE}\""
echo "    batch_size: 2048"
echo "    # ... (see config/cuda_mppi_params.example.yaml)"
