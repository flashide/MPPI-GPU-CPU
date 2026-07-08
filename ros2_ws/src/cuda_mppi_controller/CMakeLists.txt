cmake_minimum_required(VERSION 3.18)
project(cuda_mppi_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ---- GPU backend options -------------------------------------------------
option(MPPI_USE_OPENCL "Build OpenCL backend (Mali-G610, ARM)"  ON)
option(MPPI_USE_CPU    "Build CPU reference backend"            ON)

set(MPPI_BACKEND_SOURCES
  src/mppi_optimizer.cpp
  src/mppi_solver.cpp
)
set(MPPI_BACKEND_LIBS "")

if(MPPI_USE_CPU)
  list(APPEND MPPI_BACKEND_SOURCES src/backends/cpu_backend.cpp)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

if(MPPI_USE_OPENCL)
  list(APPEND MPPI_BACKEND_SOURCES src/backends/opencl_backend.cpp)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    list(APPEND MPPI_BACKEND_LIBS OpenCL::OpenCL)
    add_compile_definitions(MPPI_HAS_OPENCL=1)
    message(STATUS "MPPI: OpenCL backend enabled (${OpenCL_INCLUDE_DIRS})")
  else()
    find_library(OPENCL_LIB OpenCL PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
    find_path(OPENCL_INC CL/cl.h PATHS /usr/include)
    if(OPENCL_LIB AND OPENCL_INC)
      list(APPEND MPPI_BACKEND_LIBS ${OPENCL_LIB})
      include_directories(${OPENCL_INC})
      add_compile_definitions(MPPI_HAS_OPENCL=1)
      message(STATUS "MPPI: OpenCL backend enabled (manual: ${OPENCL_LIB})")
    else()
      message(WARNING "MPPI: OpenCL not found — OpenCL backend will not link")
      add_compile_definitions(MPPI_HAS_OPENCL=0)
    endif()
  endif()
endif()

set(ROS_DEPS
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  geometry_msgs
  pluginlib
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_library(${PROJECT_NAME} SHARED
  src/cuda_mppi_controller.cpp
  ${MPPI_BACKEND_SOURCES}
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} ${ROS_DEPS})
target_link_libraries(${PROJECT_NAME} ${MPPI_BACKEND_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(DEFINED ENV{ROS_DISTRO} AND "$ENV{ROS_DISTRO}" STREQUAL "humble")
  target_compile_definitions(${PROJECT_NAME} PRIVATE CUDAMPPI_NAV2_HUMBLE=1)
endif()

# ---- test executables ----------------------------------------------------
add_executable(mppi_opencl_standalone test/mppi_opencl_standalone.cpp)
target_link_libraries(mppi_opencl_standalone ${PROJECT_NAME})

add_executable(plugin_load_test test/plugin_load_test.cpp)
ament_target_dependencies(plugin_load_test ${ROS_DEPS})

add_executable(parameter_validation_test test/parameter_validation_test.cpp)
target_link_libraries(parameter_validation_test ${PROJECT_NAME})
ament_target_dependencies(parameter_validation_test ${ROS_DEPS})

add_executable(controller_benchmark test/controller_benchmark.cpp)
ament_target_dependencies(controller_benchmark ${ROS_DEPS})

# ---- install -------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS plugin_load_test parameter_validation_test controller_benchmark
        mppi_opencl_standalone
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(FILES cuda_mppi_plugin.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(MPPI_USE_OPENCL)
  install(DIRECTORY src/kernels/ DESTINATION lib/${PROJECT_NAME}/kernels
          FILES_MATCHING PATTERN "*.cl")
endif()

pluginlib_export_plugin_description_file(nav2_core cuda_mppi_plugin.xml)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${ROS_DEPS})
ament_package()
