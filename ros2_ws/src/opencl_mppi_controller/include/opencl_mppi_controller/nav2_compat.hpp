#ifndef opencl_mppi_controller__NAV2_COMPAT_HPP_
#define opencl_mppi_controller__NAV2_COMPAT_HPP_

// Small compile-time shims so the plugin builds on Humble (pre-Iron exceptions)
// and newer Nav2 distros without forking the controller implementation.

#if defined(OPENCL_MPPI_CONTROLLER_NAV2_HUMBLE)
#include "nav2_core/exceptions.hpp"
namespace mppi_controller
{
using ControllerInvalidPath = nav2_core::PlannerException;
using ControllerTFError = nav2_core::PlannerException;
using ControllerException = nav2_core::PlannerException;
using NoValidControl = nav2_core::PlannerException;
}
#else
#include "nav2_core/controller_exceptions.hpp"
namespace mppi_controller
{
using ControllerInvalidPath = nav2_core::InvalidPath;
using ControllerTFError = nav2_core::ControllerTFError;
using ControllerException = nav2_core::ControllerException;
using NoValidControl = nav2_core::NoValidControl;
}
#endif

#endif  // opencl_mppi_controller__NAV2_COMPAT_HPP_
