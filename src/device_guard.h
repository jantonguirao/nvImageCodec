/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda.h>

namespace nvimgcdcs {

// /**
//  * Simple RAII device handling:
//  * Switch to new device on construction, back to old
//  * device on destruction
//  */
class DeviceGuard {
 public:
  /// @brief Saves current device id and restores it upon object destruction
  DeviceGuard();

  /// @brief Saves current device id, sets a new one and switches back
  ///        to the original device upon object destruction.
  //         for device id < 0 it is no-op
  explicit DeviceGuard(int new_device);
  ~DeviceGuard();
 private:
  CUcontext old_context_;
};

} // namespace nvimgcdcs


