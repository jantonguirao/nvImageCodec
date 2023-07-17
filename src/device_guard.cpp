/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "device_guard.h"
#include "exception.h"

namespace nvimgcdcs {

bool cuInitChecked()
{
    static CUresult res = cuInit(0);
    return res == CUDA_SUCCESS;
}

DeviceGuard::DeviceGuard() :
  old_context_(NULL) {
 if (!cuInitChecked()){
     throw Exception(INTERNAL_ERROR,
         "Failed to load libcuda.so. "
         "Check your library paths and if the driver is installed correctly.");
 }
  CHECK_CU(cuCtxGetCurrent(&old_context_));
}

DeviceGuard::DeviceGuard(int new_device) :
  old_context_(NULL) {
  if (new_device >= 0) {
     if (!cuInitChecked()) {
         throw Exception(INTERNAL_ERROR,
             "Failed to load libcuda.so. "
             "Check your library paths and if the driver is installed correctly.");
     }
     CHECK_CU(cuCtxGetCurrent(&old_context_));
     CHECK_CUDA(cudaSetDevice(new_device));
  }
}

DeviceGuard::~DeviceGuard() {
  if (old_context_ != NULL) {
    CUresult err = cuCtxSetCurrent(old_context_);
    if (err != CUDA_SUCCESS) {
         std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
         std::terminate();
    }
  }
}

} // namespace nvimgcdcs
