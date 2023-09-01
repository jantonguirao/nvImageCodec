/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>

#include <dlpack/dlpack.h>

#include <pybind11/buffer_info.h>

namespace nvimgcdcs {

namespace py = pybind11;

class DLPackTensor final
{
  public:
    DLPackTensor() noexcept;
    explicit DLPackTensor(const DLTensor& tensor);
    explicit DLPackTensor(DLManagedTensor&& tensor);
    explicit DLPackTensor(const nvimgcdcsImageInfo_t& image_info);
    DLPackTensor(DLPackTensor&& that) noexcept;
    ~DLPackTensor();

    DLPackTensor& operator=(DLPackTensor&& that) noexcept;

    const DLTensor* operator->() const;
    DLTensor* operator->();

    const DLTensor& operator*() const;
    DLTensor& operator*();

    void getImageInfo(nvimgcdcsImageInfo_t* image_info);
  private:
    DLManagedTensor dl_managed_tensor_;
};

bool is_cuda_accessible(DLDeviceType devType);

} // namespace nvimgcdcs
