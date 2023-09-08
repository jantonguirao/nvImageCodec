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
#include <pybind11/pybind11.h>

namespace nvimgcdcs {

namespace py = pybind11;

class DLPackTensor final
{
  public:
    DLPackTensor() noexcept;
    DLPackTensor(DLPackTensor&& that) noexcept;

    explicit DLPackTensor(DLManagedTensor* dl_managed_tensor);
    explicit DLPackTensor(const nvimgcdcsImageInfo_t& image_info, std::shared_ptr<unsigned char> image_buffer);

    ~DLPackTensor();

    const DLTensor* operator->() const;
    DLTensor* operator->();

    const DLTensor& operator*() const;
    DLTensor& operator*();

    void getImageInfo(nvimgcdcsImageInfo_t* image_info);
    py::capsule getPyCapsule();

  private:
    DLManagedTensor internal_dl_managed_tensor_;
    DLManagedTensor* dl_managed_tensor_ptr_;
    std::shared_ptr<unsigned char> image_buffer_;
};

bool is_cuda_accessible(DLDeviceType devType);

} // namespace nvimgcdcs
