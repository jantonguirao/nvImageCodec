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
#include <memory>
#include <vector>

namespace nvbmp {

class NvBmpEncoderPlugin
{
  public:
    explicit NvBmpEncoderPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsEncoderDesc_t* getEncoderDesc();

  private:
    nvimgcdcsStatus_t create(nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);
    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvbmp_encoder";
    nvimgcdcsEncoderDesc_t encoder_desc_;
    const nvimgcdcsFrameworkDesc_t* framework_;
};

} // namespace nvbmp
