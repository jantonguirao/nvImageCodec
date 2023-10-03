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

#include <nvimgcodec.h>
#include <memory>
#include <vector>

namespace nvbmp {

class NvBmpEncoderPlugin
{
  public:
    explicit NvBmpEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecEncoderDesc_t* getEncoderDesc();

  private:
    nvimgcodecStatus_t create(nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvbmp_encoder";
    nvimgcodecEncoderDesc_t encoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace nvbmp
