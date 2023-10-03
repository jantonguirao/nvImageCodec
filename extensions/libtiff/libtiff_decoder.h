/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <memory>
#include <vector>
#include "nvimgcodec.h"

namespace libtiff {

class LibtiffDecoderPlugin
{
  public:
    explicit LibtiffDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecDecoderDesc_t* getDecoderDesc();

  private:
    nvimgcodecStatus_t create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "libtiff_decoder";
    nvimgcodecDecoderDesc_t decoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace libtiff
