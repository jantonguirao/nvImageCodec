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
#include "nvimgcodecs.h"

namespace libtiff {

class LibtiffDecoderPlugin
{
  public:
    explicit LibtiffDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsDecoderDesc_t getDecoderDesc();

  private:
    nvimgcdcsStatus_t create(
        nvimgcdcsDecoder_t* decoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);
    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);

    struct nvimgcdcsDecoderDesc decoder_desc_;

    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace libtiff
