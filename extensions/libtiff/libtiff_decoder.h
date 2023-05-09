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

#include "nvimgcodecs.h"
#include <memory>
#include <vector>

namespace libtiff {

class LibtiffDecoderPlugin
{
  public:
    explicit LibtiffDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsDecoderDesc_t getDecoderDesc();

  private:
    nvimgcdcsStatus_t create(nvimgcdcsDecoder_t* decoder, int device_id);
    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id);

    struct nvimgcdcsDecoderDesc decoder_desc_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace libtiff
