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

#include <nvimgcdcs_module.h>
#include <nvimgcodecs.h>
#include <memory>
#include <string>
#include "iimage_encoder.h"

namespace nvimgcdcs {

class IEncodeState;
class IImage;
class ICodeStream;

class ImageEncoder : public IImageEncoder
{
  public:
    ImageEncoder(const struct nvimgcdcsEncoderDesc* desc, nvimgcdcsEncodeParams_t* params);
    ~ImageEncoder() override;
    std::unique_ptr<IEncodeState> createEncodeState(cudaStream_t cuda_stream) const override;
    void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) override;
    void encode(ICodeStream* code_stream, IImage* image, nvimgcdcsEncodeParams_t* params) override;

  private:
    const struct nvimgcdcsEncoderDesc* encoder_desc_;
    nvimgcdcsEncoder_t encoder_;
};

} // namespace nvimgcdcs