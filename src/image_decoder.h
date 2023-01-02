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

#include <nvimgcodecs.h>
#include <nvimgcodecs.h>
#include <string>
#include <memory>
#include "iimage_decoder.h"

namespace nvimgcdcs {
class IDecodeState;
class IImage;
class ICodeStream;

class ImageDecoder : public IImageDecoder
{
  public:
    ImageDecoder(const struct nvimgcdcsDecoderDesc* desc, nvimgcdcsDecodeParams_t* params);
    ~ImageDecoder() override;
    std::unique_ptr<IDecodeState> createDecodeState() const override;
    void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) override;
    void decode(ICodeStream* code_stream, IImage* image, nvimgcdcsDecodeParams_t* params) override;

  private:
    const struct nvimgcdcsDecoderDesc* decoder_desc_;
    nvimgcdcsDecoder_t decoder_;
};

} // namespace nvimgcdcs