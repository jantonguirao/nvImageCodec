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
#include <memory>
#include <string>
#include "iimage_decoder_factory.h"

namespace nvimgcdcs {

class IImageDecoder;

class ImageDecoderFactory : public IImageDecoderFactory
{
  public:
    explicit ImageDecoderFactory(const struct nvimgcdcsDecoderDesc* desc);
    std::string getDecoderId() const override;
    std::string getCodecName() const override;
    bool canDecode(
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params) override;
    std::unique_ptr<IImageDecoder> createDecoder(const nvimgcdcsDecodeParams_t* params) const override;

  private:
    const struct nvimgcdcsDecoderDesc* decoder_desc_;
};
} // namespace nvimgcdcs