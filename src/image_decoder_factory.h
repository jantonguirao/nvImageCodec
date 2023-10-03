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

#include <nvimgcodec.h>
#include <memory>
#include <string>
#include "iimage_decoder_factory.h"

namespace nvimgcodec {

class IImageDecoder;

class ImageDecoderFactory : public IImageDecoderFactory
{
  public:
    explicit ImageDecoderFactory(const nvimgcodecDecoderDesc_t* desc);
    std::string getDecoderId() const override;
    std::string getCodecName() const override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    std::unique_ptr<IImageDecoder> createDecoder(
        const nvimgcodecExecutionParams_t* exec_params, const char* options) const override;

  private:
    const nvimgcodecDecoderDesc_t* decoder_desc_;
};
} // namespace nvimgcodec