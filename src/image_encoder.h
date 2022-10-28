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

namespace nvimgcdcs {

class EncodeState;
class Image;
class CodeStream;

class ImageEncoder
{
  public:
    ImageEncoder(const struct nvimgcdcsEncoderDesc* desc, nvimgcdcsEncodeParams_t* params);
    ~ImageEncoder();
    std::unique_ptr<EncodeState> createEncodeState() const;
    void encode(CodeStream* code_stream, Image* image, nvimgcdcsEncodeParams_t* params);

  private:
    const struct nvimgcdcsEncoderDesc* encoder_desc_;
    nvimgcdcsEncoder_t encoder_;
};

class ImageEncoderFactory
{
  public:
    explicit ImageEncoderFactory(const struct nvimgcdcsEncoderDesc* desc);
    std::string getEncoderId() const;
    std::string getCodecName() const;
    bool canEncode(nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params);
    std::unique_ptr<ImageEncoder> createEncoder(nvimgcdcsEncodeParams_t* params) const;

  private:
    const struct nvimgcdcsEncoderDesc* encoder_desc_;
};
} // namespace nvimgcdcs