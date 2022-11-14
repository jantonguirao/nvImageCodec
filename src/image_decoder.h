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
#include <string>
#include <memory>

namespace nvimgcdcs {
class DecodeState;
class Image;
class CodeStream;

class ImageDecoder
{
  public:
    ImageDecoder(const struct nvimgcdcsDecoderDesc* desc, nvimgcdcsDecodeParams_t* params);
    ~ImageDecoder();
    std::unique_ptr<DecodeState> createDecodeState() const;
    void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
    void decode(CodeStream* code_stream, Image* image, nvimgcdcsDecodeParams_t* params);
  private:
    const struct nvimgcdcsDecoderDesc* decoder_desc_;
    nvimgcdcsDecoder_t decoder_;
};

class ImageDecoderFactory
{
  public:
    explicit ImageDecoderFactory(const struct nvimgcdcsDecoderDesc* desc);
    std::string getDecoderId() const;
    std::string getCodecName() const;
    bool canDecode(nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params);
    std::unique_ptr<ImageDecoder> createDecoder(nvimgcdcsDecodeParams_t* params) const;
  private:
    const struct nvimgcdcsDecoderDesc* decoder_desc_;
};
} // namespace nvimgcdcs