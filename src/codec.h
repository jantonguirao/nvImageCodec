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

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "image_decoder.h"
#include "image_encoder.h"
#include "image_parser.h"

namespace nvimgcdcs {

class Codec
{
  public:
    explicit Codec(const char* name);

    std::unique_ptr<ImageParser> createParser(nvimgcdcsCodeStreamDesc_t code_stream) const;
    std::unique_ptr<ImageDecoder> createDecoder(
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params) const;
    std::unique_ptr<ImageEncoder> createEncoder(
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params) const;

    const std::string& name() const;

    void registerParser(std::unique_ptr<ImageParserFactory> factory, float priority);
    void registerEncoder(std::unique_ptr<ImageEncoderFactory> factory, float priority);
    void registerDecoder(std::unique_ptr<ImageDecoderFactory> factory, float priority);

  private:
    std::string name_;
    std::multimap<float, std::unique_ptr<ImageParserFactory>> parsers_;
    std::multimap<float, std::unique_ptr<ImageEncoderFactory>> encoders_;
    std::multimap<float, std::unique_ptr<ImageDecoderFactory>> decoders_;
};
} // namespace nvimgcdcs