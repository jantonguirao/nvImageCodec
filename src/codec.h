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
class CodeStream;
class ImageDecoderFactory;
class ImageEncoderFactory;
class ImageParser;

class Codec
{
  public:
    explicit Codec(const char* name);

    bool matches(CodeStream* code_stream) const;

    const std::string& name() const;

    std::span<ImageParser* const> parsers() const;
    std::span<ImageEncoderFactory* const> encoders() const;
    std::span<ImageDecoderFactory* const> decoders() const;

    void registerParser(std::unique_ptr<ImageParser> factory, float priority);
    void registerEncoder(std::unique_ptr<ImageEncoderFactory> factory, float priority);
    void registerDecoder(std::unique_ptr<ImageDecoderFactory> factory, float priority);

  private:
    std::string name_;
    std::multimap<float, std::unique_ptr<ImageParser>> parsers_;
    std::multimap<float, std::unique_ptr<ImageEncoderFactory>> encoders_;
    std::multimap<float, std::unique_ptr<ImageDecoderFactory>> decoders_;
    std::vector<ImageParser*> parser_ptrs_;
    std::vector<ImageEncoderFactory*> encoder_ptrs_;
    std::vector<ImageDecoderFactory*> decoder_ptrs_;
};
} // namespace nvimgcdcs