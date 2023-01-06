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
#include <string>

#include "iimage_parser_factory.h"
#include "iimage_encoder_factory.h"
#include "iimage_decoder_factory.h"

#include "icodec.h"

namespace nvimgcdcs {

class IImageParser;
class IImageEncoder;
class IImageDecoder;

class Codec : public ICodec
{
  public:
    explicit Codec(const char* name);
    const std::string& name() const override;
    std::unique_ptr<IImageParser> createParser(nvimgcdcsCodeStreamDesc_t code_stream) const override;
    std::unique_ptr<IImageDecoder> createDecoder(nvimgcdcsCodeStreamDesc_t code_stream,
        nvimgcdcsImageDesc_t image, nvimgcdcsDecodeParams_t* params) const override;
    std::unique_ptr<IImageEncoder> createEncoder(nvimgcdcsImageDesc_t image,
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params) const override;
    void registerParserFactory(
        std::unique_ptr<IImageParserFactory> factory, float priority) override;
    void registerEncoderFactory(
        std::unique_ptr<IImageEncoderFactory> factory, float priority) override;
    void registerDecoderFactory(
        std::unique_ptr<IImageDecoderFactory> factory, float priority) override;
  private:
    std::string name_;
    std::multimap<float, std::unique_ptr<IImageParserFactory>> parsers_;
    std::multimap<float, std::unique_ptr<IImageEncoderFactory>> encoders_;
    std::multimap<float, std::unique_ptr<IImageDecoderFactory>> decoders_;
};
} // namespace nvimgcdcs