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

#include "iimage_decoder_factory.h"
#include "iimage_encoder_factory.h"
#include "iimage_parser_factory.h"

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
    std::unique_ptr<IImageParser> createParser(nvimgcdcsCodeStreamDesc_t* code_stream) const override;
    int getDecodersNum() const override;
    IImageDecoderFactory* getDecoderFactory(int index) const override;
    int getEncodersNum() const override;
    IImageEncoderFactory* getEncoderFactory(int index) const override;
    void registerEncoderFactory(std::unique_ptr<IImageEncoderFactory> factory, float priority) override;
    void unregisterEncoderFactory(const std::string encoder_id) override;
    void registerDecoderFactory(std::unique_ptr<IImageDecoderFactory> factory, float priority) override;
    void unregisterDecoderFactory(const std::string decoder_id) override;
    void registerParserFactory(std::unique_ptr<IImageParserFactory> factory, float priority) override;
    void unregisterParserFactory(const std::string parser_id) override;

  private:
    std::string name_;
    std::multimap<float, std::unique_ptr<IImageParserFactory>> parsers_;
    std::multimap<float, std::unique_ptr<IImageEncoderFactory>> encoders_;
    std::multimap<float, std::unique_ptr<IImageDecoderFactory>> decoders_;
};
} // namespace nvimgcdcs
