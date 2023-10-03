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

namespace nvimgcodec {

class IImageParser;
class IImageEncoder;
class IImageDecoder;
class IImageParserFactory;
class IImageEncoderFactory;
class IImageDecoderFactory;

class ICodec
{
  public:
    virtual ~ICodec() = default;

    virtual const std::string& name() const = 0;

    virtual std::unique_ptr<IImageParser> createParser(nvimgcodecCodeStreamDesc_t* code_stream) const = 0;
    virtual int getDecodersNum() const = 0;
    virtual IImageDecoderFactory* getDecoderFactory(int index) const = 0;
    virtual int getEncodersNum() const = 0;
    virtual IImageEncoderFactory* getEncoderFactory(int index) const = 0;

    virtual void registerEncoderFactory(std::unique_ptr<IImageEncoderFactory> factory, float priority) = 0;
    virtual void unregisterEncoderFactory(const std::string encoder_id) = 0;
    virtual void registerDecoderFactory(std::unique_ptr<IImageDecoderFactory> factory, float priority) = 0;
    virtual void unregisterDecoderFactory(const std::string decoder_id) = 0;
    virtual void registerParserFactory(std::unique_ptr<IImageParserFactory> factory, float priority) = 0;
    virtual void unregisterParserFactory(const std::string parser_id) = 0;
};
} // namespace nvimgcodec
