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
#include <memory>
#include <string>

namespace nvimgcdcs {

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

    virtual std::unique_ptr<IImageParser> createParser(
        nvimgcdcsCodeStreamDesc_t code_stream) const = 0;
    virtual std::unique_ptr<IImageDecoder> createDecoder(nvimgcdcsCodeStreamDesc_t code_stream,
        nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params) const = 0;
    virtual int getDecodersNum() const = 0;
    virtual std::unique_ptr<IImageDecoder> createDecoder(
        int index, const nvimgcdcsDecodeParams_t* params) const = 0;
    virtual int getEncodersNum() const = 0;
    virtual std::unique_ptr<IImageEncoder> createEncoder(
        int index, const nvimgcdcsEncodeParams_t* params) const = 0;
    virtual std::unique_ptr<IImageEncoder> createEncoder(nvimgcdcsImageDesc_t image,
        nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params) const = 0;

    virtual void registerParserFactory(
        std::unique_ptr<IImageParserFactory> factory, float priority) = 0;
    virtual void registerEncoderFactory(
        std::unique_ptr<IImageEncoderFactory> factory, float priority) = 0;
    virtual void registerDecoderFactory(
        std::unique_ptr<IImageDecoderFactory> factory, float priority) = 0;
};
} // namespace nvimgcdcs