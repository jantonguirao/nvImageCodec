/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "codec.h"
#include <iostream>
#include "image_decoder.h"
#include "image_encoder.h"
#include "image_parser.h"
#include "log.h"

namespace nvimgcdcs {
Codec::Codec(const char* name)
    : name_(name)
{
}

std::unique_ptr<IImageParser> Codec::createParser(nvimgcdcsCodeStreamDesc_t code_stream) const
{
    NVIMGCDCS_LOG_TRACE("Codec::getParser " << name_);
    for (const auto& entry : parsers_) {
        NVIMGCDCS_LOG_TRACE("- probing parser:" << entry.second->getParserId());
        if (entry.second->canParse(code_stream)) {
            NVIMGCDCS_LOG_TRACE("- - can parse");
            return entry.second->createParser();
        }
    }

    return nullptr;
}

std::unique_ptr<IImageDecoder> Codec::createDecoder(nvimgcdcsCodeStreamDesc_t code_stream,
    nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params) const
{
    NVIMGCDCS_LOG_TRACE("Codec::createDecoder " << name_);
    for (const auto& entry : decoders_) {
        NVIMGCDCS_LOG_TRACE("- probing decoder:" << entry.second->getDecoderId());
        if (entry.second->canDecode(code_stream, image, params)) {
            NVIMGCDCS_LOG_TRACE("- - can decode");
            return entry.second->createDecoder(params);
        }
    }
    return nullptr;
}

int Codec::getDecodersNum() const
{
    return decoders_.size();
}

std::unique_ptr<IImageDecoder> Codec::createDecoder(
    int index, const nvimgcdcsDecodeParams_t* params) const
{
    if (size_t(index) >= decoders_.size()) {
        return std::unique_ptr<IImageDecoder>();
    }
    auto it = decoders_.begin();
    for (int i = 0; i < index; ++i)
        it++;
    return it->second->createDecoder(params);
}

int Codec::getEncodersNum() const
{
    return encoders_.size();
}

std::unique_ptr<IImageEncoder> Codec::createEncoder(
    int index, const nvimgcdcsEncodeParams_t* params) const
{
    if (size_t(index) >= encoders_.size()) {
        return std::unique_ptr<IImageEncoder>();
    }
    auto it = encoders_.begin();
    for (int i = 0; i < index; ++i)
        it++;
    return it->second->createEncoder(params);
}

std::unique_ptr<IImageEncoder> Codec::createEncoder(nvimgcdcsImageDesc_t image,
    nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params) const
{
    NVIMGCDCS_LOG_TRACE("Codec::createEncoder " << name_);
    for (const auto& entry : encoders_) {
        NVIMGCDCS_LOG_TRACE("- probing encoder:" << entry.second->getEncoderId());
        if (entry.second->canEncode(image, code_stream, params)) {
            NVIMGCDCS_LOG_TRACE("- - can encode");
            return entry.second->createEncoder(params);
        }
    }
    return nullptr;
}

const std::string& Codec::name() const
{
    return name_;
}

void Codec::registerParserFactory(
    std::unique_ptr<IImageParserFactory> parserFactory, float priority)
{
    NVIMGCDCS_LOG_TRACE("Codec::registerParser");
    parsers_.emplace(priority, std::move(parserFactory));
}

void Codec::registerEncoderFactory(
    std::unique_ptr<IImageEncoderFactory> encoderFactory, float priority)
{
    NVIMGCDCS_LOG_TRACE("Codec::registerEncoder");
    encoders_.emplace(priority, std::move(encoderFactory));
}

void Codec::registerDecoderFactory(
    std::unique_ptr<IImageDecoderFactory> decoderFactory, float priority)
{
    NVIMGCDCS_LOG_TRACE("Codec::registerDecoder");
    decoders_.emplace(priority, std::move(decoderFactory));
}

} // namespace nvimgcdcs