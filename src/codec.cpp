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

int Codec::getDecodersNum() const
{
    return decoders_.size();
}

std::unique_ptr<IImageDecoder> Codec::createDecoder(int index, int device_id) const
{
    if (size_t(index) >= decoders_.size()) {
        return std::unique_ptr<IImageDecoder>();
    }
    auto it = decoders_.begin();
    for (int i = 0; i < index; ++i)
        it++;
    return it->second->createDecoder(device_id);
}

int Codec::getEncodersNum() const
{
    return encoders_.size();
}

std::unique_ptr<IImageEncoder> Codec::createEncoder(int index, int device_id) const
{
    if (size_t(index) >= encoders_.size()) {
        return std::unique_ptr<IImageEncoder>();
    }
    auto it = encoders_.begin();
    for (int i = 0; i < index; ++i)
        it++;
    return it->second->createEncoder(device_id);
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