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
#include "image_decoder.h"
#include "image_encoder.h"
#include "image_parser.h"

#include <iostream>

namespace nvimgcdcs {
Codec::Codec(const char* name)
    : name_(name)
{
}

std::unique_ptr <ImageParser> Codec::createParser(nvimgcdcsCodeStreamDesc_t code_stream) const
{
    std::cout << "Codec::getParser " << name_ << std::endl;
    for (const auto& entry : parsers_) {
        std::cout << "- probing parser:" << entry.second->getParserId() << std::endl;
        if (entry.second->canParse(code_stream)) {
            return entry.second->createParser();
        }
    }

    return nullptr;
}

std::unique_ptr<ImageDecoder> Codec::createDecoder(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params) const
{
    std::cout << "Codec::getDecoder " << name_ << std::endl;
    for (const auto& entry : decoders_) {
        std::cout << "- probing decoder:" << entry.second->getDecoderId() << std::endl;
        if (entry.second->canDecode(code_stream , params)) {
          return entry.second->createDecoder(params);
        }
    }
    return nullptr;
}

const std::string& Codec::name() const
{
    return name_;
}

void Codec::registerParser(std::unique_ptr<ImageParserFactory> parserFactory, float priority)
{
    std::cout << "Codec::registerParser" << std::endl;
    auto it = parsers_.emplace(priority, std::move(parserFactory));
}

void Codec::registerEncoder(std::unique_ptr<ImageEncoderFactory> encoderFactory, float priority)
{
    auto it = encoders_.emplace(priority, std::move(encoderFactory));
}

void Codec::registerDecoder(std::unique_ptr<ImageDecoderFactory> decoderFactory, float priority)
{
    auto it = decoders_.emplace(priority, std::move(decoderFactory));
}

} // namespace nvimgcdcs