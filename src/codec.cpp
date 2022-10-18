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

ImageParser* Codec::matches(CodeStream* code_stream) const
{
    std::cout << "Codec::matches " << name_ << std::endl;
    for (const auto& entry : parsers_) {
        std::cout << "- probing parser:" << entry.second->getParserId() << std::endl;
        if (entry.second->canParse(code_stream))
            return entry.second.get();
    }

    return nullptr;
}

const std::string& Codec::name() const
{
    return name_;
}

std::span<ImageParser* const> Codec::parsers() const
{
    return {parser_ptrs_.data(), parser_ptrs_.size()};
}

std::span<ImageEncoderFactory* const> Codec::encoders() const
{
    return {encoder_ptrs_.data(), encoder_ptrs_.size()};
}

std::span<ImageDecoderFactory* const> Codec::decoders() const
{
    return {decoder_ptrs_.data(), decoder_ptrs_.size()};
}

void Codec::registerParser(std::unique_ptr<ImageParser> parser, float priority)
{
    std::cout << "Codec::registerParser" << std::endl;
    auto it = parsers_.emplace(priority, std::move(parser));
    if (std::next(it) == parsers_.end()) {
        parser_ptrs_.push_back(it->second.get());
    } else {
        parser_ptrs_.clear();
        for (const auto& entry : parsers_) {
            parser_ptrs_.push_back(entry.second.get());
        }
    }
}

void Codec::registerEncoder(std::unique_ptr<ImageEncoderFactory> encoderFactory, float priority)
{
    auto it = encoders_.emplace(priority, std::move(encoderFactory));
    if (std::next(it) == encoders_.end()) {
        encoder_ptrs_.push_back(it->second.get());
    } else {
        encoder_ptrs_.clear();
        for (const auto& entry : encoders_) {
            encoder_ptrs_.push_back(entry.second.get());
        }
    }
}

void Codec::registerDecoder(std::unique_ptr<ImageDecoderFactory> decoderFactory, float priority)
{
    auto it = decoders_.emplace(priority, std::move(decoderFactory));
    if (std::next(it) == decoders_.end()) {
        decoder_ptrs_.push_back(it->second.get());
    } else {
        decoder_ptrs_.clear();
        for (const auto& entry : decoders_) {
            decoder_ptrs_.push_back(entry.second.get());
        }
    }
}

} // namespace nvimgcdcs