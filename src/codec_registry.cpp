/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "codec_registry.h"
#include "codec.h"

#include <stdexcept>

namespace nvimgcdcs {

CodecRegistry::CodecRegistry()
{
}

void CodecRegistry::registerCodec(std::unique_ptr<Codec> codec)
{
    if (by_name_.find(codec->name()) != by_name_.end())
        throw std::invalid_argument("A different codec with the same name already registered.");
    by_name_.insert(std::make_pair(codec->name(), std::move(codec)));

    codec_ptrs_.push_back(codec.get());
}

const Codec* CodecRegistry::getCodec(CodeStream* code_stream) const
{
    for (auto& codec : codec_ptrs_) {
        if (codec->matches(code_stream)) {
            return codec;
        }
    }
    return nullptr;
}

Codec* CodecRegistry::getCodecByName(const char* name)
{
    auto it = by_name_.find(name);
    if (it != by_name_.end())
        return it->second.get();
    else
        return nullptr;
}

std::span<Codec* const> CodecRegistry::codecs() const
{
    return {codec_ptrs_.data(), codec_ptrs_.size()};
}

} // namespace nvimgcdcs