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
#include "iimage_parser.h"
#include "log.h"

#include <iostream>
#include <stdexcept>

namespace nvimgcdcs {

CodecRegistry::CodecRegistry()
{
}

void CodecRegistry::registerCodec(std::unique_ptr<ICodec> codec)
{
    if (by_name_.find(codec->name()) != by_name_.end())
        throw std::invalid_argument("A different codec with the same name already registered.");
    by_name_.insert(std::make_pair(codec->name(), std::move(codec)));
}

const std::pair<ICodec*, std::unique_ptr<IImageParser>> CodecRegistry::getCodecAndParser(
    nvimgcdcsCodeStreamDesc_t code_stream) const
{
    NVIMGCDCS_LOG_TRACE("CodecRegistry::getCodecAndParser");
    for (const auto& entry : by_name_) {
        std::unique_ptr<IImageParser> parser = entry.second->createParser(code_stream);
        if (parser) {
            return std::make_pair(entry.second.get(), std::move(parser));
        }
    }

    return std::make_pair(nullptr, nullptr);
}

ICodec* CodecRegistry::getCodecByName(const char* name)
{
    auto it = by_name_.find(name);
    if (it != by_name_.end())
        return it->second.get();
    else
        return nullptr;
}

} // namespace nvimgcdcs