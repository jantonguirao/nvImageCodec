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
#include <map>
#include <memory>
#include <string>

namespace nvimgcdcs {

class ICodec;
class IImageParser;

class ICodecRegistry
{
  public:
    virtual ~ICodecRegistry() = default;
    virtual void registerCodec(std::unique_ptr<ICodec> codec) = 0;
    virtual const std::pair<ICodec*, std::unique_ptr<IImageParser>> getCodecAndParser(
        nvimgcdcsCodeStreamDesc_t code_stream) const = 0;
    virtual ICodec* getCodecByName(const char* name) = 0;
};
} // namespace nvimgcdcs