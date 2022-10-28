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
#include <vector>
#include "codec.h"
namespace nvimgcdcs {

class ImageParser;
class CodecRegistry
{
  public:
    CodecRegistry();
    ~CodecRegistry() = default;
    void registerCodec(std::unique_ptr<Codec> codec);
    const std::pair<Codec*, std::unique_ptr<ImageParser>> getCodecAndParser(
        nvimgcdcsCodeStreamDesc_t code_stream) const;
    Codec* getCodecByName(const char* name);
  private:
    std::vector<Codec*> codec_ptrs_;
    std::map<std::string, std::unique_ptr<Codec>> by_name_;
};
} // namespace nvimgcdcs