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
#include <deque>
#include "icodec.h"
#include "icodec_registry.h"

namespace nvimgcdcs {

class IImageParser;

class CodecRegistry : public ICodecRegistry
{
  public:
    CodecRegistry();
    void registerCodec(std::unique_ptr<ICodec> codec) override;
    std::unique_ptr<IImageParser> getParser(
        nvimgcdcsCodeStreamDesc_t code_stream) const override;
    ICodec* getCodecByName(const char* name)  override;
  private:
    std::deque<ICodec*> codec_ptrs_;
    std::map<std::string, std::unique_ptr<ICodec>> by_name_;
};
} // namespace nvimgcdcs