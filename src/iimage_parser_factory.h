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

#include <nvimgcodec.h>
#include <memory>
#include <string>

namespace nvimgcodec {

class IImageParser;
class IImageParserFactory
{
  public:
    virtual ~IImageParserFactory() = default;
    virtual std::string getParserId() const = 0;
    virtual std::string getCodecName() const = 0;
    virtual bool canParse(nvimgcodecCodeStreamDesc_t* code_stream) = 0;
    virtual std::unique_ptr<IImageParser> createParser() const = 0;
};

} // namespace nvimgcodec