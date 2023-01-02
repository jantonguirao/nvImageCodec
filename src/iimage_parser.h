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
#include <memory>
#include <string>

namespace nvimgcdcs {
class IParseState;
class IImageParser
{
  public:
    virtual ~IImageParser()                  = default;
    virtual std::string getParserId() const  = 0;
    virtual std::string getCodecName() const = 0;
    virtual void getImageInfo(
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageInfo_t* image_info) = 0;
    virtual std::unique_ptr<IParseState> createParseState()                       = 0;
};

} // namespace nvimgcdcs