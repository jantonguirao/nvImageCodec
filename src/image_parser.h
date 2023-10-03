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
#include "iimage_parser.h"

namespace nvimgcodec {
class ImageParser : public IImageParser
{
  public:
    explicit ImageParser(const nvimgcodecParserDesc_t* desc);
    ~ImageParser() override;
    std::string getParserId() const override;
    std::string getCodecName() const override;
    nvimgcodecStatus_t getImageInfo(nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageInfo_t* image_info) override;
  private:
    const nvimgcodecParserDesc_t* parser_desc_;
    nvimgcodecParser_t parser_;
};

} // namespace nvimgcodec