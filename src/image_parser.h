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
#include "iimage_parser.h"

namespace nvimgcdcs {
class ImageParser : public IImageParser
{
  public:
    explicit ImageParser(const nvimgcdcsParserDesc_t* desc);
    ~ImageParser() override;
    std::string getParserId() const override;
    std::string getCodecName() const override;
    nvimgcdcsStatus_t getImageInfo(nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageInfo_t* image_info) override;
  private:
    const nvimgcdcsParserDesc_t* parser_desc_;
    nvimgcdcsParser_t parser_;
};

} // namespace nvimgcdcs