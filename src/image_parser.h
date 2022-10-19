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

#include <nvimgcdcs_module.h>
#include <string>
#include <memory>

namespace nvimgcdcs {

class ImageParser
{
  public:
    ImageParser(const struct nvimgcdcsParserDesc* desc);
    ~ImageParser();
    void getImageInfo(nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageInfo_t* image_info);

  private:
    const struct nvimgcdcsParserDesc* parser_desc_;
    nvimgcdcsParser_t parser_;
};

class ImageParserFactory
{
  public:
    ImageParserFactory(const struct nvimgcdcsParserDesc* desc);
    std::string getParserId() const;
    std::string getCodecName() const;
    bool canParse(nvimgcdcsCodeStreamDesc_t code_stream);
    std::unique_ptr<ImageParser> createParser() const;
  private:
    const struct nvimgcdcsParserDesc* parser_desc_;
};

} // namespace nvimgcdcs