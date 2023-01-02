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
#include "iimage_parser_factory.h"

namespace nvimgcdcs {

class IImageParser;
class ImageParserFactory : public IImageParserFactory
{
  public:
    explicit ImageParserFactory(const struct nvimgcdcsParserDesc* desc);
    std::string getParserId() const override;
    std::string getCodecName() const override;
    bool canParse(nvimgcdcsCodeStreamDesc_t code_stream) override;
    std::unique_ptr<IImageParser> createParser() const override;

  private:
    const struct nvimgcdcsParserDesc* parser_desc_;
};

} // namespace nvimgcdcs