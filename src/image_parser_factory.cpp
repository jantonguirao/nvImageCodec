/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <cassert>
#include <iostream>
#include "image_parser.h"
#include "image_parser_factory.h"
#include "log.h"

namespace nvimgcdcs {

ImageParserFactory::ImageParserFactory(const struct nvimgcdcsParserDesc* desc)
    : parser_desc_(desc)
{
}
std::string ImageParserFactory::getParserId() const
{
    return parser_desc_->id;
}

std::string ImageParserFactory::getCodecName() const
{
    return parser_desc_->codec;
}

std::unique_ptr<IImageParser> ImageParserFactory::createParser() const
{
    return std::make_unique<ImageParser>(parser_desc_);
}

bool ImageParserFactory::canParse(nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("ImageParser::canParse");
    assert(code_stream);
    bool result = false;
    parser_desc_->canParse(parser_desc_->instance, &result, code_stream);
    return result;
}

} // namespace nvimgcdcs