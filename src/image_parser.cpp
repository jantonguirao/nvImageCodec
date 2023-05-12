/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_parser.h"
#include <cassert>
#include <iostream>
#include "log.h"

namespace nvimgcdcs {

ImageParser::ImageParser(const struct nvimgcdcsParserDesc* desc)
    : parser_desc_(desc)
{
    parser_desc_->create(parser_desc_->instance, &parser_);
}

ImageParser::~ImageParser()
{
    parser_desc_->destroy(parser_);
}
std::string ImageParser::getParserId() const
{
    return parser_desc_->id;
}

std::string ImageParser::getCodecName() const
{
    return parser_desc_->codec;
}

nvimgcdcsStatus_t ImageParser::getImageInfo(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageInfo_t* image_info)
{
    NVIMGCDCS_LOG_TRACE("ImageParser::getImageInfo");
    assert(code_stream);
    assert(parser_desc_->getImageInfo);
    return parser_desc_->getImageInfo(parser_, image_info, code_stream);
}

} // namespace nvimgcdcs