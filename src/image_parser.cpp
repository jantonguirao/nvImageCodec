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

void ImageParser::getImageInfo(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageInfo_t* image_info)
{
    std::cout << "ImageParser::getImageInfo" << std::endl;
    assert(code_stream);
    assert(parser_desc_->getImageInfo);
    parser_desc_->getImageInfo(parser_, image_info, code_stream);
}

std::unique_ptr<ParseState> ImageParser::createParseState()
{
    return std::make_unique<ParseState>(parser_desc_, parser_);
}

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

std::unique_ptr<ImageParser> ImageParserFactory::createParser() const
{
    return std::make_unique<ImageParser>(parser_desc_);
}

bool ImageParserFactory::canParse(nvimgcdcsCodeStreamDesc_t code_stream)
{
    std::cout << "ImageParser::canParse" << std::endl;
    assert(code_stream);
    bool result = false;
    parser_desc_->canParse(parser_desc_->instance, & result, code_stream);
    std::cout << "ImageParser::canParse:" << result << std::endl;
    return result;
}


} // namespace nvimgcdcs