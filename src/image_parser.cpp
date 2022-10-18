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
#include "code_stream.h"

#include <cassert>
#include <iostream>

namespace nvimgcdcs {

ImageParser::ImageParser(const struct nvimgcdcsParserDesc* desc)
    : parser_desc_(desc)
{
}
const std::string ImageParser::getParserId() const
{

    return parser_desc_->id;
}

const std::string ImageParser::getCodecName() const
{
    return parser_desc_->codec;
}

bool ImageParser::canParse(CodeStream* code_stream)
{
    std::cout << "ImageParser::canParse" << std::endl;
    assert(code_stream);
    bool result = false;
    parser_desc_->canParse(&result, code_stream->getInputStreamDesc());
    std::cout << "ImageParser::canParse:" << result << std::endl;
    return result;
}

void ImageParser::getImageInfo(CodeStream* code_stream, nvimgcdcsImageInfo_t* image_info)
{
    std::cout << "ImageParser::getImageInfo" << std::endl;
    assert(code_stream);
    assert(parser_desc_->getImageInfo);
    parser_desc_->getImageInfo(image_info, code_stream->getInputStreamDesc());
}

} // namespace nvimgcdcs