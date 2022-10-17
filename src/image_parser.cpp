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

namespace nvimgcdcs {

ImageParser::ImageParser(const struct nvimgcdcsParserDesc* desc)
    : parser_desc_(desc)
{
}

const std::string ImageParser::getCodecName() const
{
    return parser_desc_->codec;
}

bool ImageParser::canParse(CodeStream* code_stream)
{
    assert(code_stream);
    bool result = true;
    parser_desc_->canParse(&result, code_stream->getInputStreamDesc());
    return result;
}

} // namespace nvimgcdcs