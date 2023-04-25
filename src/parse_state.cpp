/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "parse_state.h"

namespace nvimgcdcs {

ParseState::ParseState(const struct nvimgcdcsParserDesc* parser_desc, nvimgcdcsParser_t parser)
    : parser_desc_(parser_desc)
    , parser_(parser)
    , parse_state_(nullptr)
{
    parser_desc_->createParseState(parser_, &parse_state_);
}

ParseState::~ParseState()
{
    if (parse_state_)
        parser_desc_->destroyParseState(parse_state_);
}

nvimgcdcsParseState_t ParseState::getInternalParseState()
{
    return parse_state_;
}
} // namespace nvimgcdcs