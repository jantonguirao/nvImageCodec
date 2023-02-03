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
#include "iparse_state.h"

namespace nvimgcdcs {
class ParseState : public IParseState
{
  public:
    ParseState(const struct nvimgcdcsParserDesc*parser_desc, nvimgcdcsParser_t parser);
    ~ParseState() override;
    nvimgcdcsParseState_t getInternalParseState() override;

  private:
    const struct nvimgcdcsParserDesc* parser_desc_;
    nvimgcdcsParser_t parser_;
    nvimgcdcsParseState_t parse_state_;
};
} // namespace nvimgcdcs