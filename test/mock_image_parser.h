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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include "../src/iimage_parser.h"
#include "../src/iparse_state.h"
#include <memory>

namespace nvimgcdcs { namespace test {

class MockImageParser : public IImageParser
{
  public:
    MOCK_METHOD(std::string, getParserId,(), (const, override));
    MOCK_METHOD(std::string, getCodecName,(), (const, override));
    MOCK_METHOD(void, getImageInfo,(
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageInfo_t* image_info) ,(override));
    MOCK_METHOD(std::unique_ptr<IParseState>, createParseState,(),(override));
};

}} // namespace nvimgcdcs::test