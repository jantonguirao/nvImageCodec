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
#include "../src/icodec_registry.h"
#include "../src/iimage_parser.h"
#include "../src/icodec.h"
#include <memory>

namespace nvimgcodec { namespace test {

class MockCodecRegistry : public ICodecRegistry
{
  public:
    MOCK_METHOD(void, registerCodec, (std::unique_ptr<ICodec> codec), (override));
    MOCK_METHOD((std::unique_ptr<IImageParser>), getParser,(
        nvimgcodecCodeStreamDesc_t* code_stream), (const, override));
    MOCK_METHOD(ICodec*, getCodecByName, (const char* name), (override));
    MOCK_METHOD(size_t, getCodecsCount, (), (const, override));
    MOCK_METHOD(ICodec*, getCodecByIndex, (size_t i), (override));
};

}} // namespace nvimgcodec::test