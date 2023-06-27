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
#include "../src/icodec.h"
#include "../src/iimage_decoder.h"
#include "../src/iimage_decoder_factory.h"
#include "../src/iimage_encoder.h"
#include "../src/iimage_encoder_factory.h"
#include "../src/iimage_parser.h"
#include "../src/iimage_parser_factory.h"

#include <memory>

namespace nvimgcdcs { namespace test {

class MockCodec : public ICodec
{
  public:
    MOCK_METHOD(const std::string&, name, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageParser>, createParser,
        (nvimgcdcsCodeStreamDesc_t code_stream), (const, override));
    MOCK_METHOD(int, getDecodersNum, (), (const, override));
    MOCK_METHOD(IImageDecoderFactory*, getDecoderFactory, (int index), (const, override));
    MOCK_METHOD(int, getEncodersNum, (), (const, override));
    MOCK_METHOD(IImageEncoderFactory*, getEncoderFactory, (int index), (const, override));
    MOCK_METHOD(void, registerEncoderFactory,
        (std::unique_ptr<IImageEncoderFactory> factory, float priority), (override));
    MOCK_METHOD(void, unregisterEncoderFactory,(const std::string encoder_id), (override));
    MOCK_METHOD(void, registerDecoderFactory,
        (std::unique_ptr<IImageDecoderFactory> factory, float priority), (override));
    MOCK_METHOD(void, unregisterDecoderFactory, (const std::string decoder_id), (override));
    MOCK_METHOD(void, registerParserFactory,
        (std::unique_ptr<IImageParserFactory> factory, float priority), (override));
    MOCK_METHOD(void, unregisterParserFactory, (const std::string parser_id) ,(override));
};

}} // namespace nvimgcdcs::test