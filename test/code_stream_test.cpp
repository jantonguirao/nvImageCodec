/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <string>

#include "../src/code_stream.h"
#include "mock_codec.h"
#include "mock_codec_registry.h"
#include "mock_image_parser.h"
#include "mock_iostream_factory.h"
#include "mock_parse_state.h"

namespace nvimgcdcs { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::Const;

TEST(CodeStreamTest, test_parse_from_file)
{
    const std::string codec_name("test_codec");
    MockCodec codec;
    EXPECT_CALL(codec, name()).WillRepeatedly(ReturnRef(codec_name));

    nvimgcdcsParseState_t nvimg_parse_state = nullptr;

    std::unique_ptr<MockParseState> parse_state = std::make_unique<MockParseState>();
    EXPECT_CALL(*parse_state.get(), getInternalParseState())
        .WillRepeatedly(Return(nvimg_parse_state));

    std::unique_ptr<MockImageParser> parser = std::make_unique<MockImageParser>();
    EXPECT_CALL(*parser.get(), createParseState()).WillRepeatedly(Return(ByMove(std::move(parse_state))));

    MockCodecRegistry codec_registry;
    EXPECT_CALL(codec_registry, getParser(_))
        .Times(1)
        .WillRepeatedly(Return(ByMove(std::move(parser))));

    std::unique_ptr<MockIoStreamFactory> iostream_factory = std::make_unique<MockIoStreamFactory>();
    EXPECT_CALL(*iostream_factory.get(), createFileIoStream(_, _, _, false)).Times(1);

    CodeStream code_stream(&codec_registry, std::move(iostream_factory));
    code_stream.parseFromFile("test_file");
}

TEST(CodeStreamTest, test_parse_from_mem)
{
    const std::string codec_name("test_codec");
    MockCodec codec;
    EXPECT_CALL(codec, name()).WillRepeatedly(ReturnRef(codec_name));

    nvimgcdcsParseState_t nvimg_parse_state = nullptr;

    std::unique_ptr<MockParseState> parse_state = std::make_unique<MockParseState>();
    EXPECT_CALL(*parse_state.get(), getInternalParseState())
        .WillRepeatedly(Return(nvimg_parse_state));

    std::unique_ptr<MockImageParser> parser = std::make_unique<MockImageParser>();
    EXPECT_CALL(*parser.get(), createParseState())
        .WillRepeatedly(Return(ByMove(std::move(parse_state))));

    MockCodecRegistry codec_registry;
    EXPECT_CALL(codec_registry, getParser(_))
        .Times(1)
        .WillRepeatedly(
            Return(ByMove(std::move(parser))));

    std::unique_ptr<MockIoStreamFactory> iostream_factory = std::make_unique<MockIoStreamFactory>();
    EXPECT_CALL(*iostream_factory.get(), createMemIoStream(_, _)).Times(1);

    CodeStream code_stream(&codec_registry, std::move(iostream_factory));
    code_stream.parseFromMem(nullptr, 0);
}

}} // namespace nvimgcdcs::test
