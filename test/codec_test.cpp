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

#include "../src/codec.h"
#include "../src/iimage_parser.h"
#include "../src/iimage_parser_factory.h"
#include "mock_image_parser_factory.h"
#include "mock_logger.h"

namespace nvimgcdcs { namespace test {

TEST(CodecTest, parsers_are_probet_in_priority_order_when_registered_in_order_123)
{
    MockLogger logger;
    Codec codec(&logger, "test_codec");
    std::unique_ptr<MockImageParserFactory> factory1 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory2 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory3 = std::make_unique<MockImageParserFactory>();

    ::testing::Sequence s1;
    EXPECT_CALL(*factory1.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory2.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory3.get(), canParse(nullptr)).Times(1).InSequence(s1);

    codec.registerParserFactory(std::move(factory1), 1);
    codec.registerParserFactory(std::move(factory2), 2);
    codec.registerParserFactory(std::move(factory3), 3);

    nvimgcdcsCodeStreamDesc_t* code_stream = nullptr;
    std::unique_ptr<IImageParser> parser  = codec.createParser(code_stream);
}

TEST(CodecTest, parsers_are_probet_in_priority_order_when_registered_in_order_231)
{
    MockLogger logger;
    Codec codec(&logger, "test_codec");
    std::unique_ptr<MockImageParserFactory> factory1 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory2 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory3 = std::make_unique<MockImageParserFactory>();

    ::testing::Sequence s1;
    EXPECT_CALL(*factory1.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory2.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory3.get(), canParse(nullptr)).Times(1).InSequence(s1);

    codec.registerParserFactory(std::move(factory2), 2);
    codec.registerParserFactory(std::move(factory3), 3);
    codec.registerParserFactory(std::move(factory1), 1);

    nvimgcdcsCodeStreamDesc_t* code_stream = nullptr;
    std::unique_ptr<IImageParser> parser  = codec.createParser(code_stream);
}

TEST(CodecTest, parsers_are_probet_in_priority_order_when_registered_in_order_321)
{
    MockLogger logger;
    Codec codec(&logger, "test_codec");
    std::unique_ptr<MockImageParserFactory> factory1 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory2 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory3 = std::make_unique<MockImageParserFactory>();

    ::testing::Sequence s1;
    EXPECT_CALL(*factory1.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory2.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory3.get(), canParse(nullptr)).Times(1).InSequence(s1);

    codec.registerParserFactory(std::move(factory3), 3);
    codec.registerParserFactory(std::move(factory2), 2);
    codec.registerParserFactory(std::move(factory1), 1);

    nvimgcdcsCodeStreamDesc_t* code_stream = nullptr;
    std::unique_ptr<IImageParser> parser  = codec.createParser(code_stream);
}

}} // namespace nvimgcdcs::test
