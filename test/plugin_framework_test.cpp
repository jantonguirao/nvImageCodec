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
#include "../src/plugin_framework.h"
#include "mock_codec_registry.h"
#include "mock_directory_scaner.h"
#include "mock_library_loader.h"
#include <memory>

namespace nvimgcdcs { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Const;
using ::testing::Return;
using ::testing::ReturnPointee;
using ::testing::Eq;

uint32_t testExtModuleGetVersion()
{
    return 1;
}

nvimgcdcsStatus_t testExtModuleLoad(
    nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsExtModule_t* module)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t testExtModuleUnload(
    nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsExtModule_t module)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

TEST(PluginFrameworkTest, test_ext_module_discovery)
{
  /*  const std::string codec_name("test_codec");
    MockCodec codec;
    EXPECT_CALL(codec, name()).WillRepeatedly(ReturnRef(codec_name));

    nvimgcdcsParseState_t nvimg_parse_state = nullptr;

    std::unique_ptr<MockParseState> parse_state = std::make_unique<MockParseState>();
    EXPECT_CALL(*parse_state.get(), getInternalParseState())
        .WillRepeatedly(Return(nvimg_parse_state));

    std::unique_ptr<MockImageParser> parser = std::make_unique<MockImageParser>();
    EXPECT_CALL(*parser.get(), createParseState())
        .WillRepeatedly(Return(ByMove(std::move(parse_state))));
*/
    MockCodecRegistry codec_registry;
    // EXPECT_CALL(codec_registry, getCodecAndParser(_))
    //     .Times(1)
    //     .WillRepeatedly(Return(ByMove(std::make_pair(&codec, std::move(parser)))));

    std::unique_ptr<MockDirectoryScaner> directory_scaner = std::make_unique<MockDirectoryScaner>();
    EXPECT_CALL(*directory_scaner.get(), start(_)).Times(1);
    EXPECT_CALL(*directory_scaner.get(), hasMore())
        .Times(3)
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(false));

    EXPECT_CALL(*directory_scaner.get(), next())
        .Times(2)
        .WillOnce(Return("libnvjpeg.so.22.11.0.0"))
        .WillOnce(Return("libnvjpeg2k.so.0.6.0.0"));

    std::unique_ptr<MockLibraryLoader> library_loader = std::make_unique<MockLibraryLoader>();
    EXPECT_CALL(*library_loader.get(), loadLibrary(_)).Times(2);
    EXPECT_CALL(*library_loader.get(), getFuncAddress(_, Eq("nvimgcdcsExtModuleGetVersion")))
        .WillRepeatedly(Return((ILibraryLoader::LibraryHandle)&testExtModuleGetVersion));
    EXPECT_CALL(*library_loader.get(), getFuncAddress(_, Eq("nvimgcdcsExtModuleLoad")))
        .WillRepeatedly(Return((ILibraryLoader::LibraryHandle)&testExtModuleLoad));
    EXPECT_CALL(*library_loader.get(), getFuncAddress(_, Eq("nvimgcdcsExtModuleUnload")))
        .WillRepeatedly(Return((ILibraryLoader::LibraryHandle)&testExtModuleUnload));
    EXPECT_CALL(*library_loader.get(), unloadLibrary(_)).Times(2);

    PluginFramework framework(
        &codec_registry, std::move(directory_scaner), std::move(library_loader));
    framework.discoverAndLoadExtModules();
}

}} // namespace nvimgcdcs::test