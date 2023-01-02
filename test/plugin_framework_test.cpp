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

nvimgcdcsStatus_t testExtModuleCreate(
    nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsExtension_t* extension)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t testExtModuleDestroy(
    nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsExtension_t extension)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t testExtModuleEntry(nvimgcdcsExtensionDesc_t* ext_desc)
{
    ext_desc->create = &testExtModuleCreate;
    ext_desc->destroy = &testExtModuleDestroy;
    return NVIMGCDCS_STATUS_SUCCESS;
}

TEST(PluginFrameworkTest, test_ext_module_discovery)
{
    MockCodecRegistry codec_registry;

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
    EXPECT_CALL(*library_loader.get(), getFuncAddress(_, Eq("nvimgcdcsExtensionModuleEntry")))
        .WillRepeatedly(Return((ILibraryLoader::LibraryHandle)&testExtModuleEntry));
    EXPECT_CALL(*library_loader.get(), unloadLibrary(_)).Times(2);

    PluginFramework framework(
        &codec_registry, std::move(directory_scaner), std::move(library_loader));
    framework.discoverAndLoadExtModules();
}

}} // namespace nvimgcdcs::test