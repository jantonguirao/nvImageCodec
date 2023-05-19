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
#include <sstream>
#include <string>
#include <vector>

#include "../src/plugin_framework.h"
#include "mock_codec_registry.h"
#include "mock_directory_scaner.h"
#include "mock_environment.h"
#include "mock_executor.h"
#include "mock_library_loader.h"

namespace nvimgcdcs { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Const;
using ::testing::Eq;
using ::testing::Return;
using ::testing::ReturnPointee;

uint32_t testExtModuleGetVersion()
{
    return 1;
}

nvimgcdcsStatus_t testExtModuleCreate(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t testExtModuleDestroy(nvimgcdcsExtension_t extension)
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

    std::unique_ptr<MockEnvironment> env = std::make_unique<MockEnvironment>();
    EXPECT_CALL(*env.get(), getVariable("NVIMGCODECS_EXTENSIONS_PATH")).Times(1).WillOnce(Return(""));

    std::unique_ptr<MockDirectoryScaner> directory_scaner = std::make_unique<MockDirectoryScaner>();
    EXPECT_CALL(*directory_scaner.get(), start(_)).Times(1);
    EXPECT_CALL(*directory_scaner.get(), hasMore())
        .Times(5)
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(false));

    EXPECT_CALL(*directory_scaner.get(), next())
        .Times(4)
        .WillOnce(Return("libnvjpeg.so.22.11.0.0"))
        .WillOnce(Return("libnvjpeg2k.so.0.6.0.0"))
        .WillOnce(Return("~libtiff.so.0.6.0.0"))
        .WillOnce(Return("libnvjpeg2k.so.0"));
    EXPECT_CALL(*directory_scaner.get(), symlinkStatus(_))
        .Times(4)
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::symlink)));
    EXPECT_CALL(*directory_scaner.get(), exists(_)).WillRepeatedly(Return(true));

    std::unique_ptr<MockLibraryLoader> library_loader = std::make_unique<MockLibraryLoader>();
    EXPECT_CALL(*library_loader.get(), loadLibrary(_)).Times(2);
    EXPECT_CALL(*library_loader.get(), getFuncAddress(_, Eq("nvimgcdcsExtensionModuleEntry")))
        .WillRepeatedly(Return((ILibraryLoader::LibraryHandle)&testExtModuleEntry));
    EXPECT_CALL(*library_loader.get(), unloadLibrary(_)).Times(2);

    std::unique_ptr<MockExecutor> executor = std::make_unique<MockExecutor>();

    PluginFramework framework(
        &codec_registry, std::move(env), std::move(directory_scaner), std::move(library_loader), std::move(executor), nullptr, nullptr, "");
    framework.discoverAndLoadExtModules();
}

class PluginFrameworkExtensionsPathTest : public ::testing::Test
{
  public:
    PluginFrameworkExtensionsPathTest() {}

    void SetUp() override
    {
        env_ = std::make_unique<MockEnvironment>();
        directory_scaner_ = std::make_unique<MockDirectoryScaner>();
        EXPECT_CALL(*directory_scaner_.get(), exists(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(*directory_scaner_.get(), hasMore()).WillRepeatedly(Return(false));
        library_loader_ = std::make_unique<MockLibraryLoader>();
        executor_ = std::make_unique<MockExecutor>();
    }

    void TearDown() override {}

    void TestExtensionsOnePath(const std::string& env_test_path, const std::string& soft_test_path, const std::string& expected_test_path)
    {
        EXPECT_CALL(*env_.get(), getVariable("NVIMGCODECS_EXTENSIONS_PATH")).WillRepeatedly(Return(env_test_path));
        EXPECT_CALL(*directory_scaner_.get(), start(fs::path(expected_test_path))).Times(1);

        PluginFramework framework(&codec_registry_, std::move(env_), std::move(directory_scaner_), std::move(library_loader_),
            std::move(executor_), nullptr, nullptr, soft_test_path);
        framework.discoverAndLoadExtModules();
    }

    void TestExtensionsMultiplePaths(
        const std::string& env_test_path, const std::string& soft_test_path, const std::vector<std::string>& expected_paths)
    {
        EXPECT_CALL(*env_.get(), getVariable("NVIMGCODECS_EXTENSIONS_PATH")).WillRepeatedly(Return(env_test_path));
        for (auto p : expected_paths) {
            EXPECT_CALL(*directory_scaner_.get(), start(fs::path(p))).Times(1);
        }

        PluginFramework framework(&codec_registry_, std::move(env_), std::move(directory_scaner_), std::move(library_loader_),
            std::move(executor_), nullptr, nullptr, soft_test_path);
        framework.discoverAndLoadExtModules();
    }

    MockCodecRegistry codec_registry_;
    std::unique_ptr<MockEnvironment> env_;
    std::unique_ptr<MockDirectoryScaner> directory_scaner_;
    std::unique_ptr<MockLibraryLoader> library_loader_;
    std::unique_ptr<MockExecutor> executor_;
};

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_empty_env_and_soft_return_default)
{
    std::string env_test_path{""};
    std::string soft_test_path{""};
    std::string expected_test_path{DefaultExtensionsPath};
    TestExtensionsOnePath(env_test_path, soft_test_path, expected_test_path);
}

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_filled_env_and_empty_soft_return_env)
{
    std::string env_test_path{"/usr/env_test_path"};
    std::string soft_test_path{""};
    std::string expected_test_path{env_test_path};
    TestExtensionsOnePath(env_test_path, soft_test_path, expected_test_path);
}

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_filled_env_and_soft_return_soft)
{
    std::string env_test_path{"/usr/env_test_path"};
    std::string soft_test_path{"/usr/soft_test_path"};
    std::string expected_test_path{soft_test_path};

    TestExtensionsOnePath(env_test_path, soft_test_path, expected_test_path);
}

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_env_with_multiple_paths)
{
    std::vector<std::string> expected_test_paths{"/usr/env_test_path1", "/usr/env_test_path2", "/usr/env_test_path3"};
    std::stringstream ss;
    ss << expected_test_paths[0] << PathSeparator << expected_test_paths[1] << PathSeparator << expected_test_paths[2];
    std::string env_test_path{ss.str()};

    std::string soft_test_path{""};
    std::string expected_test_path{soft_test_path};

    TestExtensionsMultiplePaths(env_test_path, soft_test_path, expected_test_paths);
}
TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_soft_with_multiple_paths)
{
    std::vector<std::string> expected_test_paths{"/usr/soft_test_path1", "/usr/soft_test_path2", "/usr/soft_test_path3"};
    std::stringstream ss;
    ss << expected_test_paths[0] << PathSeparator << expected_test_paths[1] << PathSeparator << expected_test_paths[2];
    std::string env_test_path{"/usr/env_test_path"};

    std::string soft_test_path{ss.str()};
    std::string expected_test_path{soft_test_path};

    TestExtensionsMultiplePaths(env_test_path, soft_test_path, expected_test_paths);
}

class PluginFrameworkExtensionsVersionTest : public ::testing::Test
{
  public:
    PluginFrameworkExtensionsVersionTest() {}

    void SetUp() override
    {
        env_ = std::make_unique<MockEnvironment>();
        EXPECT_CALL(*env_.get(), getVariable("NVIMGCODECS_EXTENSIONS_PATH")).WillRepeatedly(Return(""));
        directory_scaner_ = std::make_unique<MockDirectoryScaner>();
        EXPECT_CALL(*directory_scaner_.get(), exists(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(*directory_scaner_.get(), hasMore()).WillRepeatedly(Return(false));

        library_loader_ = std::make_unique<MockLibraryLoader>();
        executor_ = std::make_unique<MockExecutor>();
        framework_ = std::make_unique<PluginFramework>(&codec_registry_, std::move(env_), std::move(directory_scaner_),
            std::move(library_loader_), std::move(executor_), nullptr, nullptr, "");
    }

    void TearDown() override {}

    static nvimgcdcsStatus_t ExtCreate(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
    {
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t ExtDestroy(nvimgcdcsExtension_t extension) { return NVIMGCDCS_STATUS_SUCCESS; }

    MockCodecRegistry codec_registry_;
    std::unique_ptr<MockEnvironment> env_;
    std::unique_ptr<MockDirectoryScaner> directory_scaner_;
    std::unique_ptr<MockLibraryLoader> library_loader_;
    std::unique_ptr<MockExecutor> executor_;
    std::unique_ptr<PluginFramework> framework_;
};

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_there_is_already_ext_with_the_same_ver_register_returns_invalid)
{
    nvimgcdcsExtension_t ext1;
    nvimgcdcsExtensionDesc_t ext1_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, nullptr, "test_ext", 1000, 1000,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCDCS_STATUS_INVALID_PARAMETER, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->unregisterExtension(ext1));
}

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_there_is_already_ext_with_the_newer_ver_register_returns_invalid)
{
    nvimgcdcsExtension_t ext1;
    nvimgcdcsExtensionDesc_t ext1_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, nullptr, "test_ext", 1000, 1000,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};
    nvimgcdcsExtension_t ext2;
    nvimgcdcsExtensionDesc_t ext2_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, nullptr, "test_ext", 0100, 1000,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCDCS_STATUS_INVALID_PARAMETER, framework_->registerExtension(&ext2, &ext2_desc));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->unregisterExtension(ext1));
}

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_there_is_already_ext_with_the_older_ver_register_returns_success)
{
    nvimgcdcsExtension_t ext1;
    nvimgcdcsExtensionDesc_t ext1_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, nullptr, "test_ext", 1000, NVIMGCDCS_EXT_API_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};
    nvimgcdcsExtension_t ext2;
    nvimgcdcsExtensionDesc_t ext2_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, nullptr, "test_ext", 2000, NVIMGCDCS_EXT_API_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->registerExtension(&ext2, &ext2_desc));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, framework_->unregisterExtension(ext1));
}

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_registering_ext_with_newer_api_ver_register_returns_unsupported)
{
    nvimgcdcsExtension_t ext1;
    nvimgcdcsExtensionDesc_t ext1_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, nullptr, "test_ext", 1000,
        NVIMGCDCS_EXT_API_VER + 1, &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED, framework_->registerExtension(&ext1, &ext1_desc));
}

}} // namespace nvimgcdcs::test
