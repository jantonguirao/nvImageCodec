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
#include "../src/idirectory_scaner.h"

#include <filesystem>

namespace nvimgcdcs {

namespace fs = std::filesystem;

class MockDirectoryScaner : public IDirectoryScaner
{
  public:
    MOCK_METHOD(void, start, (const fs::path& directory), (override));
    MOCK_METHOD(bool, hasMore, (), (override));
    MOCK_METHOD(fs::path, next, (), (override));
    MOCK_METHOD(fs::file_status, symlinkStatus, (const fs::path& p), (override));
    MOCK_METHOD(bool, exists, (const fs::path& p), (override));
};

} // namespace nvimgcdcs