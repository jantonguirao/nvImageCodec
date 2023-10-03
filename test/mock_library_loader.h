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
#include "../src/ilibrary_loader.h"

namespace nvimgcodec {

class MockLibraryLoader : public ILibraryLoader
{
  public:
    MOCK_METHOD(LibraryHandle, loadLibrary, (const std::string& library_path), (override));
    MOCK_METHOD(void, unloadLibrary, (LibraryHandle library_handle), (override));
    MOCK_METHOD(void*, getFuncAddress, (LibraryHandle library_handle, const std::string& func_name),
        (override));
};

} // namespace nvimgcodec