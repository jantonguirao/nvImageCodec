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

#include <filesystem>

namespace nvimgcdcs {

namespace fs = std::filesystem;

class IDirectoryScaner
{
  public:
    virtual ~IDirectoryScaner()                   = default;
    virtual void start(const fs::path& directory) = 0;
    virtual bool hasMore()                         = 0;
    virtual fs::path next()                       = 0;
};

} // namespace nvimgcdcs