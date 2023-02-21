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
#include <string>
#include "idirectory_scaner.h"

namespace fs = std::filesystem;

namespace nvimgcdcs {

class DirectoryScaner : public IDirectoryScaner
{
  public:
    void start(const fs::path& directory) override { dir_it_ = fs::directory_iterator(directory); }
    bool hasMore() override { return dir_it_ != fs::directory_iterator(); }
    fs::path next() override
    {
        fs::path tmp = dir_it_->path();
        dir_it_++;
        return tmp;
    }
    fs::file_status symlinkStatus(const fs::path& p) override { return fs::symlink_status(p); }

  private:
    fs::directory_iterator dir_it_;
};

} // namespace nvimgcdcs