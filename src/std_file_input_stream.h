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

#include "file_input_stream.h"

namespace nvimgcdcs {

class StdFileInputStream : public FileInputStream
{
  public:
    explicit StdFileInputStream(const std::string& path);
    void close() override;
    std::shared_ptr<void> get(size_t n_bytes) override;
    size_t read(void* buffer, size_t n_bytes) override;
    void seek(ptrdiff_t pos, int whence = SEEK_SET) override;
    int64_t tell() const override;
    size_t size() const override;

    ~StdFileInputStream() override { close(); }

  private:
    FILE* fp_;
};
} //nvimgcdcs