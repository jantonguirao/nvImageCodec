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

#include "file_io_stream.h"

namespace nvimgcdcs {

class StdFileIoStream : public FileIoStream
{
  public:
    explicit StdFileIoStream(const std::string& path, bool to_write);
    void close() override;
    std::shared_ptr<void> get(size_t n_bytes) override;
    size_t read(void* buffer, size_t n_bytes) override;
    std::size_t write(void* buffer, size_t n_bytes) override;
    std::size_t putc(unsigned char ch) override;
    void seek(ptrdiff_t pos, int whence = SEEK_SET) override;
    int64_t tell() const override;
    size_t size() const override;

    ~StdFileIoStream() override { StdFileIoStream::close(); }

  private:
    FILE* fp_;
};
} //nvimgcdcs