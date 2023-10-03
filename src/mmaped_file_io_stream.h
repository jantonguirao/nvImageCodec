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

namespace nvimgcodec {

class MmapedFileIoStream : public FileIoStream
{
  public:
    explicit MmapedFileIoStream(const std::string& path, bool read_ahead);
    void close() override;
    std::shared_ptr<void> get(size_t n_bytes) override;
    static bool reserveFileMappings(unsigned int num);
    static void freeFileMappings(unsigned int num);
    std::size_t read(void* buffer, size_t n_bytes) override;
    void seek(int64_t pos, int whence = SEEK_SET) override;
    int64_t tell() const override;
    std::size_t size() const override;

    ~MmapedFileIoStream() override { MmapedFileIoStream::close(); }

  private:
    std::shared_ptr<void> p_;
    std::size_t length_;
    std::size_t pos_;
    bool read_ahead_whole_file_;
};
} // namespace nvimgcodec