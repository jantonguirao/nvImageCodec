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

#include "io_stream.h"

namespace nvimgcdcs {

class MemIoStream : public IoStream
{
  public:
    MemIoStream()  = default;
    ~MemIoStream() = default;
    MemIoStream(void* mem, size_t bytes)
    {
        start_ = static_cast<char*>(mem);
        size_  = bytes;
    }

    std::size_t read(void* buf, size_t bytes) override
    {
        ptrdiff_t left = size_ - pos_;
        if (left < static_cast<ptrdiff_t>(bytes))
            bytes = left;
        std::memcpy(buf, start_ + pos_, bytes);
        pos_ += bytes;
        return bytes;
    }

    std::size_t write(void* buf, size_t bytes) override
    {
        ptrdiff_t left = size_ - pos_;
        if (left < static_cast<ptrdiff_t>(bytes))
            bytes = left;
        std::memcpy(static_cast<void*>(start_ + pos_), buf, bytes);
        pos_ += bytes;
        return bytes;
    }
    std::size_t putc(unsigned char ch) override
    {
        ptrdiff_t left = size_ - pos_;
        if (left < 1)
            return 0;
        std::memcpy(static_cast<void*>(start_ + pos_), &ch, 1);
        pos_++;
        return 1;
    }

    int64_t tell() const override { return pos_; }

    void seek(int64_t offset, int whence = SEEK_SET) override
    {
        if (whence == SEEK_CUR) {
            offset += pos_;
        } else if (whence == SEEK_END) {
            offset += size_;
        } else {
            assert(whence == SEEK_SET);
        }
        if (offset < 0 || offset > size_)
            throw std::out_of_range("The requested position in the stream is out of range");
        pos_ = offset;
    }

    std::size_t size() const override { return size_; }

  private:
    char* start_ = nullptr;
    ptrdiff_t size_    = 0;
    ptrdiff_t pos_     = 0;
};


} // namespace nvimgcdcs