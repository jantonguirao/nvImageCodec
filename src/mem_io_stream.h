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

#include <cstring>
#include <functional>

#include "io_stream.h"

namespace nvimgcdcs {

template <typename T>
class MemIoStream : public IoStream
{
  public:
    MemIoStream() = default;
    ~MemIoStream() = default;
    MemIoStream(T* mem, size_t bytes)
        : start_{mem}
        , size_{bytes}
    {
    }

    MemIoStream(void* ctx, std::function<unsigned char*(void* ctx, size_t)> resize_buffer_func)
        : resize_buffer_ctx_(ctx)
        , resize_buffer_func_(resize_buffer_func)
    {
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
    std::size_t write(void* buf, size_t bytes)
    {
        if constexpr (!std::is_const<T>::value) {
            ptrdiff_t left = size_ - pos_;
            if (left < static_cast<ptrdiff_t>(bytes))
                bytes = left;

            std::memcpy(static_cast<void*>(start_ + pos_), buf, bytes);
            pos_ += bytes;
            return bytes;
        } else {
            assert(!"Forbiden write for const type");
            return 0;
        }
    }

    std::size_t putc(unsigned char ch)
    {

        if constexpr (!std::is_const<T>::value) {
            ptrdiff_t left = size_ - pos_;
            if (left < 1)
                return 0;
            std::memcpy(static_cast<void*>(start_ + pos_), &ch, 1);
            pos_++;
            return 1;
        } else {
            assert(!"Forbiden write for const type");
            return 0;
        }
    

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
        if (offset < 0 || offset > int64_t(size_))
            throw std::out_of_range("The requested position in the stream is out of range");
        pos_ = offset;
    }

    std::size_t size() const override { return size_; }

    void reserve(size_t bytes) override
    {
        if (resize_buffer_func_ && (bytes > size_)) {
            start_ = resize_buffer_func_(resize_buffer_ctx_, bytes);
            size_ = bytes;
        }
    }

    void flush() override
    {
        if (resize_buffer_func_&& (size_ != pos_)) {
            start_ = resize_buffer_func_(resize_buffer_ctx_, pos_);
            size_ = pos_;
        }
    }

    void* map(size_t offset, size_t size) const override {
        assert(offset + size <= size_);
        return (void*)(start_ + offset);
    }

  private:
    T* start_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
    void* resize_buffer_ctx_ = nullptr;
    std::function<unsigned char*(void*, size_t)> resize_buffer_func_ = nullptr;
};

} // namespace nvimgcdcs