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

#include <memory>
#include <string>

#include "file_io_stream.h"
#include "iiostream_factory.h"
#include "mem_io_stream.h"

namespace nvimgcodec {

class IoStream;
class IoStreamFactory : public IIoStreamFactory
{
  public:
    std::unique_ptr<IoStream> createFileIoStream(const std::string& file_name, bool read_ahead, bool use_mmap, bool to_write) const override
    {
        return FileIoStream::open(file_name, read_ahead, use_mmap, to_write);
    }

    std::unique_ptr<IoStream> createMemIoStream(const unsigned char* data, size_t size) const override
    {
        return std::make_unique<MemIoStream<const unsigned char>>(data, size);
    }

    std::unique_ptr<IoStream> createMemIoStream(void* ctx, std::function<unsigned char*(void* ctx, size_t)> resize_buffer_func) const override
    {
        return std::make_unique<MemIoStream<unsigned char>>(ctx, resize_buffer_func);
    }
};

} // namespace nvimgcodec