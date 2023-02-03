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

#include "iiostream_factory.h"
#include "file_io_stream.h"
#include "mem_io_stream.h"

namespace nvimgcdcs {

class IoStream;
class IoStreamFactory : public IIoStreamFactory
{
  public:
    std::unique_ptr<IoStream> createFileIoStream(
        const std::string& file_name, bool read_ahead, bool use_mmap, bool to_write) const override
        {
            return FileIoStream::open(file_name, read_ahead, use_mmap, to_write);
        };
        virtual std::unique_ptr<IoStream> createMemIoStream(unsigned char* data,
            size_t size) const override
        {
            return std::make_unique<MemIoStream>(data, size);
        };
};

} // namespace nvimgcdcs