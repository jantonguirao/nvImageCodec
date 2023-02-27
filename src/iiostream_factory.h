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


namespace nvimgcdcs {

class IoStream;
class IIoStreamFactory
{
  public:
    virtual ~IIoStreamFactory()              = default;
    virtual std::unique_ptr<IoStream> createFileIoStream(const std::string& file_name, bool read_ahead, bool use_mmap, bool to_write) const = 0;
    virtual std::unique_ptr<IoStream> createMemIoStream(const unsigned char* data, size_t size) const = 0;
    virtual std::unique_ptr<IoStream> createMemIoStream(
        unsigned char* data, size_t size) const = 0;
};

} // namespace nvimgcdcs