/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <errno.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <iostream>
#include "std_file_input_stream.h"

namespace nvimgcdcs {

StdFileInputStream::StdFileInputStream(const std::string& path)
    : FileInputStream(path)
{
    fp_ = std::fopen(path.c_str(), "rb");
    if(fp_ == nullptr) throw std::runtime_error("Could not open file " + path + ": " + std::strerror(errno));
}

void StdFileInputStream::close()
{
    if (fp_ != nullptr) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

void StdFileInputStream::seek(ptrdiff_t pos, int whence)
{
    if (std::fseek(fp_, pos, whence))
        throw std::runtime_error(std::string("Seek operation failed: ") + std::strerror(errno));
}

int64_t StdFileInputStream::tell() const
{
    return std::ftell(fp_);
}

std::size_t StdFileInputStream::read(void* buffer, size_t n_bytes)
{
    size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
    return n_read;
}

std::shared_ptr<void> StdFileInputStream::get(size_t /*n_bytes*/)
{
    // this unction should return a pointer inside mmaped file
    // it doesn't make sense in case of StdFileInputStream
    return {};
}

std::size_t StdFileInputStream::size() const
{
    struct stat sb;
    if (stat(path_.c_str(), &sb) == -1) {
        throw std::runtime_error("Unable to stat file " + path_ + ": " + std::strerror(errno));
    }
    return sb.st_size;
}
} //namespace nvimgcdcs 