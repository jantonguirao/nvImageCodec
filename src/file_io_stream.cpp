/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "std_file_io_stream.h"

//#include "mmaped_file_io_stream.h"
#include <cassert>

namespace nvimgcodec {

std::unique_ptr<FileIoStream> FileIoStream::open(
    const std::string& uri, bool read_ahead, bool use_mmap, bool to_write)
{
    std::string processed_uri;

    if (uri.find("file://") == 0) {
        processed_uri = uri.substr(std::string("file://").size());
    } else {
        processed_uri = uri;
    }

    if (use_mmap) {
        assert(!"TODO");
        return std::unique_ptr<FileIoStream>(new StdFileIoStream(processed_uri, to_write));
        // return std::unique_ptr<FileIoStream>(new MmapedFileIoStream(processed_uri,
        // read_ahead));
    } else {
        return std::unique_ptr<FileIoStream>(new StdFileIoStream(processed_uri, to_write));
    }
}

bool FileIoStream::reserveFileMappings(unsigned int num)
{
    return false;
    //MmapedFileIoStream::reserveFileMappings(num);
}

void FileIoStream::freeFileMappings(unsigned int num)
{
    //MmapedFileIoStream::freeFileMappings(num);
}

} // namespace nvimgcodec