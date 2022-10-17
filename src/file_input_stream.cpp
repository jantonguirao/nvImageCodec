/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "input_stream.h"
//#include "mmaped_file_input_stream.h"
#include "std_file_input_stream.h"
namespace nvimgcdcs {

std::unique_ptr<FileInputStream> FileInputStream::open(const std::string& uri, bool read_ahead, bool use_mmap)
{
    std::string processed_uri;

    if (uri.find("file://") == 0) {
        processed_uri = uri.substr(std::string("file://").size());
    } else {
        processed_uri = uri;
    }

    if (use_mmap) {
        //TODO
        return std::unique_ptr<FileInputStream>(new StdFileInputStream(processed_uri));
        // return std::unique_ptr<FileInputStream>(new MmapedFileInputStream(processed_uri,
        // read_ahead));
    } else {
        return std::unique_ptr<FileInputStream>(new StdFileInputStream(processed_uri));
    }
}

bool FileInputStream::reserveFileMappings(unsigned int num)
{
    return false;
    //MmapedFileInputStream::reserveFileMappings(num);
}

void FileInputStream::freeFileMappings(unsigned int num)
{
    //MmapedFileInputStream::freeFileMappings(num);
}

} // namespace nvimgcdcs