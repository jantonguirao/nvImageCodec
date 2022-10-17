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

#include <nvimgcodecs.h>
#include <nvimgcdcs_module.h>
#include <string>
#include <memory>
#include "input_stream.h"

namespace nvimgcdcs {
class CodecRegistry;
class Codec;
class CodeStream
{
  public:
    explicit CodeStream(CodecRegistry* codec_registry);
    void parseFromFile(const std::string& file_name);
    void parseFromMem(const unsigned char* data, size_t size);
    void getImageInfo(nvimgcdcsImageInfo_t* image_info);
    nvimgcdcsInputStreamDesc* getInputStreamDesc();

  private:
    nvimgcdcsParserStatus_t read(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsParserStatus_t skip(size_t count);
    nvimgcdcsParserStatus_t seek(size_t offset, int whence);
    nvimgcdcsParserStatus_t tell(size_t* offset);
    nvimgcdcsParserStatus_t size(size_t* size);

    static nvimgcdcsParserStatus_t read_static(void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsParserStatus_t skip_static(void* instance, size_t count);
    static nvimgcdcsParserStatus_t seek_static(void* instance, size_t offset, int whence);
    static nvimgcdcsParserStatus_t tell_static(void* instance, size_t* offset);
    static nvimgcdcsParserStatus_t size_static(void* instance, size_t* size);

    CodecRegistry* codec_registry_;
    const Codec* codec_;
    std::unique_ptr<InputStream> input_stream_;
    nvimgcdcsInputStreamDesc input_stream_desc_;
};
} // namespace nvimgcdcs