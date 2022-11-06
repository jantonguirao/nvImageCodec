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
#include "io_stream.h"
#include "parse_state.h"

namespace nvimgcdcs {
class CodecRegistry;
class Codec;
class ImageParser;
class CodeStream
{
  public:
    explicit CodeStream(CodecRegistry* codec_registry);
    void parseFromFile(const std::string& file_name);
    void parseFromMem(unsigned char* data, size_t size);
    void setOutputToFile(const char* file_name, const char* codec_name);
    void setOutputToHostMem(unsigned char* output_buffer, size_t size, const char* codec_name);
    void getImageInfo(nvimgcdcsImageInfo_t* image_info);
    void setImageInfo(const nvimgcdcsImageInfo_t* image_info);
    std::string getCodecName() const;
    Codec* getCodec() const;
    nvimgcdcsInputStreamDesc* getInputStreamDesc();
    nvimgcdcsCodeStreamDesc* getCodeStreamDesc();

  private:
    void parse();
    nvimgcdcsStatus_t read(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsStatus_t write(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsStatus_t putc(size_t* output_size, unsigned char ch);
    nvimgcdcsStatus_t skip(size_t count);
    nvimgcdcsStatus_t seek(size_t offset, int whence);
    nvimgcdcsStatus_t tell(size_t* offset);
    nvimgcdcsStatus_t size(size_t* size);

    static nvimgcdcsStatus_t read_static(void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsStatus_t write_static(
        void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsStatus_t putc_static(void* instance, size_t* output_size, unsigned char ch);
    static nvimgcdcsStatus_t skip_static(void* instance, size_t count);
    static nvimgcdcsStatus_t seek_static(void* instance, size_t offset, int whence);
    static nvimgcdcsStatus_t tell_static(void* instance, size_t* offset);
    static nvimgcdcsStatus_t size_static(void* instance, size_t* size);

    CodecRegistry* codec_registry_;
    Codec* codec_;
    std::unique_ptr<ImageParser> parser_;
    std::unique_ptr<IoStream> io_stream_;
    nvimgcdcsInputStreamDesc io_stream_desc_;
    nvimgcdcsCodeStreamDesc code_stream_desc_;
    std::unique_ptr<nvimgcdcsImageInfo_t> image_info_;
    std::unique_ptr<ParseState> parse_state_;
};
} // namespace nvimgcdcs