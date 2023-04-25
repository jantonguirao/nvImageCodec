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
#include <string>
#include <memory>
#include "io_stream.h"
#include "iparse_state.h"
#include "iimage_parser.h"
#include "icode_stream.h"
#include "iiostream_factory.h"

namespace nvimgcdcs {

class ICodecRegistry;
class ICodec;

class CodeStream : public ICodeStream
{
  public:
    explicit CodeStream(ICodecRegistry* codec_registry, std::unique_ptr<IIoStreamFactory> io_stream_factory);
    ~CodeStream();
    void parseFromFile(const std::string& file_name) override;
    void parseFromMem(const unsigned char* data, size_t size) override;
    void setOutputToFile(const char* file_name, const char* codec_name) override;
    void setOutputToHostMem(unsigned char* output_buffer, size_t size, const char* codec_name) override;
    nvimgcdcsStatus_t getImageInfo(nvimgcdcsImageInfo_t* image_info) override;
    nvimgcdcsStatus_t setImageInfo(const nvimgcdcsImageInfo_t* image_info) override;
    std::string getCodecName() const override;
    ICodec* getCodec() const override;
    nvimgcdcsIOStreamDesc* getInputStreamDesc() override;
    nvimgcdcsCodeStreamDesc* getCodeStreamDesc() override;

  private:
    void parse();
    nvimgcdcsStatus_t read(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsStatus_t write(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsStatus_t putc(size_t* output_size, unsigned char ch);
    nvimgcdcsStatus_t skip(size_t count);
    nvimgcdcsStatus_t seek(size_t offset, int whence);
    nvimgcdcsStatus_t tell(size_t* offset);
    nvimgcdcsStatus_t size(size_t* size);
    nvimgcdcsStatus_t raw_data(const void** raw_data);

    static nvimgcdcsStatus_t read_static(void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsStatus_t write_static(
        void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsStatus_t putc_static(void* instance, size_t* output_size, unsigned char ch);
    static nvimgcdcsStatus_t skip_static(void* instance, size_t count);
    static nvimgcdcsStatus_t seek_static(void* instance, size_t offset, int whence);
    static nvimgcdcsStatus_t tell_static(void* instance, size_t* offset);
    static nvimgcdcsStatus_t size_static(void* instance, size_t* size);
    static nvimgcdcsStatus_t raw_data_static(void* instance, const void** raw_data);

    static nvimgcdcsStatus_t static_get_codec_name(void* instance, char* codec_name);
    static nvimgcdcsStatus_t static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result);

    ICodecRegistry* codec_registry_;
    ICodec* codec_;
    std::string codec_name_;
    std::unique_ptr<IImageParser> parser_;
    std::unique_ptr<IIoStreamFactory> io_stream_factory_;
    std::unique_ptr<IoStream> io_stream_;
    nvimgcdcsIOStreamDesc io_stream_desc_;
    nvimgcdcsCodeStreamDesc code_stream_desc_;
    std::unique_ptr<nvimgcdcsImageInfo_t> image_info_;
    std::unique_ptr<IParseState> parse_state_;
};
} // namespace nvimgcdcs