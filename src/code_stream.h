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
    void setOutputToFile(const char* file_name) override;
    void setOutputToHostMem(void* ctx, nvimgcdcsResizeBufferFunc_t get_buffer_func) override;
    nvimgcdcsStatus_t getImageInfo(nvimgcdcsImageInfo_t* image_info) override;
    nvimgcdcsStatus_t setImageInfo(const nvimgcdcsImageInfo_t* image_info) override;
    std::string getCodecName() const override;
    ICodec* getCodec() const override;
    nvimgcdcsIoStreamDesc_t* getInputStreamDesc() override;
    nvimgcdcsCodeStreamDesc_t* getCodeStreamDesc() override;

  private:
    void parse();
    nvimgcdcsStatus_t read(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsStatus_t write(size_t* output_size, void* buf, size_t bytes);
    nvimgcdcsStatus_t putc(size_t* output_size, unsigned char ch);
    nvimgcdcsStatus_t skip(size_t count);
    nvimgcdcsStatus_t seek(ptrdiff_t offset, int whence);
    nvimgcdcsStatus_t tell(ptrdiff_t* offset);
    nvimgcdcsStatus_t size(size_t* size);
    nvimgcdcsStatus_t reserve(size_t bytes);
    nvimgcdcsStatus_t flush();
    nvimgcdcsStatus_t map(void** addr, size_t offset, size_t size);
    nvimgcdcsStatus_t unmap(void* addr, size_t size);

    static nvimgcdcsStatus_t read_static(void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsStatus_t write_static(
        void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcdcsStatus_t putc_static(void* instance, size_t* output_size, unsigned char ch);
    static nvimgcdcsStatus_t skip_static(void* instance, size_t count);
    static nvimgcdcsStatus_t seek_static(void* instance, ptrdiff_t offset, int whence);
    static nvimgcdcsStatus_t tell_static(void* instance, ptrdiff_t* offset);
    static nvimgcdcsStatus_t size_static(void* instance, size_t* size);
    static nvimgcdcsStatus_t reserve_static(void* instance, size_t bytes);
    static nvimgcdcsStatus_t flush_static(void* instance);
    static nvimgcdcsStatus_t map_static(void* instance, void** addr, size_t offset, size_t size);
    static nvimgcdcsStatus_t unmap_static(void* instance, void* addr, size_t size);

    static nvimgcdcsStatus_t static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result);

    ICodecRegistry* codec_registry_;
    std::unique_ptr<IImageParser> parser_;
    std::unique_ptr<IIoStreamFactory> io_stream_factory_;
    std::unique_ptr<IoStream> io_stream_;
    nvimgcdcsIoStreamDesc_t io_stream_desc_;
    nvimgcdcsCodeStreamDesc_t code_stream_desc_;
    std::unique_ptr<nvimgcdcsImageInfo_t> image_info_;
};
} // namespace nvimgcdcs