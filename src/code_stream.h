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

#include <nvimgcodec.h>
#include <string>
#include <memory>
#include "io_stream.h"
#include "iimage_parser.h"
#include "icode_stream.h"
#include "iiostream_factory.h"

namespace nvimgcodec {

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
    void setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t get_buffer_func) override;
    nvimgcodecStatus_t getImageInfo(nvimgcodecImageInfo_t* image_info) override;
    nvimgcodecStatus_t setImageInfo(const nvimgcodecImageInfo_t* image_info) override;
    std::string getCodecName() const override;
    ICodec* getCodec() const override;
    nvimgcodecIoStreamDesc_t* getInputStreamDesc() override;
    nvimgcodecCodeStreamDesc_t* getCodeStreamDesc() override;

  private:
    void parse();
    nvimgcodecStatus_t read(size_t* output_size, void* buf, size_t bytes);
    nvimgcodecStatus_t write(size_t* output_size, void* buf, size_t bytes);
    nvimgcodecStatus_t putc(size_t* output_size, unsigned char ch);
    nvimgcodecStatus_t skip(size_t count);
    nvimgcodecStatus_t seek(ptrdiff_t offset, int whence);
    nvimgcodecStatus_t tell(ptrdiff_t* offset);
    nvimgcodecStatus_t size(size_t* size);
    nvimgcodecStatus_t reserve(size_t bytes);
    nvimgcodecStatus_t flush();
    nvimgcodecStatus_t map(void** addr, size_t offset, size_t size);
    nvimgcodecStatus_t unmap(void* addr, size_t size);

    static nvimgcodecStatus_t read_static(void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcodecStatus_t write_static(
        void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcodecStatus_t putc_static(void* instance, size_t* output_size, unsigned char ch);
    static nvimgcodecStatus_t skip_static(void* instance, size_t count);
    static nvimgcodecStatus_t seek_static(void* instance, ptrdiff_t offset, int whence);
    static nvimgcodecStatus_t tell_static(void* instance, ptrdiff_t* offset);
    static nvimgcodecStatus_t size_static(void* instance, size_t* size);
    static nvimgcodecStatus_t reserve_static(void* instance, size_t bytes);
    static nvimgcodecStatus_t flush_static(void* instance);
    static nvimgcodecStatus_t map_static(void* instance, void** addr, size_t offset, size_t size);
    static nvimgcodecStatus_t unmap_static(void* instance, void* addr, size_t size);

    static nvimgcodecStatus_t static_get_image_info(void* instance, nvimgcodecImageInfo_t* result);

    ICodecRegistry* codec_registry_;
    std::unique_ptr<IImageParser> parser_;
    std::unique_ptr<IIoStreamFactory> io_stream_factory_;
    std::unique_ptr<IoStream> io_stream_;
    nvimgcodecIoStreamDesc_t io_stream_desc_;
    nvimgcodecCodeStreamDesc_t code_stream_desc_;
    std::unique_ptr<nvimgcodecImageInfo_t> image_info_;
};
} // namespace nvimgcodec