/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "code_stream.h"
#include <iostream>
#include <string>
#include <cstring>
#include "codec.h"
#include "codec_registry.h"
#include "image_parser.h"

    namespace nvimgcdcs {

CodeStream::CodeStream(
    ICodecRegistry* codec_registry, std::unique_ptr<IIoStreamFactory> io_stream_factory)
    : codec_registry_(codec_registry)
    , codec_name_("")
    , parser_(nullptr)
    , io_stream_factory_(std::move(io_stream_factory))
    , io_stream_(nullptr)
    , io_stream_desc_{NVIMGCDCS_STRUCTURE_TYPE_IO_STREAM_DESC, nullptr, this, read_static,
          write_static, putc_static, skip_static, seek_static, tell_static, size_static}
    , code_stream_desc_{NVIMGCDCS_STRUCTURE_TYPE_CODE_STREAM_DESC, nullptr, this, &io_stream_desc_, nullptr, static_get_codec_name,
          static_get_image_info}
    , image_info_(nullptr)
    , parse_state_(nullptr)
{
}

void CodeStream::parse()
{
    auto parser = codec_registry_->getParser(&code_stream_desc_);
    if (!parser)
        throw std::runtime_error("Could not match parser");
    parser_                       = std::move(parser);
    codec_name_                   = parser_->getCodecName();
    parse_state_                  = parser_->createParseState();
    code_stream_desc_.parse_state = parse_state_->getInternalParseState();
}

void CodeStream::parseFromFile(const std::string& file_name)
{
    io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, false, false);
    parse();
}

void CodeStream::parseFromMem(const unsigned char* data, size_t size)
{
    io_stream_ = io_stream_factory_->createMemIoStream(data, size);
    parse();
}
void CodeStream::setOutputToFile(const char* file_name, const char* codec_name)
{
    io_stream_  = io_stream_factory_->createFileIoStream(file_name, false, false, true);
    codec_      = codec_registry_->getCodecByName(codec_name);
    codec_name_ = std::string(codec_name);
}

void CodeStream::setOutputToHostMem(
    unsigned char* output_buffer, size_t size, const char* codec_name)
{
    io_stream_ = io_stream_factory_->createMemIoStream(output_buffer, size);
    codec_     = codec_registry_->getCodecByName(codec_name);
    codec_name_ = std::string(codec_name);
}

void CodeStream::getImageInfo(nvimgcdcsImageInfo_t* image_info)
{
    assert(image_info);
    if (!image_info_) {
        assert(parser_);
        image_info_ = std::make_unique<nvimgcdcsImageInfo_t>();
        parser_->getImageInfo(&code_stream_desc_, image_info_.get());
    }
    *image_info = *image_info_.get();
}

void CodeStream::setImageInfo(const nvimgcdcsImageInfo_t* image_info)
{
    if (!image_info_) {
        image_info_ = std::make_unique<nvimgcdcsImageInfo_t>();
    }
    *image_info_.get() = *image_info;
}
std::string CodeStream::getCodecName() const
{
    return codec_name_;
}
ICodec* CodeStream::getCodec() const
{
    return codec_registry_->getCodecByName(codec_name_.c_str());
}

nvimgcdcsIOStreamDesc* CodeStream::getInputStreamDesc()
{
    return &io_stream_desc_;
}

nvimgcdcsCodeStreamDesc* CodeStream::getCodeStreamDesc()
{
    return &code_stream_desc_;
}

nvimgcdcsStatus_t CodeStream::read(size_t* output_size, void* buf, size_t bytes)
{
    assert(io_stream_);
    *output_size = io_stream_->read(buf, bytes);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t CodeStream::write(size_t* output_size, void* buf, size_t bytes)
{
    assert(io_stream_);
    *output_size = io_stream_->write(buf, bytes);
    return NVIMGCDCS_STATUS_SUCCESS;
}
nvimgcdcsStatus_t CodeStream::putc(size_t* output_size, unsigned char ch)
{
    assert(io_stream_);
    *output_size = io_stream_->putc(ch);

    return *output_size == 1 ? NVIMGCDCS_STATUS_SUCCESS : NVIMGCDCS_STATUS_BAD_CODESTREAM;
}

nvimgcdcsStatus_t CodeStream::skip(size_t count)
{
    assert(io_stream_);
    io_stream_->skip(count);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t CodeStream::seek(size_t offset, int whence)
{
    assert(io_stream_);
    io_stream_->seek(offset, whence);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t CodeStream::tell(size_t* offset)
{
    assert(io_stream_);
    *offset = io_stream_->tell();
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t CodeStream::size(size_t* size)
{
    assert(io_stream_);
    *size = io_stream_->size();
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t CodeStream::read_static(
    void* instance, size_t* output_size, void* buf, size_t bytes)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->read(output_size, buf, bytes);
}

nvimgcdcsStatus_t CodeStream::write_static(
    void* instance, size_t* output_size, void* buf, size_t bytes)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->write(output_size, buf, bytes);
}

nvimgcdcsStatus_t CodeStream::putc_static(void* instance, size_t* output_size, unsigned char ch)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->putc(output_size, ch);
}

nvimgcdcsStatus_t CodeStream::skip_static(void* instance, size_t count)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->skip(count);
}

nvimgcdcsStatus_t CodeStream::seek_static(void* instance, size_t offset, int whence)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->seek(offset, whence);
}

nvimgcdcsStatus_t CodeStream::tell_static(void* instance, size_t* offset)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->tell(offset);
}

nvimgcdcsStatus_t CodeStream::size_static(void* instance, size_t* size)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->size(size);
}

nvimgcdcsStatus_t CodeStream::static_get_codec_name(void* instance, char* codec_name)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    std::string str = handle->getCodecName();
    strcpy(codec_name, str.c_str());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t CodeStream::static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    handle->getImageInfo(result);
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs