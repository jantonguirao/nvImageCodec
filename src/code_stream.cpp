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
#include "codec.h"
#include "codec_registry.h"
#include "file_input_stream.h"
#include "mem_input_stream.h"
#include "image_parser.h"
#include "parse_state.h"
#include <string>
#include <iostream>
namespace nvimgcdcs {

CodeStream::CodeStream(CodecRegistry* codec_registry)
    : codec_registry_(codec_registry)
    , input_stream_desc_{this, read_static, skip_static, seek_static, tell_static, size_static}
    , code_stream_desc_{this, "", &input_stream_desc_}
    , codec_(nullptr)
    , parser_(nullptr)
{
}

void CodeStream::parse()
{
    auto [codec, parser] = codec_registry_->getCodecAndParser(&code_stream_desc_);
    if (codec == nullptr || !parser)
        throw std::runtime_error("Could not match parser");
    codec_  = codec;
    code_stream_desc_.codec = codec->name().c_str();
    parser_ = std::move(parser);
    parse_state_ = parser_->createParseState();
    code_stream_desc_.parse_state = parse_state_->getInternalParseState();
}

void CodeStream::parseFromFile(const std::string& file_name)
{
    input_stream_ = FileInputStream::open(file_name, false, false);
    parse();
}

void CodeStream::parseFromMem(const unsigned char* data, size_t size)
{
    input_stream_ = std::make_unique<MemInputStream>(data, size);
    parse();
}

void CodeStream::getImageInfo(nvimgcdcsImageInfo_t* image_info)
{
    assert(parser_);
    assert(image_info);
    if (!image_info_) {
        image_info_ = std::make_unique<nvimgcdcsImageInfo_t>();
        parser_->getImageInfo(&code_stream_desc_, image_info_.get());
    }
    *image_info = *image_info_.get();
}

Codec* CodeStream::getCodec() const
{
    return codec_;
}

nvimgcdcsInputStreamDesc* CodeStream::getInputStreamDesc()
{
    return &input_stream_desc_;
}

nvimgcdcsCodeStreamDesc* CodeStream::getCodeStreamDesc()
{
    return &code_stream_desc_;
}

nvimgcdcsParserStatus_t CodeStream::read(size_t* output_size, void* buf, size_t bytes)
{
    assert(input_stream_);
    *output_size = input_stream_->read(buf, bytes);
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}
nvimgcdcsParserStatus_t CodeStream::skip(size_t count)
{
    assert(input_stream_);
    input_stream_->skip(count);
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

nvimgcdcsParserStatus_t CodeStream::seek(size_t offset, int whence)
{
    assert(input_stream_);
    input_stream_->seek(offset, whence);
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

nvimgcdcsParserStatus_t CodeStream::tell(size_t* offset)
{
    assert(input_stream_);
    *offset = input_stream_->tell();
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

nvimgcdcsParserStatus_t CodeStream::size(size_t* size)
{
    assert(input_stream_);
    *size = input_stream_->size();
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

nvimgcdcsParserStatus_t CodeStream::read_static(void* instance, size_t* output_size, void* buf, size_t bytes)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->read(output_size, buf, bytes);
}
nvimgcdcsParserStatus_t CodeStream::skip_static(void* instance, size_t count)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->skip(count);
}
nvimgcdcsParserStatus_t CodeStream::seek_static(void* instance, size_t offset, int whence)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->seek(offset, whence);
}
nvimgcdcsParserStatus_t CodeStream::tell_static(void* instance, size_t* offset)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->tell(offset);
}
nvimgcdcsParserStatus_t CodeStream::size_static(void* instance, size_t* size)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->size(size);
}

} // namespace nvimgcdcs