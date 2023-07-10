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
#include <nvimgcodecs.h>
#include <memory>
#include <string>

namespace nvimgcdcs {

class CodecRegistry;
class ICodec;

class ICodeStream
{
  public:
    virtual ~ICodeStream() = default;
    virtual void parseFromFile(const std::string& file_name) = 0;
    virtual void parseFromMem(const unsigned char* data, size_t size) = 0;
    virtual void setOutputToFile(const char* file_name) = 0;
    virtual void setOutputToHostMem(void* ctx, nvimgcdcsGetBufferFunc_t get_buffer_func) = 0;
    virtual nvimgcdcsStatus_t getImageInfo(nvimgcdcsImageInfo_t* image_info) = 0;
    virtual nvimgcdcsStatus_t setImageInfo(const nvimgcdcsImageInfo_t* image_info) = 0;
    virtual std::string getCodecName() const = 0;
    virtual ICodec* getCodec() const = 0;
    virtual nvimgcdcsIoStreamDesc_t* getInputStreamDesc() = 0;
    virtual nvimgcdcsCodeStreamDesc_t* getCodeStreamDesc() = 0;
};
} // namespace nvimgcdcs