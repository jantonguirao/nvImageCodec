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
#include <nvimgcodec.h>
#include <memory>
#include <string>

namespace nvimgcodec {

class CodecRegistry;
class ICodec;

class ICodeStream
{
  public:
    virtual ~ICodeStream() = default;
    virtual void parseFromFile(const std::string& file_name) = 0;
    virtual void parseFromMem(const unsigned char* data, size_t size) = 0;
    virtual void setOutputToFile(const char* file_name) = 0;
    virtual void setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t get_buffer_func) = 0;
    virtual nvimgcodecStatus_t getImageInfo(nvimgcodecImageInfo_t* image_info) = 0;
    virtual nvimgcodecStatus_t setImageInfo(const nvimgcodecImageInfo_t* image_info) = 0;
    virtual std::string getCodecName() const = 0;
    virtual ICodec* getCodec() const = 0;
    virtual nvimgcodecIoStreamDesc_t* getInputStreamDesc() = 0;
    virtual nvimgcodecCodeStreamDesc_t* getCodeStreamDesc() = 0;
};
} // namespace nvimgcodec