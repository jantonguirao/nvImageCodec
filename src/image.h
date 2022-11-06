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

namespace nvimgcdcs {
class DecodeState;
class EncodeState;

class Image
{
  public:
    explicit Image(nvimgcdcsImageInfo_t* image_info);
    ~Image();
    void setHostBuffer(void* buffer, size_t size);
    void getHostBuffer(void** buffer, size_t* size);
    void setDeviceBuffer(void* buffer, size_t size);
    void getDeviceBuffer(void** buffer, size_t* size);
    void setImageInfo(const nvimgcdcsImageInfo_t* image_info);
    void getImageInfo(nvimgcdcsImageInfo_t* image_info);
    void attachDecodeState(DecodeState* decode_state);
    DecodeState* getAttachedDecodeState();
    void detachDecodeState();
    void attachEncodeState(EncodeState* encode_state);
    EncodeState* getAttachedEncodeState();
    void detachEncodeState();
    nvimgcdcsImageDesc* getImageDesc();

  private:
    static nvimgcdcsStatus_t getImageInfo(void* instance, nvimgcdcsImageInfo_t* result);
    static nvimgcdcsStatus_t getDeviceBuffer(void* instance, void** buffer, size_t* size);
    static nvimgcdcsStatus_t getHostBuffer(void* instance, void** buffer, size_t* size);
    nvimgcdcsImageInfo_t image_info_;
    void* host_buffer_;
    size_t host_buffer_size_;
    void* device_buffer_;
    size_t device_buffer_size_;
    DecodeState* decode_state_;
    EncodeState* encode_state_;
    nvimgcdcsImageDesc image_desc_;
};
} // namespace nvimgcdcs