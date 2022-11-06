/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image.h"
#include <cassert>

namespace nvimgcdcs {

Image::Image(nvimgcdcsImageInfo_t* image_info)
    : image_info_(*image_info)
    , host_buffer_(nullptr)
    , host_buffer_size_(0)
    , device_buffer_(nullptr)
    , device_buffer_size_(0)
    , encode_state_(nullptr)
    , decode_state_(nullptr)
    , image_desc_{this, Image::getImageInfo, Image::getDeviceBuffer, Image::getHostBuffer}
{
}

Image::~Image()
{
}
void Image::setHostBuffer(void* buffer, size_t size)
{
    host_buffer_      = buffer;
    host_buffer_size_ = size;
}
void Image::getHostBuffer(void** buffer, size_t* size)
{
    *buffer = host_buffer_;
    *size   = host_buffer_size_;
}
void Image::setDeviceBuffer(void* buffer, size_t size)
{
    device_buffer_      = buffer;
    device_buffer_size_ = size;
}
void Image::getDeviceBuffer(void** buffer, size_t* size)
{
    *buffer = device_buffer_;
    *size   = device_buffer_size_;
}
void Image::setImageInfo(const nvimgcdcsImageInfo_t* image_info)
{
    image_info_ = *image_info;
}
void Image::getImageInfo(nvimgcdcsImageInfo_t* image_info)
{
    *image_info = image_info_;
}

void Image::attachDecodeState(DecodeState* decode_state)
{
    decode_state_ = decode_state;
}
void Image::detachDecodeState()
{
    decode_state_ = nullptr;
}
DecodeState* Image::getAttachedDecodeState()
{
    return decode_state_;
}

void Image::attachEncodeState(EncodeState* encode_state)
{
    encode_state_ = encode_state;
}
void Image::detachEncodeState()
{
    encode_state_ = nullptr;
}
EncodeState* Image::getAttachedEncodeState()
{
    return encode_state_;
}

nvimgcdcsImageDesc* Image::getImageDesc()
{
    return &image_desc_;
}

nvimgcdcsStatus_t Image::getImageInfo(void* instance, nvimgcdcsImageInfo_t* result)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getImageInfo(result);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::getDeviceBuffer(void* instance, void** buffer, size_t* size)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getDeviceBuffer(buffer, size);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::getHostBuffer(void* instance, void** buffer, size_t* size)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getHostBuffer(buffer, size);
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs