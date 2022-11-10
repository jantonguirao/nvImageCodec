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
#include <iostream>

namespace nvimgcdcs {

Image::Image(ThreadSafeQueue<nvimgcdcsImageDesc_t>* ready_images_queue)
    : image_info_()
    , ready_images_queue_(ready_images_queue)
    , host_buffer_(nullptr)
    , host_buffer_size_(0)
    , device_buffer_(nullptr)
    , device_buffer_size_(0)
    , encode_state_(nullptr)
    , decode_state_(nullptr)
    , image_desc_{this, Image::static_get_image_info, Image::static_get_device_buffer,
          Image::static_get_host_buffer, Image::static_image_ready}
    , processing_status_(NVIMGCDCS_PROCESSING_STATUS_UNKNOWN)
{
    memset(&image_info_, 0, sizeof(image_info_));
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

nvimgcdcsImageDesc_t Image::getImageDesc()
{
    return &image_desc_;
}

void Image::setProcessingStatus(nvimgcdcsProcessingStatus_t processing_status)
{
    processing_status_ = processing_status;
}

nvimgcdcsProcessingStatus_t Image::getProcessingStatus() const
{
    return processing_status_;
}

nvimgcdcsStatus_t Image::imageReady(nvimgcdcsProcessingStatus_t processing_status)
{
    setProcessingStatus(processing_status);
    ready_images_queue_->push(&image_desc_);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getImageInfo(result);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::static_get_device_buffer(void* instance, void** buffer, size_t* size)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getDeviceBuffer(buffer, size);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::static_get_host_buffer(void* instance, void** buffer, size_t* size)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getHostBuffer(buffer, size);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::static_image_ready(
    void* instance, nvimgcdcsProcessingStatus_t processing_status)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->imageReady(processing_status);
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs