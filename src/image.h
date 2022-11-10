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

#include <nvimgcdcs_module.h>
#include <nvimgcodecs.h>
#include "thread_safe_queue.h"

namespace nvimgcdcs {
class DecodeState;
class EncodeState;

class Image
{
  public:
    explicit Image(
        nvimgcdcsImageInfo_t* image_info, ThreadSafeQueue<nvimgcdcsImageDesc_t>* ready_images_queue);
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
    nvimgcdcsImageDesc_t getImageDesc();
    void setProcessingStatus(nvimgcdcsProcessingStatus_t processing_status);
    nvimgcdcsProcessingStatus_t getProcessingStatus() const;

  private:
    nvimgcdcsStatus_t imageReady(nvimgcdcsProcessingStatus_t processing_status);

    static nvimgcdcsStatus_t static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result);
    static nvimgcdcsStatus_t static_get_device_buffer(void* instance, void** buffer, size_t* size);
    static nvimgcdcsStatus_t static_get_host_buffer(void* instance, void** buffer, size_t* size);
    static nvimgcdcsStatus_t static_image_ready(
        void* instance, nvimgcdcsProcessingStatus_t processing_status);

    nvimgcdcsImageInfo_t image_info_;
    void* host_buffer_;
    size_t host_buffer_size_;
    void* device_buffer_;
    size_t device_buffer_size_;
    DecodeState* decode_state_;
    EncodeState* encode_state_;
    nvimgcdcsImageDesc image_desc_;
    ThreadSafeQueue<nvimgcdcsImageDesc_t>* ready_images_queue_;
    nvimgcdcsProcessingStatus_t processing_status_;
};
} // namespace nvimgcdcs