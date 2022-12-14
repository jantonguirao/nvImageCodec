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
class IDecodeState;
class IEncodeState;

class IImage
{
  public:
    virtual ~IImage() = default;
    virtual void setHostBuffer(void* buffer, size_t size) = 0;
    virtual void getHostBuffer(void** buffer, size_t* size) = 0;
    virtual void setDeviceBuffer(void* buffer, size_t size) = 0;
    virtual void getDeviceBuffer(void** buffer, size_t* size) = 0;
    virtual void setImageInfo(const nvimgcdcsImageInfo_t* image_info) = 0;
    virtual void getImageInfo(nvimgcdcsImageInfo_t* image_info) = 0;
    virtual void attachDecodeState(IDecodeState* decode_state) = 0;
    virtual IDecodeState* getAttachedDecodeState() = 0;
    virtual void detachDecodeState() = 0;
    virtual void attachEncodeState(IEncodeState* encode_state) = 0;
    virtual IEncodeState* getAttachedEncodeState() = 0;
    virtual void detachEncodeState() = 0;
    virtual nvimgcdcsImageDesc_t getImageDesc() = 0;
    virtual void setProcessingStatus(nvimgcdcsProcessingStatus_t processing_status) = 0;
    virtual nvimgcdcsProcessingStatus_t getProcessingStatus() const = 0;

};
} // namespace nvimgcdcs