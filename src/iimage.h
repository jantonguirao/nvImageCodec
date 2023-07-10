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

namespace nvimgcdcs {
class IDecodeState;
class IEncodeState;
class ProcessingResultsPromise;

class IImage
{
  public:
    virtual ~IImage() = default;
    virtual void setIndex(int index) = 0;
    virtual void setImageInfo(const nvimgcdcsImageInfo_t* image_info) = 0;
    virtual void getImageInfo(nvimgcdcsImageInfo_t* image_info) = 0;
    virtual nvimgcdcsImageDesc_t* getImageDesc() = 0;
    virtual void setPromise(const ProcessingResultsPromise& promise) = 0;
};
} // namespace nvimgcdcs
