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
#include <cstring>
#include <iostream>
#include "idecode_state.h"
#include "iencode_state.h"
#include "processing_results.h"

namespace nvimgcdcs {

Image::Image()
    : index_(0)
    , image_info_{}
    , decode_state_(nullptr)
    , encode_state_(nullptr)
    , image_desc_{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_DESC, nullptr, this, Image::static_get_image_info,
          Image::static_image_ready}
    , promise_(nullptr)
{
}

Image::~Image()
{
}

void Image::setIndex(int index)
{
    index_ = index;
}

void Image::setImageInfo(const nvimgcdcsImageInfo_t* image_info)
{
    image_info_ = *image_info;
}
void Image::getImageInfo(nvimgcdcsImageInfo_t* image_info)
{
    *image_info = image_info_;
}

nvimgcdcsImageDesc_t Image::getImageDesc()
{
    return &image_desc_;
}

void Image::setPromise(const ProcessingResultsPromise& promise)
{
    promise_ = std::make_unique<ProcessingResultsPromise>(promise);
}

nvimgcdcsStatus_t Image::imageReady(nvimgcdcsProcessingStatus_t processing_status)
{
    assert(promise_);
    promise_->set(index_, {processing_status, {}});
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t Image::static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result)
{
    assert(instance);
    Image* handle = reinterpret_cast<Image*>(instance);
    handle->getImageInfo(result);
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
