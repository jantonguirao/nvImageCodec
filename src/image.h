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

#include <memory>
#include <nvimgcodecs.h>
#include <nvimgcodecs.h>
#include "iimage.h"

namespace nvimgcdcs {
class IDecodeState;
class IEncodeState;

class Image : public IImage
{
  public:
    explicit Image();
    ~Image() override;
    void setIndex(int index) override;
    void setImageInfo(const nvimgcdcsImageInfo_t* image_info) override;
    void getImageInfo(nvimgcdcsImageInfo_t* image_info) override;
    nvimgcdcsImageDesc_t* getImageDesc() override;
    void setPromise(const ProcessingResultsPromise& promise) override;
  private:
    nvimgcdcsStatus_t imageReady(nvimgcdcsProcessingStatus_t processing_status);

    static nvimgcdcsStatus_t static_get_image_info(void* instance, nvimgcdcsImageInfo_t* result);
    static nvimgcdcsStatus_t static_image_ready(
        void* instance, nvimgcdcsProcessingStatus_t processing_status);
    int index_;
    nvimgcdcsImageInfo_t image_info_;
    IDecodeState* decode_state_;
    IEncodeState* encode_state_;
    nvimgcdcsImageDesc_t image_desc_;
    std::unique_ptr<ProcessingResultsPromise> promise_;
};
} // namespace nvimgcdcs
