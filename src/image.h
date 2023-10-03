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
#include <nvimgcodec.h>
#include <nvimgcodec.h>
#include "iimage.h"

namespace nvimgcodec {
class IDecodeState;
class IEncodeState;

class Image : public IImage
{
  public:
    explicit Image();
    ~Image() override;
    void setIndex(int index) override;
    void setImageInfo(const nvimgcodecImageInfo_t* image_info) override;
    void getImageInfo(nvimgcodecImageInfo_t* image_info) override;
    nvimgcodecImageDesc_t* getImageDesc() override;
    void setPromise(const ProcessingResultsPromise& promise) override;
  private:
    nvimgcodecStatus_t imageReady(nvimgcodecProcessingStatus_t processing_status);

    static nvimgcodecStatus_t static_get_image_info(void* instance, nvimgcodecImageInfo_t* result);
    static nvimgcodecStatus_t static_image_ready(
        void* instance, nvimgcodecProcessingStatus_t processing_status);
    int index_;
    nvimgcodecImageInfo_t image_info_;
    IDecodeState* decode_state_;
    IEncodeState* encode_state_;
    nvimgcodecImageDesc_t image_desc_;
    std::unique_ptr<ProcessingResultsPromise> promise_;
};
} // namespace nvimgcodec
