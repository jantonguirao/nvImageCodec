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
#include "iimage_encoder_factory.h"

namespace nvimgcodec {

class IImageEncoder;

class ImageEncoderFactory : public IImageEncoderFactory
{
  public:
    explicit ImageEncoderFactory(const nvimgcodecEncoderDesc_t* desc);
    std::string getEncoderId() const override;
    std::string getCodecName() const override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    std::unique_ptr<IImageEncoder> createEncoder(
        const nvimgcodecExecutionParams_t* exec_params, const char* options) const override;

  private:
    const nvimgcodecEncoderDesc_t* encoder_desc_;
};
} // namespace nvimgcodec