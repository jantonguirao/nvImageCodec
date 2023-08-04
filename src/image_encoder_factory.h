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
#include <memory>
#include <string>
#include "iimage_encoder_factory.h"

namespace nvimgcdcs {

class IImageEncoder;

class ImageEncoderFactory : public IImageEncoderFactory
{
  public:
    explicit ImageEncoderFactory(const nvimgcdcsEncoderDesc_t* desc);
    std::string getEncoderId() const override;
    std::string getCodecName() const override;
    nvimgcdcsBackendKind_t getBackendKind() const override;
    std::unique_ptr<IImageEncoder> createEncoder(
        const nvimgcdcsExecutionParams_t* exec_params, const char* options) const override;

  private:
    const nvimgcdcsEncoderDesc_t* encoder_desc_;
};
} // namespace nvimgcdcs