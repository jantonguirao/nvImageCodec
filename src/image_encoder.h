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
#include <string>

namespace nvimgcdcs {
class ImageEncoderFactory
{
  public:
    ImageEncoderFactory(const struct nvimgcdcsEncoderDesc *desc);
    const std::string getCodecName() const;

  private:
    const struct nvimgcdcsEncoderDesc *encoder_desc_;
};
} // namespace nvimgcdcs