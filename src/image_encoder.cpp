/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_encoder.h"

namespace nvimgcdcs {

ImageEncoderFactory::ImageEncoderFactory(const struct nvimgcdcsEncoderDesc *desc)
    : encoder_desc_(desc)
{
}

const std::string ImageEncoderFactory::getCodecName() const { return encoder_desc_->codec; }

} // namespace nvimgcdcs