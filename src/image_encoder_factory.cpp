/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_encoder_factory.h"
#include <cassert>
#include "code_stream.h"
#include "encode_state.h"
#include "image.h"
#include "image_encoder.h"
namespace nvimgcdcs {

ImageEncoderFactory::ImageEncoderFactory(const nvimgcdcsEncoderDesc_t desc)
    :  encoder_desc_(desc)
{
}

std::string ImageEncoderFactory::getCodecName() const
{
    return encoder_desc_->codec;
}

std::string ImageEncoderFactory::getEncoderId() const
{
    return encoder_desc_->id;
}

bool ImageEncoderFactory::canEncode(nvimgcdcsImageDesc_t image,
    nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params)
{
    assert(code_stream);
    assert(encoder_desc_->canEncode);
    bool result = false;
    encoder_desc_->canEncode(encoder_desc_->instance, &result, image, code_stream, params);
    return result;
}

std::unique_ptr<IImageEncoder> ImageEncoderFactory::createEncoder(
    const nvimgcdcsEncodeParams_t* params) const
{
    return std::make_unique<ImageEncoder>(encoder_desc_, params);
}
} // namespace nvimgcdcs