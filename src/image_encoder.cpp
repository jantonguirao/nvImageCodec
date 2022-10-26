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
#include <cassert>
#include "code_stream.h"
#include "encode_state.h"
#include "image.h"

namespace nvimgcdcs {

ImageEncoder::ImageEncoder(const struct nvimgcdcsEncoderDesc* desc, nvimgcdcsEncodeParams_t* params)
    : encoder_desc_(desc)
{
    encoder_desc_->create(encoder_desc_->instance, &encoder_, params);
}

ImageEncoder::~ImageEncoder()
{
    encoder_desc_->destroy(encoder_);
}

std::unique_ptr<EncodeState> ImageEncoder::createEncodeState() const
{
    return std::make_unique<EncodeState>(encoder_desc_, encoder_);
}

void ImageEncoder::encode(CodeStream* code_stream, Image* image, nvimgcdcsEncodeParams_t* params)
{
    EncodeState* encode_state                    = image->getAttachedEncodeState();
    nvimgcdcsEncodeState_t internal_encode_state = encode_state->getInternalEncodeState();
    nvimgcdcsImageDesc* image_desc               = image->getImageDesc();

    nvimgcdcsCodeStreamDesc* code_stream_desc = code_stream->getCodeStreamDesc();

    encoder_desc_->encode(encoder_, internal_encode_state, code_stream_desc, image_desc, params);
}

ImageEncoderFactory::ImageEncoderFactory(const struct nvimgcdcsEncoderDesc* desc)
    : encoder_desc_(desc)
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

bool ImageEncoderFactory::canEncode(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params)
{
    assert(code_stream);
    assert(encoder_desc_->canEncode);
    bool result = false;
    encoder_desc_->canEncode(encoder_desc_->instance, &result, code_stream, params);
    return result;
}

std::unique_ptr<ImageEncoder> ImageEncoderFactory::createEncoder(
    nvimgcdcsEncodeParams_t* params) const
{
    return std::make_unique<ImageEncoder>(encoder_desc_, params);
}
} // namespace nvimgcdcs