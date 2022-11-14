/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_decoder.h"
#include <cassert>
#include "decode_state.h"
#include "image.h"
#include "code_stream.h"

namespace nvimgcdcs {

ImageDecoder::ImageDecoder(const struct nvimgcdcsDecoderDesc* desc, nvimgcdcsDecodeParams_t* params)
    : decoder_desc_(desc)
{
    decoder_desc_->create(decoder_desc_->instance, &decoder_, params);
}

ImageDecoder::~ImageDecoder()
{
    decoder_desc_->destroy(decoder_);
}

std::unique_ptr<DecodeState> ImageDecoder::createDecodeState() const
{
    return std::make_unique<DecodeState>(decoder_desc_, decoder_);
}

void ImageDecoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    decoder_desc_->getCapabilities(decoder_, capabilities, size);
}

void ImageDecoder::decode(CodeStream* code_stream, Image* image, nvimgcdcsDecodeParams_t* params)
{
    DecodeState* decode_state                    = image->getAttachedDecodeState();
    nvimgcdcsDecodeState_t internal_decode_state = decode_state->getInternalDecodeState();
    nvimgcdcsImageDesc* image_desc               = image->getImageDesc();

    nvimgcdcsCodeStreamDesc* code_stream_desc = code_stream->getCodeStreamDesc();
    image->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);

    decoder_desc_->decode(decoder_, internal_decode_state, code_stream_desc, image_desc, params);
}


ImageDecoderFactory::ImageDecoderFactory(const struct nvimgcdcsDecoderDesc* desc)
    : decoder_desc_(desc)
{
}

std::string ImageDecoderFactory::getDecoderId() const
{
    return decoder_desc_->id;
}

std::string ImageDecoderFactory::getCodecName() const
{
    return decoder_desc_->codec;
}

bool ImageDecoderFactory::canDecode(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params)
{
    assert(code_stream);
    assert(decoder_desc_->canDecode);
    bool result = false;
    decoder_desc_->canDecode(decoder_desc_->instance, &result, code_stream, params);
    return result;
}

std::unique_ptr<ImageDecoder> ImageDecoderFactory::createDecoder(
    nvimgcdcsDecodeParams_t* params) const
{
    return std::make_unique<ImageDecoder>(decoder_desc_, params);
}

} // namespace nvimgcdcs