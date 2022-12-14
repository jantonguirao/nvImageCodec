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
#include "iimage.h"
#include "icode_stream.h"

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

std::unique_ptr<IDecodeState> ImageDecoder::createDecodeState() const
{
    return std::make_unique<DecodeState>(decoder_desc_, decoder_);
}

void ImageDecoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    decoder_desc_->getCapabilities(decoder_, capabilities, size);
}

void ImageDecoder::decode(ICodeStream* code_stream, IImage* image, nvimgcdcsDecodeParams_t* params)
{
    IDecodeState* decode_state                    = image->getAttachedDecodeState();
    nvimgcdcsDecodeState_t internal_decode_state = decode_state->getInternalDecodeState();
    nvimgcdcsImageDesc* image_desc               = image->getImageDesc();

    nvimgcdcsCodeStreamDesc* code_stream_desc = code_stream->getCodeStreamDesc();
    image->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);

    decoder_desc_->decode(decoder_, internal_decode_state, code_stream_desc, image_desc, params);
}

} // namespace nvimgcdcs