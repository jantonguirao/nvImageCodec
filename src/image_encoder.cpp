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
#include "icode_stream.h"
#include "encode_state.h"
#include "iimage.h"

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

std::unique_ptr<IEncodeState> ImageEncoder::createEncodeState(cudaStream_t cuda_stream) const
{
    return std::make_unique<EncodeState>(encoder_desc_, encoder_, cuda_stream);
}

void ImageEncoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    encoder_desc_->getCapabilities(encoder_, capabilities, size);
}

void ImageEncoder::encode(ICodeStream* code_stream, IImage* image, nvimgcdcsEncodeParams_t* params)
{
    IEncodeState* encode_state                    = image->getAttachedEncodeState();
    nvimgcdcsEncodeState_t internal_encode_state = encode_state->getInternalEncodeState();
    nvimgcdcsImageDesc* image_desc               = image->getImageDesc();

    nvimgcdcsCodeStreamDesc* code_stream_desc = code_stream->getCodeStreamDesc();
    image->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_ENCODING);
    encoder_desc_->encode(encoder_, internal_encode_state, code_stream_desc, image_desc, params);
}

} // namespace nvimgcdcs