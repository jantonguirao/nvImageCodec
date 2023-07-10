/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_decoder_factory.h"

#include <cassert>
#include "code_stream.h"
#include "image.h"
#include "image_decoder.h"

namespace nvimgcdcs {

ImageDecoderFactory::ImageDecoderFactory(const nvimgcdcsDecoderDesc_t* desc)
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

nvimgcdcsBackendKind_t ImageDecoderFactory::getBackendKind() const
{
    return decoder_desc_->backend_kind;
}

std::unique_ptr<IImageDecoder> ImageDecoderFactory::createDecoder(
    int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options) const
{
    return std::make_unique<ImageDecoder>(decoder_desc_, device_id, backend_params, options);
}

} // namespace nvimgcdcs