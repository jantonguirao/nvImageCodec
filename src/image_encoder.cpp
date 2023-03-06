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
#include "encode_state.h"
#include "encode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"

namespace nvimgcdcs {

ImageEncoder::ImageEncoder(
    const nvimgcdcsEncoderDesc_t desc, const nvimgcdcsEncodeParams_t* params)
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

std::unique_ptr<IEncodeState> ImageEncoder::createEncodeStateBatch(cudaStream_t cuda_stream) const
{
    return std::make_unique<EncodeStateBatch>(encoder_desc_, encoder_, cuda_stream);
}

void ImageEncoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    encoder_desc_->getCapabilities(encoder_, capabilities, size);
}

bool ImageEncoder::canEncode(nvimgcdcsImageDesc_t image, nvimgcdcsCodeStreamDesc_t code_stream,
    const nvimgcdcsEncodeParams_t* params) const
{
    bool result = false;
    encoder_desc_->canEncode(encoder_, &result, image, code_stream, params);
    return result;
}

void ImageEncoder::canEncode(const std::vector<IImage*>& images,
    const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params,
    std::vector<bool>* result) const
{
    result->clear();
    for (size_t i = 0; i < code_streams.size(); ++i) {
        bool r;
        nvimgcdcsCodeStreamDesc* cs_desc = code_streams[i]->getCodeStreamDesc();
        nvimgcdcsImageDesc* im_desc = images[i]->getImageDesc();
        encoder_desc_->canEncode(encoder_, &r, im_desc, cs_desc, params);
        result->push_back(r);
    }
}

std::unique_ptr<ProcessingResultsFuture> ImageEncoder::encode(
    ICodeStream* code_stream, IImage* image, const nvimgcdcsEncodeParams_t* params)
{
    ProcessingResultsPromise results(1);
    auto future = results.getFuture();
    IEncodeState* encode_state = image->getAttachedEncodeState();
    encode_state->setPromise(std::move(results));

    nvimgcdcsEncodeState_t internal_encode_state = encode_state->getInternalEncodeState();
    nvimgcdcsImageDesc* image_desc = image->getImageDesc();

    nvimgcdcsCodeStreamDesc* code_stream_desc = code_stream->getCodeStreamDesc();
    image->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_ENCODING);
    image->setPromise(encode_state->getPromise());
    encoder_desc_->encode(encoder_, internal_encode_state, image_desc, code_stream_desc, params);

    return future;
}

std::unique_ptr<ProcessingResultsFuture> ImageEncoder::encodeBatch(IEncodeState* encode_state_batch,
    const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    const nvimgcdcsEncodeParams_t* params)
{
    assert(code_streams.size() == images.size());

    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();
    encode_state_batch->setPromise(std::move(results));

    if (encoder_desc_->encodeBatch == nullptr) {
        for (size_t i = 0; i < code_streams.size(); ++i) {
            IEncodeState* encode_state = images[i]->getAttachedEncodeState();
            nvimgcdcsEncodeState_t internal_encode_state = encode_state->getInternalEncodeState();
            nvimgcdcsImageDesc* image_desc = images[i]->getImageDesc();
            nvimgcdcsCodeStreamDesc* code_stream_desc = code_streams[i]->getCodeStreamDesc();
            images[i]->setIndex(i);
            images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
            images[i]->setPromise(encode_state_batch->getPromise());
            encoder_desc_->encode(
                encoder_, internal_encode_state, image_desc, code_stream_desc, params);
        }
    } else {

        std::vector<nvimgcdcsEncodeState_t> encode_states;
        std::vector<nvimgcdcsCodeStreamDesc*> code_stream_descs;
        std::vector<nvimgcdcsImageDesc*> image_descs;

        for (size_t i = 0; i < code_streams.size(); ++i) {

            code_stream_descs.push_back(code_streams[i]->getCodeStreamDesc());
            IEncodeState* encode_state = images[i]->getAttachedEncodeState();
            encode_states.push_back(encode_state->getInternalEncodeState());
            image_descs.push_back(images[i]->getImageDesc());
            images[i]->setIndex(i);
            images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
            images[i]->setPromise(encode_state_batch->getPromise());
        }

        encoder_desc_->encodeBatch(encoder_, encode_state_batch->getInternalEncodeState(),
            encode_states.data(), image_descs.data(), code_stream_descs.data(), code_streams.size(),
            params);
    }

    return future;
}

} // namespace nvimgcdcs
