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
#include "decode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"
#include "log.h"
#include "processing_results.h"

namespace nvimgcdcs {

ImageDecoder::ImageDecoder(
    const struct nvimgcdcsDecoderDesc* desc, const nvimgcdcsDecodeParams_t* params)
    : decoder_desc_(desc)
{
    decoder_desc_->create(decoder_desc_->instance, &decoder_, params);
}

ImageDecoder::~ImageDecoder()
{
    decoder_desc_->destroy(decoder_);
}

std::unique_ptr<IDecodeState> ImageDecoder::createDecodeState(cudaStream_t cuda_stream) const
{
    return std::make_unique<DecodeState>(decoder_desc_, decoder_, cuda_stream);
}

std::unique_ptr<IDecodeState> ImageDecoder::createDecodeStateBatch(cudaStream_t cuda_stream) const
{
    return std::make_unique<DecodeStateBatch>(decoder_desc_, decoder_, cuda_stream);
}

void ImageDecoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    decoder_desc_->getCapabilities(decoder_, capabilities, size);
}

bool ImageDecoder::canDecode(nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image,
    const nvimgcdcsDecodeParams_t* params) const
{
    bool result = false;
    decoder_desc_->canDecode(decoder_, &result, code_stream, image, params);
    return result;
}

void ImageDecoder::canDecode(const std::vector<ICodeStream*>& code_streams,
    const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params,
    std::vector<bool>* result) const
{
    result->clear();
    for (size_t i = 0; i < code_streams.size(); ++i) {
        bool r;
        nvimgcdcsCodeStreamDesc* cs_desc = code_streams[i]->getCodeStreamDesc();
        nvimgcdcsImageDesc* im_desc = images[i]->getImageDesc();
        decoder_desc_->canDecode(decoder_, &r, cs_desc, im_desc, params);
        result->push_back(r);
    }
}

std::unique_ptr<ProcessingResultsFuture> ImageDecoder::decode(
    ICodeStream* code_stream, IImage* image, const nvimgcdcsDecodeParams_t* params)
{
    ProcessingResultsPromise results(1);
    auto future = results.getFuture();
    IDecodeState* decode_state = image->getAttachedDecodeState();
    decode_state->setPromise(std::move(results));
    nvimgcdcsDecodeState_t internal_decode_state = decode_state->getInternalDecodeState();
    nvimgcdcsImageDesc* image_desc = image->getImageDesc();
    nvimgcdcsCodeStreamDesc* code_stream_desc = code_stream->getCodeStreamDesc();
    image->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
    image->setPromise(decode_state->getPromise());

    decoder_desc_->decode(decoder_, internal_decode_state, code_stream_desc, image_desc, params);

    return future;
}

std::unique_ptr<ProcessingResultsFuture> ImageDecoder::decodeBatch(IDecodeState* decode_state_batch,
    const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
    const nvimgcdcsDecodeParams_t* params)
{
    assert(code_streams.size() == images.size());

    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();
    decode_state_batch->setPromise(std::move(results));

    if (decoder_desc_->decodeBatch == nullptr) {
        for (size_t i = 0; i < code_streams.size(); ++i) {
            IDecodeState* decode_state = images[i]->getAttachedDecodeState();
            nvimgcdcsDecodeState_t internal_decode_state = decode_state->getInternalDecodeState();
            nvimgcdcsImageDesc* image_desc = images[i]->getImageDesc();
            nvimgcdcsCodeStreamDesc* code_stream_desc = code_streams[i]->getCodeStreamDesc();
            images[i]->setIndex(i);
            images[i]->setPromise(decode_state_batch->getPromise());
            images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
            decoder_desc_->decode(
                decoder_, internal_decode_state, code_stream_desc, image_desc, params);
        }
    } else {
        std::vector<nvimgcdcsDecodeState_t> decode_states;
        std::vector<nvimgcdcsCodeStreamDesc*> code_stream_descs;
        std::vector<nvimgcdcsImageDesc*> image_descs;

        for (size_t i = 0; i < code_streams.size(); ++i) {

            code_stream_descs.push_back(code_streams[i]->getCodeStreamDesc());
            IDecodeState* decode_state = images[i]->getAttachedDecodeState();
            decode_states.push_back(decode_state->getInternalDecodeState());
            image_descs.push_back(images[i]->getImageDesc());
            images[i]->setIndex(i);
            images[i]->setPromise(decode_state_batch->getPromise());
            images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
        }

        decoder_desc_->decodeBatch(decoder_, decode_state_batch->getInternalDecodeState(),
            decode_states.data(), code_stream_descs.data(), image_descs.data(), code_streams.size(),
            params);
    }

    return future;
}

} // namespace nvimgcdcs
