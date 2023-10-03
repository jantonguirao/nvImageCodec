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
#include "encode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"

namespace nvimgcodec {

ImageEncoder::ImageEncoder(const nvimgcodecEncoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : encoder_desc_(desc)
{
    auto ret = encoder_desc_->create(encoder_desc_->instance, &encoder_, exec_params, options);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        encoder_ = nullptr;
    }
}

ImageEncoder::~ImageEncoder()
{
    if (encoder_)
        encoder_desc_->destroy(encoder_);
}

nvimgcodecBackendKind_t ImageEncoder::getBackendKind() const 
{
    return encoder_desc_->backend_kind;
}

std::unique_ptr<IEncodeState> ImageEncoder::createEncodeStateBatch() const
{
    return std::make_unique<EncodeStateBatch>();
}

void ImageEncoder::canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    const nvimgcodecEncodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcodecProcessingStatus_t>* status) const
{
    assert(result->size() == code_streams.size());
    assert(status->size() == code_streams.size());

    // in case the encoder couldn't be created for some reason
    if (!encoder_) {
        for (size_t i = 0; i < code_streams.size(); ++i) {
            (*result)[i] = false;
        }
        return;
    }

    std::vector<nvimgcodecCodeStreamDesc_t*> cs_descs(code_streams.size());
    std::vector<nvimgcodecImageDesc_t*> im_descs(code_streams.size());
    for (size_t i = 0; i < code_streams.size(); ++i) {
        cs_descs[i] = code_streams[i]->getCodeStreamDesc();
        im_descs[i] = images[i]->getImageDesc();
    }
    encoder_desc_->canEncode(encoder_, &(*status)[0], &im_descs[0], & cs_descs[0], code_streams.size(), params);
    for (size_t i = 0; i < code_streams.size(); ++i) {
        (*result)[i] = (*status)[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    }
}

std::unique_ptr<ProcessingResultsFuture> ImageEncoder::encode(IEncodeState* encode_state_batch, const std::vector<IImage*>& images,
    const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params)
{
    assert(code_streams.size() == images.size());

    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();
    encode_state_batch->setPromise(std::move(results));

    std::vector<nvimgcodecCodeStreamDesc_t*> code_stream_descs;
    std::vector<nvimgcodecImageDesc_t*> image_descs;

    for (size_t i = 0; i < code_streams.size(); ++i) {

        code_stream_descs.push_back(code_streams[i]->getCodeStreamDesc());
        image_descs.push_back(images[i]->getImageDesc());
        images[i]->setIndex(i);
        images[i]->setPromise(encode_state_batch->getPromise());
    }

    encoder_desc_->encode(
        encoder_, image_descs.data(), code_stream_descs.data(), code_streams.size(), params);

    return future;
}

} // namespace nvimgcodec
