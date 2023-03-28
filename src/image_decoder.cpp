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
#include "decode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"
#include "log.h"
#include "processing_results.h"

namespace nvimgcdcs {

ImageDecoder::ImageDecoder(const nvimgcdcsDecoderDesc_t desc, const nvimgcdcsDecodeParams_t* params)
    : decoder_desc_(desc)
{
    decoder_desc_->create(decoder_desc_->instance, &decoder_, params);
}

ImageDecoder::~ImageDecoder()
{
    decoder_desc_->destroy(decoder_);
}

std::unique_ptr<IDecodeState> ImageDecoder::createDecodeStateBatch() const
{
    return std::make_unique<DecodeStateBatch>(decoder_desc_, decoder_);
}

void ImageDecoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    decoder_desc_->getCapabilities(decoder_, capabilities, size);
}

void ImageDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
    const nvimgcdcsDecodeParams_t* params, std::vector<bool>* result) const
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

std::unique_ptr<ProcessingResultsFuture> ImageDecoder::decode(IDecodeState* decode_state_batch,
    const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params)
{
    assert(code_streams.size() == images.size());

    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();
    decode_state_batch->setPromise(std::move(results));

    std::vector<nvimgcdcsCodeStreamDesc*> code_stream_descs;
    std::vector<nvimgcdcsImageDesc*> image_descs;

    for (size_t i = 0; i < code_streams.size(); ++i) {
        code_stream_descs.push_back(code_streams[i]->getCodeStreamDesc());
        image_descs.push_back(images[i]->getImageDesc());
        images[i]->setIndex(i);
        images[i]->setPromise(decode_state_batch->getPromise());
            if (images[i]->getProcessingStatus() != NVIMGCDCS_PROCESSING_STATUS_ERROR)
        images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
    }

    decoder_desc_->decode(
        decoder_, decode_state_batch->getInternalDecodeState(), code_stream_descs.data(), image_descs.data(), code_streams.size(), params);

    return future;
}

} // namespace nvimgcdcs
