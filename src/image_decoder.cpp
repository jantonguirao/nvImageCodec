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
    return std::make_unique<DecodeStateBatch>();
}

void ImageDecoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    decoder_desc_->getCapabilities(decoder_, capabilities, size);
}

void ImageDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
    const nvimgcdcsDecodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcdcsProcessingStatus_t>* status) const
{
    assert(result->size() == code_streams.size());
    assert(status->size() == code_streams.size());

    std::vector<nvimgcdcsCodeStreamDesc*> cs_descs(code_streams.size());
    std::vector<nvimgcdcsImageDesc*> im_descs(code_streams.size());
    for (size_t i = 0; i < code_streams.size(); ++i) {
        cs_descs[i] = code_streams[i]->getCodeStreamDesc();
        im_descs[i] = images[i]->getImageDesc();
    }
    decoder_desc_->canDecode(decoder_, &(*status)[0], &cs_descs[0], &im_descs[0], code_streams.size(), params);
    for (size_t i = 0; i < code_streams.size(); ++i) {
        (*result)[i] = (*status)[i] == NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
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
    }

    decoder_desc_->decode(
        decoder_, code_stream_descs.data(), image_descs.data(), code_streams.size(), params);

    return future;
}

} // namespace nvimgcdcs
