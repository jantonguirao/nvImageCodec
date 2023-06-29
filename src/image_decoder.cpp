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
#include <algorithm>
#include <cassert>
#include "decode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"
#include "log.h"
#include "processing_results.h"

namespace nvimgcdcs {

ImageDecoder::ImageDecoder(const nvimgcdcsDecoderDesc_t desc, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options)
    : decoder_desc_(desc)
{
    auto ret = decoder_desc_->create(decoder_desc_->instance, &decoder_, device_id, backend_params, options);
    if (NVIMGCDCS_STATUS_SUCCESS != ret) {
        decoder_ = nullptr;
    }
}

ImageDecoder::~ImageDecoder()
{
    if (decoder_)
        decoder_desc_->destroy(decoder_);
}

nvimgcdcsBackendKind_t ImageDecoder::getBackendKind() const
{
    return decoder_desc_->backend_kind;
}


std::unique_ptr<IDecodeState> ImageDecoder::createDecodeStateBatch() const
{
    return std::make_unique<DecodeStateBatch>();
}


static void sortSamples(std::vector<size_t>& order, ICodeStream *const * streams, int batch_size)
{
    order.clear();
    auto subsampling_score = [](nvimgcdcsChromaSubsampling_t subsampling) -> uint32_t {
        switch (subsampling) {
        case NVIMGCDCS_SAMPLING_444:
            return 8;
        case NVIMGCDCS_SAMPLING_422:
            return 7;
        case NVIMGCDCS_SAMPLING_420:
            return 6;
        case NVIMGCDCS_SAMPLING_440:
            return 5;
        case NVIMGCDCS_SAMPLING_411:
            return 4;
        case NVIMGCDCS_SAMPLING_410:
            return 3;
        case NVIMGCDCS_SAMPLING_GRAY:
            return 2;
        case NVIMGCDCS_SAMPLING_410V:
        default:
            return 1;
        }
    };

    using sort_elem_t = std::tuple<uint32_t, uint64_t, int>;
    std::vector<sort_elem_t> sample_meta;
    sample_meta.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        streams[i]->getImageInfo(&image_info);
        uint64_t area = image_info.plane_info[0].height * image_info.plane_info[0].width;
        // we prefer i to be in ascending order
        sample_meta.push_back(sort_elem_t{subsampling_score(image_info.chroma_subsampling), area, -i});
    }
    auto order_fn = [](const sort_elem_t& lhs, const sort_elem_t& rhs) { return lhs > rhs; };
    std::sort(sample_meta.begin(), sample_meta.end(), order_fn);

    order.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        int sample_idx = -std::get<2>(sample_meta[i]);
        order[i] = sample_idx;
    }
}


void ImageDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
    const nvimgcdcsDecodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcdcsProcessingStatus_t>* status) const
{
    assert(result->size() == code_streams.size());
    assert(status->size() == code_streams.size());

    // in case the decoder couldn't be created for some reason
    if (!decoder_) {
        for (size_t i = 0; i < code_streams.size(); ++i) {
           (*result)[i] = false;
        }
        return;
    }

    std::vector<size_t> order;
    sortSamples(order, code_streams.data(), code_streams.size());
    assert(order.size() == code_streams.size());

    std::vector<nvimgcdcsCodeStreamDesc*> cs_descs(code_streams.size());
    std::vector<nvimgcdcsImageDesc*> im_descs(code_streams.size());
    std::vector<nvimgcdcsProcessingStatus_t> internal_status(code_streams.size(), NVIMGCDCS_STATUS_NOT_INITIALIZED);
    for (size_t i = 0; i < code_streams.size(); ++i) {
        int orig_idx = order[i];
        cs_descs[i] = code_streams[orig_idx]->getCodeStreamDesc();
        im_descs[i] = images[orig_idx]->getImageDesc();
    }
    decoder_desc_->canDecode(decoder_, &internal_status[0], &cs_descs[0], &im_descs[0], code_streams.size(), params);
    for (size_t i = 0; i < code_streams.size(); ++i) {
        int orig_idx = order[i];
        (*status)[orig_idx] = internal_status[i];
        (*result)[orig_idx] = internal_status[i] == NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
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

    std::vector<size_t> order;
    sortSamples(order, code_streams.data(), code_streams.size());
    assert(order.size() == code_streams.size());

    std::vector<nvimgcdcsCodeStreamDesc*> code_stream_descs(code_streams.size());
    std::vector<nvimgcdcsImageDesc*> image_descs(code_streams.size());

    for (size_t i = 0; i < code_streams.size(); ++i) {
        int orig_idx = order[i];
        code_stream_descs[i] = code_streams[orig_idx]->getCodeStreamDesc();
        image_descs[i] = images[orig_idx]->getImageDesc();
        images[i]->setIndex(orig_idx);
        images[i]->setPromise(decode_state_batch->getPromise());
    }

    decoder_desc_->decode(
        decoder_, code_stream_descs.data(), image_descs.data(), code_streams.size(), params);

    return future;
}

const char* ImageDecoder::decoderId() const {
    return decoder_desc_->id;
}

} // namespace nvimgcdcs
