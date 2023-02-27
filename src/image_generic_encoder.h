/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "iimage_encoder.h"
#include "iwork_manager.h"
#include "processing_results.h"

namespace nvimgcdcs {

class IEncodeState;
class IImage;
class ICodeStream;
class ICodecRegistry;
class ICodec;

class ImageGenericEncoder : public IImageEncoder, public IWorkManager
{
  public:
    explicit ImageGenericEncoder(ICodecRegistry* codec_registry);
    ~ImageGenericEncoder() override;
    std::unique_ptr<IEncodeState> createEncodeState(cudaStream_t cuda_stream) const override;
    std::unique_ptr<IEncodeState> createEncodeStateBatch(cudaStream_t cuda_stream) const override;
    void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) override;
    bool canEncode(nvimgcdcsImageDesc_t image, nvimgcdcsCodeStreamDesc_t code_stream,
        const nvimgcdcsEncodeParams_t* params) const override;
    void canEncode(const std::vector<IImage*>& images,
        const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params,
        std::vector<bool>* result) const;
    std::unique_ptr<ProcessingResultsFuture> encode(
        ICodeStream* code_stream, IImage* image, const nvimgcdcsEncodeParams_t* params) override;
    std::unique_ptr<ProcessingResultsFuture> encodeBatch(IEncodeState* encode_state_batch,
        const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
        const nvimgcdcsEncodeParams_t* params) override;

  private:
    class Worker;
    ImageGenericEncoder::Worker* getWorker(const ICodec* codec, int device_id);

    std::unique_ptr<IWorkManager::Work> createNewWork(
        const ProcessingResultsPromise& results, const void* params);
    void recycleWork(std::unique_ptr<IWorkManager::Work> work) override;
    void combineWork(IWorkManager::Work* target, std::unique_ptr<IWorkManager::Work> source);
    void distributeWork(std::unique_ptr<Work> work);

    std::mutex work_mutex_;
    std::unique_ptr<Work> free_work_items_;
    std::map<const ICodec*, std::unique_ptr<Worker>> workers_;
    std::set<const ICodec*> filtered_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    ICodecRegistry* codec_registry_;
};

} // namespace nvimgcdcs
